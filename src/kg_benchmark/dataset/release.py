from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin
from urllib.request import urlopen

import ijson
from jsonschema import Draft202012Validator

from kg_benchmark.dataset.gates import (
    CANONICAL_FILES,
    REPO_ARTIFACTS,
    SCHEMA_ARTIFACTS,
    WORK_ARTIFACTS,
    DatasetGateError,
    validate_dataset_semantics,
)
from kg_benchmark.methodology import require_frozen_methodology

EMPTY_ALLOWED_ROLES = {"replacements"}


def _json_default(value: Any) -> int | float:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _artifact_record(root: Path, relative_path: str, *, allow_empty: bool = False) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file() or (path.stat().st_size == 0 and not allow_empty):
        raise ValueError(f"Required dataset artifact is missing or empty: {path}")
    record: dict[str, Any] = {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        record["records"] = _jsonl_count(path)
    return record


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"Canonical JSONL row for {path} is not an object.")
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=_json_default,
                    )
                    + "\n"
                )
                count += 1
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _array_items(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


def _object_rows(path: Path, *, value_field: str) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for record_id, payload in ijson.kvitems(handle, ""):
            if not isinstance(record_id, str) or not isinstance(payload, dict):
                raise ValueError(f"Invalid keyed record in {path}.")
            yield {"id" if value_field == "world_state" else "qid": record_id, value_field: payload}


def canonicalize_acquisition(*, acquisition_dir: Path, work_dir: Path) -> dict[str, int]:
    """Convert legacy construction outputs into the one canonical JSONL representation."""
    inputs = {
        "popularity": acquisition_dir / "00_entity_popularity.json",
        "candidates": acquisition_dir / "01_repair_candidates.json",
        "repairs": acquisition_dir / "02_wikidata_repairs.json",
        "world_state": acquisition_dir / "03_world_state.json",
    }
    for role, path in inputs.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Cannot canonicalize missing {role} artifact: {path}")
    counts = {
        "popularity": _write_rows(
            work_dir / CANONICAL_FILES["popularity"],
            _object_rows(inputs["popularity"], value_field="popularity"),
        ),
        "candidates": _write_rows(work_dir / CANONICAL_FILES["candidates"], _array_items(inputs["candidates"])),
        "repairs": _write_rows(work_dir / CANONICAL_FILES["repairs"], _array_items(inputs["repairs"])),
        "world_state": _write_rows(
            work_dir / CANONICAL_FILES["world_state"],
            _object_rows(inputs["world_state"], value_field="world_state"),
        ),
    }
    return counts


def canonicalize_case_context_references(cases_path: Path) -> int:
    """Replace construction-machine Stage 3 paths with the canonical release-relative reference."""

    def rows() -> Iterable[dict[str, Any]]:
        with cases_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid classified case JSON at {cases_path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Classified case at {cases_path}:{line_number} is not an object.")
                context = row.get("context_ref")
                if not isinstance(context, dict):
                    raise ValueError(f"Classified case at {cases_path}:{line_number} has no context_ref object.")
                context["world_state_path"] = CANONICAL_FILES["world_state"]
                yield row

    return _write_rows(cases_path, rows())


def write_source_provenance(
    *,
    acquisition_dir: Path,
    work_dir: Path,
    dump_path: Path,
    acquisition_config_path: Path,
    cache_dir: Path,
) -> Path:
    source_paths = {
        "popularity": acquisition_dir / "00_entity_popularity.json",
        "candidates": acquisition_dir / "01_repair_candidates.json",
        "repairs": acquisition_dir / "02_wikidata_repairs.json",
        "stage2_exclusions": acquisition_dir / "02_stage2_exclusions.json",
        "world_state": acquisition_dir / "03_world_state.json",
        "wikidata_dump": dump_path,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Cannot bind source provenance; missing artifacts: {missing}")
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not acquisition_config_path.is_file() or acquisition_config_path.stat().st_size == 0:
        raise FileNotFoundError(f"Acquisition configuration is missing: {acquisition_config_path}")
    acquisition_config = json.loads(acquisition_config_path.read_text(encoding="utf-8"))
    if not isinstance(acquisition_config, dict) or acquisition_config.get("status") != "complete":
        raise ValueError("Acquisition configuration must record a completed acquisition.")
    cache_files: list[dict[str, Any]] = []
    cache_aggregate = hashlib.sha256()
    total_cache_bytes = 0
    if cache_dir.is_dir():
        for path in sorted(candidate for candidate in cache_dir.rglob("*") if candidate.is_file()):
            relative = path.relative_to(cache_dir).as_posix()
            size = path.stat().st_size
            digest = sha256_file(path)
            cache_files.append({"path": relative, "bytes": size, "sha256": digest})
            total_cache_bytes += size
            cache_aggregate.update(relative.encode("utf-8"))
            cache_aggregate.update(b"\0")
            cache_aggregate.update(str(size).encode("ascii"))
            cache_aggregate.update(b"\0")
            cache_aggregate.update(digest.encode("ascii"))
            cache_aggregate.update(b"\0")
    source_records: dict[str, dict[str, Any]] = {}
    for role, path in source_paths.items():
        record: dict[str, Any] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if role == "stage2_exclusions":
            record["records"] = sum(1 for _ in _array_items(path))
        source_records[role] = record

    payload: dict[str, Any] = {
        "manifest_type": "source_provenance",
        "manifest_version": 2,
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_revision": git_revision,
        "sources": source_records,
        "acquisition_config": {
            "path": str(acquisition_config_path.resolve()),
            "bytes": acquisition_config_path.stat().st_size,
            "sha256": sha256_file(acquisition_config_path),
            "config": acquisition_config,
        },
        "cache_provenance": {
            "root": str(cache_dir.resolve()),
            "file_count": len(cache_files),
            "total_bytes": total_cache_bytes,
            "aggregate_sha256": cache_aggregate.hexdigest(),
            "files": cache_files,
        },
    }
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "source-provenance.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(payload)
    destination = work_dir / "source-provenance.json"
    _write_json_atomic(destination, payload)
    return destination


def build_dataset_manifest(
    root: Path,
) -> dict[str, Any]:
    artifacts = {
        role: _artifact_record(root, path, allow_empty=role in EMPTY_ALLOWED_ROLES)
        for role, path in CANONICAL_FILES.items()
    }
    protocol = artifacts["protocol"]
    methodology_lock = json.loads((root / CANONICAL_FILES["methodology_lock"]).read_text(encoding="utf-8"))
    return {
        "manifest_type": "kg_benchmark_dataset",
        "manifest_version": 2,
        "dataset_id": "wikidata-repair-eval-paper",
        "status": "final",
        "protocol": {
            "path": protocol["path"],
            "sha256": protocol["sha256"],
        },
        "methodology": {
            "lock": {
                "path": artifacts["methodology_lock"]["path"],
                "sha256": artifacts["methodology_lock"]["sha256"],
            },
            "freeze_scope_sha256": methodology_lock["freeze_scope_sha256"],
            "source_git_revision": methodology_lock["source_git_revision"],
        },
        "source_provenance": {
            "path": artifacts["source_provenance"]["path"],
            "sha256": artifacts["source_provenance"]["sha256"],
        },
        "lineage": {
            "path": artifacts["lineage"]["path"],
            "sha256": artifacts["lineage"]["sha256"],
        },
        "release_validation": {
            "semantic_gates_passed": True,
            "manifest_byte_reproduction_passed": True,
        },
        "artifacts": artifacts,
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def promote_dataset(
    *,
    work_dir: Path,
    dataset_dir: Path,
    protocol_path: Path,
    source_provenance_path: Path | None = None,
    lineage_manifest_path: Path | None = None,
    repo_root: Path = Path("."),
    methodology_check: Any = require_frozen_methodology,
) -> dict[str, Any]:
    """Atomically promote a complete work tree into the one final dataset directory."""
    if dataset_dir.exists():
        raise FileExistsError(
            f"Final dataset already exists at {dataset_dir}; it is immutable and cannot be overwritten."
        )
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work directory does not exist: {work_dir}")
    repo_root = repo_root.resolve()
    methodology = methodology_check(repo_root)
    canonical_protocol = repo_root / REPO_ARTIFACTS["protocol"][0]
    if protocol_path.resolve() != canonical_protocol.resolve() or not canonical_protocol.is_file():
        raise ValueError(f"Promotion requires the canonical frozen protocol: {canonical_protocol}")
    if methodology.get("files", {}).get("paper/protocol.json") != sha256_file(canonical_protocol):
        raise ValueError("Frozen methodology lock does not bind the promoted protocol hash.")
    lock_path = repo_root / REPO_ARTIFACTS["methodology_lock"][0]
    if not lock_path.is_file():
        raise FileNotFoundError(f"Frozen methodology lock is missing: {lock_path}")

    if source_provenance_path is None:
        raise ValueError("A source-provenance manifest is required for final dataset promotion.")
    canonical_provenance = work_dir / WORK_ARTIFACTS["source_provenance"]
    if source_provenance_path.resolve() != canonical_provenance.resolve():
        raise ValueError(f"Promotion requires canonical source provenance at {canonical_provenance}.")
    lineage_manifest_path = lineage_manifest_path or work_dir / WORK_ARTIFACTS["lineage"]
    canonical_lineage = work_dir / WORK_ARTIFACTS["lineage"]
    if lineage_manifest_path.resolve() != canonical_lineage.resolve():
        raise ValueError(f"Promotion requires canonical lineage at {canonical_lineage}.")

    temporary = dataset_dir.with_name(f".{dataset_dir.name}.promoting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        promotion_sources = {
            **{role: work_dir / relative_path for role, relative_path in WORK_ARTIFACTS.items()},
            **{
                role: repo_root / source_path
                for role, (source_path, _) in REPO_ARTIFACTS.items()
            },
            **{
                role: repo_root / "schemas" / filename
                for role, filename in SCHEMA_ARTIFACTS.items()
            },
        }
        for role, relative_path in CANONICAL_FILES.items():
            source = promotion_sources[role]
            if not source.is_file() or (source.stat().st_size == 0 and role not in EMPTY_ALLOWED_ROLES):
                raise ValueError(f"Required work artifact is missing or empty: {source}")
            destination = temporary / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest = build_dataset_manifest(temporary)
        _write_json_atomic(temporary / "manifest.json", manifest)
        verify_dataset(temporary, check_external_sources=True)
        reproduced = build_dataset_manifest(temporary)
        if _manifest_bytes(reproduced) != (temporary / "manifest.json").read_bytes():
            raise ValueError("Dataset manifest did not reproduce byte-for-byte before promotion.")
        os.replace(temporary, dataset_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _iter_manifest_artifacts(manifest: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Dataset manifest has no artifacts object.")
    for role, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid artifact record for {role}.")
        yield str(role), record


def verify_dataset(dataset_dir: Path, *, check_external_sources: bool = False) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = dataset_dir / CANONICAL_FILES["schema_dataset_manifest"]
    errors = list(
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).iter_errors(manifest)
    )
    if errors:
        raise DatasetGateError(f"Dataset manifest fails its published schema: {errors[0].message}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(CANONICAL_FILES):
        raise ValueError("Dataset manifest roles do not exactly match the canonical release roles.")
    checked = 0
    for role, record in _iter_manifest_artifacts(manifest):
        relative_path = record.get("path")
        if not isinstance(relative_path, str):
            raise ValueError(f"Artifact {role} has no path.")
        if relative_path != CANONICAL_FILES[role]:
            raise ValueError(f"Artifact {role} has a noncanonical path.")
        path = dataset_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Artifact {role} is missing: {path}")
        if path.stat().st_size != record.get("bytes"):
            raise ValueError(f"Artifact {role} byte-size mismatch.")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Artifact {role} SHA-256 mismatch.")
        if path.suffix == ".jsonl" and _jsonl_count(path) != record.get("records"):
            raise ValueError(f"Artifact {role} record-count mismatch.")
        checked += 1
    semantics = validate_dataset_semantics(dataset_dir, check_external_sources=check_external_sources)
    reproduced = build_dataset_manifest(dataset_dir)
    if _manifest_bytes(reproduced) != manifest_path.read_bytes():
        raise ValueError("Dataset manifest is not byte-reproducible from the published artifacts.")
    return {
        "valid": True,
        "checked_artifacts": checked,
        "dataset_id": manifest.get("dataset_id"),
        "manifest_sha256": sha256_file(manifest_path),
        "semantic_validation": semantics,
        "manifest_byte_reproduced": True,
    }


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as handle:  # noqa: S310 - explicit release URL
        shutil.copyfileobj(response, handle)


def fetch_dataset(
    *,
    manifest_url: str,
    dataset_dir: Path,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically fetch the one canonical dataset release from an external distribution."""
    if dataset_dir.exists():
        raise FileExistsError(f"Dataset already exists at {dataset_dir}; refusing to overwrite it.")
    temporary = dataset_dir.with_name(f".{dataset_dir.name}.fetching")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        manifest_path = temporary / "manifest.json"
        _download(manifest_url, manifest_path)
        if manifest_sha256 is not None and sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("Downloaded dataset manifest SHA-256 mismatch.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(CANONICAL_FILES):
            raise ValueError("Downloaded manifest does not contain the canonical artifact roles.")
        for role, relative_path in CANONICAL_FILES.items():
            record = artifacts.get(role)
            if not isinstance(record, dict) or record.get("path") != relative_path:
                raise ValueError(f"Downloaded manifest has a noncanonical {role} artifact.")
            _download(urljoin(manifest_url, relative_path), temporary / relative_path)
        result = verify_dataset(temporary)
        os.replace(temporary, dataset_dir)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
