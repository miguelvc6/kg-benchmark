from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin
from urllib.request import urlopen

import ijson
from jsonschema import Draft202012Validator

CANONICAL_FILES = {
    "popularity": "source/popularity.jsonl",
    "candidates": "source/candidates.jsonl",
    "repairs": "source/repairs.jsonl",
    "world_state": "source/world-state.jsonl",
    "cases": "cases.jsonl",
    "dispositions": "audit/dispositions.jsonl",
    "audit_summary": "audit/summary.json",
    "eligibility_order": "selections/eligibility-order.jsonl",
    "support_bank": "selections/support-bank.json",
    "main_selection": "selections/main-1200.json",
    "api_selection": "selections/azure-600.json",
}


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


def _artifact_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file() or path.stat().st_size == 0:
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
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
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


def write_source_provenance(
    *,
    acquisition_dir: Path,
    work_dir: Path,
    dump_path: Path,
    acquisition_config_path: Path | None = None,
) -> Path:
    source_paths = {
        "popularity": acquisition_dir / "00_entity_popularity.json",
        "candidates": acquisition_dir / "01_repair_candidates.json",
        "repairs": acquisition_dir / "02_wikidata_repairs.json",
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
    payload: dict[str, Any] = {
        "manifest_type": "source_provenance",
        "manifest_version": 1,
        "git_revision": git_revision,
        "sources": {
            role: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for role, path in source_paths.items()
        },
    }
    if acquisition_config_path is not None and acquisition_config_path.is_file():
        payload["acquisition_config"] = {
            "path": str(acquisition_config_path),
            "sha256": sha256_file(acquisition_config_path),
            "config": json.loads(acquisition_config_path.read_text(encoding="utf-8")),
        }
    destination = work_dir / "source-provenance.json"
    _write_json_atomic(destination, payload)
    return destination


def build_dataset_manifest(
    root: Path,
    *,
    protocol_path: Path,
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = {role: _artifact_record(root, path) for role, path in CANONICAL_FILES.items()}
    return {
        "manifest_type": "kg_benchmark_dataset",
        "manifest_version": 1,
        "dataset_id": "wikidata-repair-eval-paper",
        "status": "final",
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "source_provenance": source_provenance or {},
        "artifacts": artifacts,
    }


def promote_dataset(
    *,
    work_dir: Path,
    dataset_dir: Path,
    protocol_path: Path,
    source_provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically promote a complete work tree into the one final dataset directory."""
    if dataset_dir.exists():
        raise FileExistsError(
            f"Final dataset already exists at {dataset_dir}; it is immutable and cannot be overwritten."
        )
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work directory does not exist: {work_dir}")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol does not exist: {protocol_path}")

    if source_provenance_path is None:
        raise ValueError("A source-provenance manifest is required for final dataset promotion.")
    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    if not isinstance(source_provenance, dict) or not source_provenance:
        raise ValueError("Source provenance must be a nonempty JSON object.")

    temporary = dataset_dir.with_name(f".{dataset_dir.name}.promoting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for relative_path in CANONICAL_FILES.values():
            source = work_dir / relative_path
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"Required work artifact is missing or empty: {source}")
            destination = temporary / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest = build_dataset_manifest(
            temporary,
            protocol_path=protocol_path,
            source_provenance=source_provenance,
        )
        _write_json_atomic(temporary / "manifest.json", manifest)
        verify_dataset(temporary)
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


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "dataset-manifest.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(manifest)
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
    protocol = manifest.get("protocol")
    protocol_path = protocol.get("path") if isinstance(protocol, dict) else None
    protocol_sha256 = protocol.get("sha256") if isinstance(protocol, dict) else None
    if not isinstance(protocol_path, str) or not isinstance(protocol_sha256, str):
        raise ValueError("Dataset manifest has invalid protocol provenance.")
    bound_protocol = dataset_dir.parent / protocol_path
    if bound_protocol.is_file() and sha256_file(bound_protocol) != protocol_sha256:
        raise ValueError("Dataset protocol SHA-256 mismatch.")
    return {"valid": True, "checked_artifacts": checked, "dataset_id": manifest.get("dataset_id")}


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
