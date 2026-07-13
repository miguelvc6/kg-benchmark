#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_STATUSES = {"bundled", "published", "unresolved"}


class ManifestError(ValueError):
    """Raised when a distribution manifest cannot be used safely."""


@dataclass(frozen=True)
class ArtifactResult:
    artifact_id: str
    status: str
    detail: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ManifestError(f"Artifact path must be a safe relative path: {relative_path!r}")
    target = (root / relative).resolve()
    root = root.resolve()
    if target == root or root not in target.parents:
        raise ManifestError(f"Artifact path escapes acquisition root: {relative_path!r}")
    return target


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ManifestError("Distribution manifest must be a JSON object.")
    if manifest.get("manifest_type") != "kg_benchmark_distribution":
        raise ManifestError("Unsupported manifest_type.")
    if manifest.get("manifest_version") != 1:
        raise ManifestError("Unsupported manifest_version.")
    if manifest.get("status") not in {"draft", "published"}:
        raise ManifestError("Manifest status must be 'draft' or 'published'.")

    release = manifest.get("release")
    if not isinstance(release, dict):
        raise ManifestError("Manifest release metadata is required.")
    for key in ("benchmark_version", "repository_url", "release_manifest_url", "release_manifest_sha256"):
        if key not in release:
            raise ManifestError(f"Release metadata is missing {key!r}.")
    release_digest = release["release_manifest_sha256"]
    if release_digest is not None and not SHA256_RE.fullmatch(release_digest):
        raise ManifestError("release_manifest_sha256 must be null or a lowercase SHA-256 digest.")
    if manifest["status"] == "published":
        if not release.get("release_manifest_url") or release_digest is None:
            raise ManifestError("Published distributions must bind an immutable release manifest URL and hash.")

    sources = manifest.get("source_snapshots")
    if not isinstance(sources, list) or not sources:
        raise ManifestError("At least one source snapshot declaration is required.")
    source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
            raise ManifestError("Every source snapshot requires a string source_id.")
        source_id = source["source_id"]
        if source_id in source_ids:
            raise ManifestError(f"Duplicate source_id: {source_id}")
        source_ids.add(source_id)
        if source.get("status") not in {"resolved", "unresolved"}:
            raise ManifestError(f"Invalid source snapshot status for {source_id}.")
        if source.get("status") == "resolved" and not source.get("snapshot_at_utc"):
            raise ManifestError(f"Resolved source snapshot {source_id} lacks snapshot_at_utc.")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("At least one artifact declaration is required.")
    artifact_ids: set[str] = set()
    artifact_paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ManifestError("Artifact declarations must be JSON objects.")
        artifact_id = artifact.get("artifact_id")
        relative_path = artifact.get("path")
        status = artifact.get("status")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ManifestError("Every artifact requires a non-empty artifact_id.")
        if artifact_id in artifact_ids:
            raise ManifestError(f"Duplicate artifact_id: {artifact_id}")
        artifact_ids.add(artifact_id)
        if not isinstance(relative_path, str):
            raise ManifestError(f"Artifact {artifact_id} requires a path.")
        _safe_target(Path.cwd(), relative_path)
        if relative_path in artifact_paths:
            raise ManifestError(f"Duplicate artifact path: {relative_path}")
        artifact_paths.add(relative_path)
        if status not in ARTIFACT_STATUSES:
            raise ManifestError(f"Invalid status for artifact {artifact_id}: {status!r}")
        urls = artifact.get("urls")
        digest = artifact.get("sha256")
        size = artifact.get("size_bytes")
        if not isinstance(urls, list) or not all(isinstance(url, str) and url for url in urls):
            raise ManifestError(f"Artifact {artifact_id} urls must be a list of non-empty strings.")
        if digest is not None and not SHA256_RE.fullmatch(digest):
            raise ManifestError(f"Artifact {artifact_id} has an invalid SHA-256 digest.")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise ManifestError(f"Artifact {artifact_id} has an invalid size_bytes value.")
        if status == "published" and (not urls or digest is None or size is None):
            raise ManifestError(f"Published artifact {artifact_id} requires URLs, SHA-256, and size.")
        if status == "published" and not artifact.get("license_id"):
            raise ManifestError(f"Published artifact {artifact_id} requires an explicit license_id.")
        if status == "bundled" and (digest is None or size is None):
            raise ManifestError(f"Bundled artifact {artifact_id} requires SHA-256 and size.")
        if status == "unresolved" and (urls or digest is not None or size is not None):
            raise ManifestError(f"Unresolved artifact {artifact_id} must not contain guessed locators or checksums.")
        linked_sources = artifact.get("source_snapshot_ids")
        if not isinstance(linked_sources, list) or not all(isinstance(item, str) for item in linked_sources):
            raise ManifestError(f"Artifact {artifact_id} source_snapshot_ids must be a list of strings.")
        unknown_sources = set(linked_sources) - source_ids
        if unknown_sources:
            raise ManifestError(
                f"Artifact {artifact_id} references unknown source snapshot(s): {', '.join(sorted(unknown_sources))}"
            )
        if manifest["status"] == "published" and status == "unresolved" and artifact.get("required", True):
            raise ManifestError(f"Published distribution contains unresolved required artifact {artifact_id}.")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read distribution manifest {path}: {exc}") from exc
    return validate_manifest(payload)


def _selected_artifacts(manifest: dict[str, Any], artifact_ids: Iterable[str] | None) -> list[dict[str, Any]]:
    artifacts = manifest["artifacts"]
    if not artifact_ids:
        return list(artifacts)
    wanted = set(artifact_ids)
    known = {artifact["artifact_id"] for artifact in artifacts}
    unknown = wanted - known
    if unknown:
        raise ManifestError(f"Unknown artifact id(s): {', '.join(sorted(unknown))}")
    return [artifact for artifact in artifacts if artifact["artifact_id"] in wanted]


def verify_artifacts(
    manifest: dict[str, Any],
    *,
    root: str | Path,
    artifact_ids: Iterable[str] | None = None,
    allow_unresolved: bool = False,
) -> list[ArtifactResult]:
    validate_manifest(manifest)
    root_path = Path(root).resolve()
    results: list[ArtifactResult] = []
    for artifact in _selected_artifacts(manifest, artifact_ids):
        artifact_id = artifact["artifact_id"]
        if artifact["status"] == "unresolved":
            status = "skipped" if allow_unresolved or not artifact.get("required", True) else "error"
            results.append(ArtifactResult(artifact_id, status, "publication locator and checksum unresolved"))
            continue
        target = _safe_target(root_path, artifact["path"])
        if not target.is_file():
            results.append(ArtifactResult(artifact_id, "error", f"missing: {target}"))
            continue
        actual_size = target.stat().st_size
        if actual_size != artifact["size_bytes"]:
            detail = f"size mismatch: expected {artifact['size_bytes']}, got {actual_size}"
            results.append(
                ArtifactResult(artifact_id, "error", detail)
            )
            continue
        actual_digest = sha256_file(target)
        if actual_digest != artifact["sha256"]:
            results.append(ArtifactResult(artifact_id, "error", "SHA-256 mismatch"))
            continue
        results.append(ArtifactResult(artifact_id, "verified", str(target)))
    return results


def _copy_url_to_temp(url: str, target: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"https", "file"}:
        raise ManifestError(f"Unsupported artifact URL scheme {parsed.scheme!r}; use HTTPS or file URLs.")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def fetch_artifacts(
    manifest: dict[str, Any],
    *,
    root: str | Path,
    artifact_ids: Iterable[str] | None = None,
    overwrite: bool = False,
    allow_unresolved: bool = False,
) -> list[ArtifactResult]:
    validate_manifest(manifest)
    root_path = Path(root).resolve()
    results: list[ArtifactResult] = []
    for artifact in _selected_artifacts(manifest, artifact_ids):
        artifact_id = artifact["artifact_id"]
        if artifact["status"] == "unresolved":
            status = "skipped" if allow_unresolved or not artifact.get("required", True) else "error"
            results.append(ArtifactResult(artifact_id, status, "publication locator and checksum unresolved"))
            continue
        target = _safe_target(root_path, artifact["path"])
        if target.exists() and not overwrite:
            verification = verify_artifacts(manifest, root=root_path, artifact_ids=[artifact_id])
            if verification[0].status == "verified":
                results.append(ArtifactResult(artifact_id, "verified", f"already present: {target}"))
            else:
                detail = "target exists but does not verify; use --overwrite"
                results.append(ArtifactResult(artifact_id, "error", detail))
            continue
        if artifact["status"] == "bundled" and not target.exists():
            results.append(ArtifactResult(artifact_id, "error", f"bundled artifact missing: {target}"))
            continue

        errors: list[str] = []
        installed = False
        for url in artifact["urls"]:
            temporary: Path | None = None
            try:
                temporary = _copy_url_to_temp(url, target)
                actual_size = temporary.stat().st_size
                actual_digest = sha256_file(temporary)
                if actual_size != artifact["size_bytes"]:
                    raise ManifestError(f"size mismatch: expected {artifact['size_bytes']}, got {actual_size}")
                if actual_digest != artifact["sha256"]:
                    raise ManifestError("SHA-256 mismatch")
                os.replace(temporary, target)
                installed = True
                results.append(ArtifactResult(artifact_id, "fetched", f"installed {target} from {url}"))
                break
            except (OSError, urllib.error.URLError, ManifestError) as exc:
                errors.append(f"{url}: {exc}")
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if not installed:
            results.append(ArtifactResult(artifact_id, "error", "; ".join(errors) or "no usable URL"))
    return results


def _print_results(results: Sequence[ArtifactResult]) -> None:
    for result in results:
        print(f"{result.status:8} {result.artifact_id}: {result.detail}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify checksum-bound benchmark release artifacts.")
    parser.add_argument("--manifest", default="release/artifact_distribution.template.json")
    parser.add_argument("--root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "verify", "fetch"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--artifact", action="append", default=[])
        subparser.add_argument("--allow-unresolved", action="store_true")
        if command == "fetch":
            subparser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "status":
            for artifact in _selected_artifacts(manifest, args.artifact):
                print(f"{artifact['status']:10} {artifact['artifact_id']}: {artifact['path']}")
            return 0
        if args.command == "verify":
            results = verify_artifacts(
                manifest,
                root=args.root,
                artifact_ids=args.artifact,
                allow_unresolved=args.allow_unresolved,
            )
        else:
            results = fetch_artifacts(
                manifest,
                root=args.root,
                artifact_ids=args.artifact,
                overwrite=args.overwrite,
                allow_unresolved=args.allow_unresolved,
            )
        _print_results(results)
        return 1 if any(result.status == "error" for result in results) else 0
    except ManifestError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
