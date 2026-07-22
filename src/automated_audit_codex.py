"""Codex-assisted review and conservative finalization for automated audit packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from kg_benchmark.resources import artifact_path

CODEX_REVIEW_SCHEMA = json.loads(
    artifact_path("schemas/automated-audit-review.schema.json").read_text(encoding="utf-8")
)

REVIEWS_FILENAME = "codex_reviews.jsonl"
RUN_FILENAME = "codex_review_run.json"
DISAGREEMENTS_FILENAME = "audit_disagreements.jsonl"
DISPOSITIONS_FILENAME = "audit_dispositions.jsonl"
FINAL_JSON_FILENAME = "automated_audit_final.json"
FINAL_MD_FILENAME = "automated_audit_final.md"

RunCommand = Callable[..., subprocess.CompletedProcess[str]]


class AutomatedAuditError(ValueError):
    """Raised when audit inputs or model output violate the frozen contract."""


class AutomatedAuditBatchError(AutomatedAuditError):
    """Raised with a serializable record after a review batch exhausts its retries."""

    def __init__(self, message: str, batch_record: dict[str, Any]) -> None:
        super().__init__(message)
        self.batch_record = batch_record


@dataclass(frozen=True)
class AuditPacket:
    packet_id: str
    case_id: str
    audit_dimension: str
    payload: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_state() -> dict[str, Any]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return {"commit": None, "dirty": None}
    if commit_result.returncode != 0 or status_result.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": commit_result.stdout.strip() or None, "dirty": bool(status_result.stdout.strip())}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomatedAuditError(f"Could not read JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise AutomatedAuditError(f"Expected a JSON object in {source}.")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AutomatedAuditError(f"Invalid JSON at {source}:{line_number}.") from exc
                if not isinstance(value, dict):
                    raise AutomatedAuditError(f"Expected an object at {source}:{line_number}.")
                rows.append(value)
    except OSError as exc:
        raise AutomatedAuditError(f"Could not read JSONL {source}: {exc}") from exc
    return rows


def _output_dir(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    value = manifest.get("output_dir")
    if value is None and isinstance(manifest.get("outputs"), dict):
        value = manifest["outputs"].get("directory")
    if not isinstance(value, str) or not value.strip():
        raise AutomatedAuditError("Audit manifest must declare output_dir or outputs.directory.")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _model_name(manifest: dict[str, Any], explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    codex = manifest.get("codex")
    value = codex.get("model") if isinstance(codex, dict) else manifest.get("model")
    if not isinstance(value, str) or not value.strip():
        raise AutomatedAuditError("Codex model must be explicit in the manifest or model argument.")
    return value.strip()


def _manifest_artifact(manifest_path: Path, manifest: dict[str, Any], name: str) -> Path:
    artifacts = manifest.get("artifacts")
    entry = artifacts.get(name) if isinstance(artifacts, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise AutomatedAuditError(f"Audit manifest does not bind artifact {name!r}.")
    path = Path(entry["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise AutomatedAuditError(f"Manifest artifact {name!r} is missing: {path}")
    expected = entry.get("sha256")
    observed = _sha256_file(path)
    if expected != observed:
        raise AutomatedAuditError(f"Manifest artifact {name!r} does not match its SHA-256 binding.")
    return path


def _packets(path: str | Path, audit_dimension: str) -> list[AuditPacket]:
    packets: list[AuditPacket] = []
    for row_number, row in enumerate(_load_jsonl(path), start=1):
        packet_id = row.get("packet_id", row.get("review_id"))
        case_id = row.get("case_id", row.get("blinded_case_id", packet_id))
        if not isinstance(packet_id, str) or not packet_id.strip():
            raise AutomatedAuditError(f"{path}:{row_number} requires a non-empty packet_id.")
        if not isinstance(case_id, str) or not case_id.strip():
            raise AutomatedAuditError(f"{path}:{row_number} requires a non-empty case_id.")
        packets.append(AuditPacket(packet_id.strip(), case_id.strip(), audit_dimension, row))
    return packets


def _validate_unique_packet_ids(packets: Sequence[AuditPacket]) -> None:
    counts = Counter(packet.packet_id for packet in packets)
    duplicates = sorted(packet_id for packet_id, count in counts.items() if count > 1)
    if duplicates:
        raise AutomatedAuditError(f"Duplicate packet_id values: {', '.join(duplicates)}")
    if not packets:
        raise AutomatedAuditError("No construct or temporal audit packets were provided.")


def _validate_review_payload(value: Any, expected: Sequence[AuditPacket]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"reviews"} or not isinstance(value["reviews"], list):
        raise AutomatedAuditError("Codex result must be an object containing only a reviews array.")
    expected_by_id = {packet.packet_id: packet for packet in expected}
    reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_keys = {"packet_id", "audit_dimension", "verdict", "rationale", "evidence"}
    for index, review in enumerate(value["reviews"], start=1):
        if not isinstance(review, dict) or set(review) != allowed_keys:
            raise AutomatedAuditError(f"Review {index} has missing or unsupported fields.")
        packet_id = review.get("packet_id")
        if packet_id not in expected_by_id:
            raise AutomatedAuditError(f"Review {index} has unexpected packet_id {packet_id!r}.")
        if packet_id in seen:
            raise AutomatedAuditError(f"Duplicate review packet_id: {packet_id}")
        seen.add(packet_id)
        packet = expected_by_id[packet_id]
        dimension = review.get("audit_dimension")
        verdict = review.get("verdict")
        if dimension != packet.audit_dimension:
            raise AutomatedAuditError(f"Review {packet_id} changed its audit dimension.")
        allowed_verdicts = {
            "construct": {"pass", "concern", "uncertain"},
            "temporal": {"pass", "uncertain", "suspected_temporal_leakage"},
        }
        if verdict not in allowed_verdicts[dimension]:
            raise AutomatedAuditError(f"Review {packet_id} has invalid {dimension} verdict {verdict!r}.")
        rationale = review.get("rationale")
        evidence = review.get("evidence")
        if not isinstance(rationale, str) or not rationale.strip():
            raise AutomatedAuditError(f"Review {packet_id} requires a rationale.")
        if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
            raise AutomatedAuditError(f"Review {packet_id} evidence must be a list of non-empty strings.")
        if _contains_forbidden_label(review):
            raise AutomatedAuditError(f"Review {packet_id} attempted to emit EXTERNAL_CONFIRMED.")
        reviews.append(
            {
                "packet_id": packet_id,
                "case_id": packet.case_id,
                "audit_dimension": dimension,
                "verdict": verdict,
                "rationale": rationale.strip(),
                "evidence": list(dict.fromkeys(item.strip() for item in evidence)),
            }
        )
    missing = sorted(set(expected_by_id) - seen)
    if missing:
        raise AutomatedAuditError(f"Codex result omitted packet_id values: {', '.join(missing)}")
    return reviews


def _contains_forbidden_label(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_forbidden_label(key) or _contains_forbidden_label(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_label(item) for item in value)
    return isinstance(value, str) and "EXTERNAL_CONFIRMED" in value


def _review_prompt(batch: Sequence[AuditPacket]) -> str:
    packet_payload = [
        {
            "packet_id": packet.packet_id,
            "case_id": packet.case_id,
            "audit_dimension": packet.audit_dimension,
            "packet": packet.payload,
        }
        for packet in batch
    ]
    return (
        "Review the audit packets below using only evidence present in each packet. "
        "For construct packets, return pass, concern, or uncertain. "
        "For temporal packets, return pass, uncertain, or suspected_temporal_leakage. "
        "Do not propose, change, or confirm benchmark labels; do not infer external evidence. "
        "Return exactly one schema-conforming review for every packet_id and no additional text.\n\n"
        + json.dumps(packet_payload, ensure_ascii=True, sort_keys=True)
    )


def _codex_version(run_command: RunCommand) -> str:
    completed = run_command(
        ["codex", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AutomatedAuditError(f"Could not determine Codex version: {completed.stderr.strip()}")
    version = completed.stdout.strip()
    if not version:
        raise AutomatedAuditError("Codex version command returned no version.")
    return version


def _run_batch(
    *,
    batch_id: str,
    batch: Sequence[AuditPacket],
    model: str,
    schema_path: Path,
    output_dir: Path,
    retries: int,
    timeout_seconds: float,
    run_command: RunCommand,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = _review_prompt(batch)
    attempt_records: list[dict[str, Any]] = []
    last_error = "unknown failure"
    with tempfile.TemporaryDirectory(prefix=f"kg-audit-{batch_id}.") as isolated_directory:
        command = [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--cd",
            isolated_directory,
            "--sandbox",
            "read-only",
            "--config",
            'approval_policy="never"',
            "--model",
            model,
            "--output-schema",
            str(schema_path),
            "--color",
            "never",
            "-",
        ]
        for attempt in range(1, retries + 2):
            with tempfile.NamedTemporaryFile(
                prefix=f".{batch_id}.", suffix=".json", dir=output_dir, delete=False
            ) as temporary:
                last_message_path = Path(temporary.name)
            attempt_command = command[:-1] + ["--output-last-message", str(last_message_path), "-"]
            try:
                completed = run_command(
                    attempt_command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_message_path.unlink(missing_ok=True)
                last_error = str(exc)
                attempt_records.append(
                    {
                        "attempt": attempt,
                        "returncode": None,
                        "stdout": "",
                        "stderr": str(exc),
                        "raw_last_message": "",
                    }
                )
                continue
            raw = last_message_path.read_text(encoding="utf-8").strip() if last_message_path.stat().st_size else ""
            last_message_path.unlink(missing_ok=True)
            if not raw:
                raw = completed.stdout.strip()
            record = {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "raw_last_message": raw,
            }
            attempt_records.append(record)
            if completed.returncode != 0:
                last_error = f"Codex exited {completed.returncode}: {completed.stderr.strip()}"
                continue
            try:
                structured = json.loads(raw)
                reviews = _validate_review_payload(structured, batch)
            except (json.JSONDecodeError, AutomatedAuditError) as exc:
                last_error = str(exc)
                continue
            return (
                {
                    "batch_id": batch_id,
                    "packet_ids": [packet.packet_id for packet in batch],
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "attempts": attempt_records,
                    "structured_result": structured,
                },
                reviews,
            )
    message = f"Batch {batch_id} failed after {retries + 1} attempt(s): {last_error}"
    raise AutomatedAuditBatchError(
        message,
        {
            "batch_id": batch_id,
            "packet_ids": [packet.packet_id for packet in batch],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "attempts": attempt_records,
            "error": message,
        },
    )


def run_codex_reviews(
    *,
    manifest_path: str | Path,
    construct_packets_path: str | Path | None = None,
    temporal_packets_path: str | Path | None = None,
    model: str | None = None,
    batch_size: int = 10,
    shard_size: int | None = None,
    workers: int = 1,
    retries: int = 2,
    timeout_seconds: float = 600,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Run schema-constrained Codex reviews and write raw and normalized review artifacts."""
    if shard_size is not None:
        if batch_size != 10 and batch_size != shard_size:
            raise AutomatedAuditError("Specify only one of batch_size or shard_size.")
        batch_size = shard_size
    if batch_size <= 0 or workers <= 0 or retries < 0 or timeout_seconds <= 0:
        raise AutomatedAuditError("batch_size/workers/timeout must be positive and retries must be non-negative.")
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_json(manifest_file)
    construct_file = (
        Path(construct_packets_path).resolve()
        if construct_packets_path is not None
        else _manifest_artifact(manifest_file, manifest, "construct_review_packets")
    )
    temporal_file = (
        Path(temporal_packets_path).resolve()
        if temporal_packets_path is not None
        else _manifest_artifact(manifest_file, manifest, "temporal_review_packets")
    )
    output_dir = _output_dir(manifest_file, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_model = _model_name(manifest, model)
    packets = _packets(construct_file, "construct") + _packets(temporal_file, "temporal")
    _validate_unique_packet_ids(packets)
    schema_path = output_dir / ".codex_review_output.schema.json"
    _write_json(schema_path, CODEX_REVIEW_SCHEMA)
    batches = [packets[index : index + batch_size] for index in range(0, len(packets), batch_size)]
    version = _codex_version(run_command)
    git_state = _git_state()

    batch_results: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    batch_failures: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_indexes = {
            executor.submit(
                _run_batch,
                batch_id=f"batch_{index:05d}",
                batch=batch,
                model=selected_model,
                schema_path=schema_path,
                output_dir=output_dir,
                retries=retries,
                timeout_seconds=timeout_seconds,
                run_command=run_command,
            ): index
            for index, batch in enumerate(batches, start=1)
        }
        for future in as_completed(future_indexes):
            index = future_indexes[future]
            try:
                batch_results[index] = future.result()
            except AutomatedAuditBatchError as exc:
                batch_failures[index] = exc.batch_record

    if batch_failures:
        completed_reviews = sum(len(result[1]) for result in batch_results.values())
        failure_report = {
            "report_type": "codex_assisted_automated_audit",
            "report_version": 1,
            "created_at_utc": _utc_now(),
            "status": "failed",
            "git": git_state,
            "manifest": {"path": str(manifest_file), "sha256": _sha256_file(manifest_file)},
            "inputs": {
                "construct_packets": {"path": str(construct_file), "sha256": _sha256_file(construct_file)},
                "temporal_packets": {"path": str(temporal_file), "sha256": _sha256_file(temporal_file)},
            },
            "codex": {
                "version": version,
                "model": selected_model,
                "ephemeral": True,
                "sandbox": "read-only",
                "ignore_user_config": True,
                "ignore_rules": True,
                "approval_policy": "never",
                "output_schema_sha256": _sha256_file(schema_path),
            },
            "execution": {
                "batch_size": batch_size,
                "workers": workers,
                "retries": retries,
                "timeout_seconds": timeout_seconds,
                "packet_count": len(packets),
                "batch_count": len(batches),
                "completed_batch_count": len(batch_results),
                "failed_batch_count": len(batch_failures),
                "completed_review_count": completed_reviews,
            },
            "batches": [
                (batch_results[index][0] if index in batch_results else batch_failures[index])
                for index in sorted(set(batch_results) | set(batch_failures))
            ],
            "reviews": {"path": None, "count": 0, "partial_results_published": False},
        }
        _write_json(output_dir / RUN_FILENAME, failure_report)
        failed_details = "; ".join(
            batch_failures[index]["error"] for index in sorted(batch_failures)
        )
        raise AutomatedAuditError(
            f"Codex review failed for {len(batch_failures)} batch(es): {failed_details}. "
            f"Inspect {output_dir / RUN_FILENAME}; no partial reviews were published."
        )

    ordered = [batch_results[index] for index in sorted(batch_results)]
    reviews = [review for _, batch_reviews in ordered for review in batch_reviews]
    expected_ids = {packet.packet_id for packet in packets}
    observed_ids = [review["packet_id"] for review in reviews]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise AutomatedAuditError("Combined Codex results are incomplete or contain duplicate packet IDs.")
    reviews.sort(key=lambda row: row["packet_id"])
    _write_jsonl(output_dir / REVIEWS_FILENAME, reviews)
    report = {
        "report_type": "codex_assisted_automated_audit",
        "report_version": 2,
        "created_at_utc": _utc_now(),
        "status": "complete",
        "git": git_state,
        "manifest": {"path": str(manifest_file), "sha256": _sha256_file(manifest_file)},
        "inputs": {
            "construct_packets": {"path": str(construct_file), "sha256": _sha256_file(construct_file)},
            "temporal_packets": {"path": str(temporal_file), "sha256": _sha256_file(temporal_file)},
        },
        "codex": {
            "version": version,
            "model": selected_model,
            "ephemeral": True,
            "sandbox": "read-only",
            "ignore_user_config": True,
            "ignore_rules": True,
            "approval_policy": "never",
            "output_schema_sha256": _sha256_file(schema_path),
        },
        "execution": {
            "batch_size": batch_size,
            "workers": workers,
            "retries": retries,
            "timeout_seconds": timeout_seconds,
            "packet_count": len(packets),
            "batch_count": len(batches),
        },
        "batches": [batch_record for batch_record, _ in ordered],
        "reviews": {"path": str(output_dir / REVIEWS_FILENAME), "count": len(reviews)},
    }
    _write_json(output_dir / RUN_FILENAME, report)
    return report


def _bool_or_nonempty(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, dict, str)):
        return bool(value)
    return False


def _deterministic_flags(row: dict[str, Any]) -> tuple[bool, bool, bool]:
    integrity = (
        row.get("status") == "error"
        or _bool_or_nonempty(row.get("integrity_error"))
        or _bool_or_nonempty(row.get("integrity_errors"))
    )
    disagreement = _bool_or_nonempty(row.get("label_disagreement")) or _bool_or_nonempty(
        row.get("deterministic_label_disagreement")
    )
    disagreement = disagreement or row.get("status") == "disagreement"
    temporal_leakage = _bool_or_nonempty(row.get("deterministic_temporal_leakage"))
    return integrity, disagreement, temporal_leakage


def _base_disposition(row: dict[str, Any]) -> str:
    status_defaults = {"error": "exclude", "disagreement": "diagnostic", "unsupported": "diagnostic", "pass": "include"}
    value = row.get("disposition", status_defaults.get(row.get("status"), "include"))
    aliases = {"main": "include", "main_score": "include", "keep": "include"}
    value = aliases.get(value, value)
    if value not in {"include", "diagnostic", "exclude", "exclude_pending_rerender"}:
        raise AutomatedAuditError(f"Unsupported deterministic disposition {value!r}.")
    return value


def _index_unique(rows: Iterable[dict[str, Any]], key: str, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise AutomatedAuditError(f"{source} row {row_number} requires {key}.")
        if value in indexed:
            raise AutomatedAuditError(f"Duplicate {key} {value!r} in {source}.")
        indexed[value] = row
    return indexed


def _validate_stored_review(review: dict[str, Any], row_number: int, source: Path) -> None:
    required = {"packet_id", "case_id", "audit_dimension", "verdict", "rationale", "evidence"}
    if set(review) != required:
        raise AutomatedAuditError(f"Review at {source}:{row_number} has missing or unsupported fields.")
    dimension = review.get("audit_dimension")
    verdict = review.get("verdict")
    allowed_verdicts = {
        "construct": {"pass", "concern", "uncertain"},
        "temporal": {"pass", "uncertain", "suspected_temporal_leakage"},
    }
    if dimension not in allowed_verdicts or verdict not in allowed_verdicts[dimension]:
        raise AutomatedAuditError(f"Review at {source}:{row_number} has an invalid dimension/verdict pair.")
    if not isinstance(review.get("rationale"), str) or not review["rationale"].strip():
        raise AutomatedAuditError(f"Review at {source}:{row_number} requires a rationale.")
    evidence = review.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
        raise AutomatedAuditError(f"Review at {source}:{row_number} has invalid evidence.")
    if _contains_forbidden_label(review):
        raise AutomatedAuditError("AI reviews must not relabel cases or emit EXTERNAL_CONFIRMED.")


def _disposition(
    deterministic: dict[str, Any], reviews: Sequence[dict[str, Any]]
) -> tuple[str, list[str], dict[str, bool]]:
    integrity, disagreement, deterministic_temporal_leakage = _deterministic_flags(deterministic)
    base = _base_disposition(deterministic)
    temporal_leakage = any(review.get("verdict") == "suspected_temporal_leakage" for review in reviews)
    ai_construct_concern = any(
        review.get("audit_dimension") == "construct" and review.get("verdict") in {"concern", "uncertain"}
        for review in reviews
    )
    ai_temporal_uncertainty = any(
        review.get("audit_dimension") == "temporal" and review.get("verdict") == "uncertain" for review in reviews
    )
    flags = {
        "deterministic_integrity_error": integrity,
        "deterministic_label_disagreement": disagreement,
        "deterministic_temporal_leakage": deterministic_temporal_leakage,
        "ai_construct_concern_or_uncertainty": ai_construct_concern,
        "ai_temporal_uncertainty": ai_temporal_uncertainty,
        "suspected_temporal_leakage": temporal_leakage,
    }
    if integrity:
        return "exclude", ["deterministic_integrity_error"], flags
    if deterministic_temporal_leakage:
        return "exclude", ["deterministic_temporal_leakage"], flags
    if base == "exclude":
        return "exclude", ["deterministic_exclude"], flags
    if temporal_leakage or base == "exclude_pending_rerender":
        reasons = []
        if temporal_leakage:
            reasons.append("suspected_temporal_leakage")
        if not reasons:
            reasons.append("deterministic_pending_rerender")
        return "exclude_pending_rerender", reasons, flags
    reasons: list[str] = []
    if disagreement:
        reasons.append("deterministic_label_disagreement")
    if base == "diagnostic":
        reasons.append("deterministic_diagnostic")
    if ai_construct_concern:
        reasons.append("ai_construct_concern_or_uncertainty")
    if ai_temporal_uncertainty:
        reasons.append("ai_temporal_uncertainty")
    if reasons:
        return "diagnostic", reasons, flags
    return "include", ["no_exclusion_or_diagnostic_signal"], flags


def _compact_disposition(row: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "case_id": row["case_id"],
        "disposition": row["disposition"],
    }
    if row["reasons"] != ["no_exclusion_or_diagnostic_signal"]:
        compact["reasons"] = row["reasons"]
    active_flags = {name: True for name, active in row["flags"].items() if active}
    if active_flags:
        compact["flags"] = active_flags
    if row["review_packet_ids"]:
        compact["review_packet_ids"] = row["review_packet_ids"]
    return compact


def finalize_audit(
    *,
    manifest_path: str | Path,
    deterministic_dispositions_path: str | Path | None = None,
    reviews_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge deterministic and AI evidence without changing any benchmark label."""
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_json(manifest_file)
    deterministic_file = (
        Path(deterministic_dispositions_path).resolve()
        if deterministic_dispositions_path is not None
        else _manifest_artifact(manifest_file, manifest, "deterministic_case_status")
    )
    output_dir = _output_dir(manifest_file, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_file = Path(reviews_path).resolve() if reviews_path is not None else output_dir / REVIEWS_FILENAME
    private_map: dict[str, str] = {}
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict) and "private_packet_map" in artifacts:
        private_map_file = _manifest_artifact(manifest_file, manifest, "private_packet_map")
        private_map_payload = _load_json(private_map_file)
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in private_map_payload.items()):
            raise AutomatedAuditError("private_packet_map must map blinded packet IDs to case IDs.")
        private_map = private_map_payload
    deterministic_rows = _load_jsonl(deterministic_file)
    review_rows = _load_jsonl(review_file)
    input_review_sha256 = _sha256_file(review_file)
    deterministic_by_case = _index_unique(deterministic_rows, "case_id", str(deterministic_file))
    if isinstance(artifacts, dict) and "temporal_audit" in artifacts:
        temporal_audit = _load_json(_manifest_artifact(manifest_file, manifest, "temporal_audit"))
        deterministic_temporal_case_ids = {
            hit.get("case_id")
            for hit in temporal_audit.get("hits", [])
            if isinstance(hit, dict) and hit.get("severity") == "high"
        }
        for case_id in deterministic_temporal_case_ids:
            if case_id in deterministic_by_case:
                deterministic_by_case[case_id] = {
                    **deterministic_by_case[case_id],
                    "deterministic_temporal_leakage": True,
                }
    review_by_packet = _index_unique(review_rows, "packet_id", str(review_file))
    if isinstance(artifacts, dict):
        packet_roles = {"construct_review_packets", "temporal_review_packets"}
        present_packet_roles = packet_roles.intersection(artifacts)
        if present_packet_roles and present_packet_roles != packet_roles:
            raise AutomatedAuditError("Manifest must bind both construct and temporal review packets.")
        if present_packet_roles:
            expected_packets = _packets(
                _manifest_artifact(manifest_file, manifest, "construct_review_packets"), "construct"
            ) + _packets(_manifest_artifact(manifest_file, manifest, "temporal_review_packets"), "temporal")
            _validate_unique_packet_ids(expected_packets)
            expected_packet_ids = {packet.packet_id for packet in expected_packets}
            observed_packet_ids = set(review_by_packet)
            missing_packet_ids = sorted(expected_packet_ids - observed_packet_ids)
            unexpected_packet_ids = sorted(observed_packet_ids - expected_packet_ids)
            if missing_packet_ids or unexpected_packet_ids:
                raise AutomatedAuditError(
                    "Codex review packet coverage mismatch: "
                    f"missing={missing_packet_ids}, unexpected={unexpected_packet_ids}."
                )
    reviews_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_number, review in enumerate(review_rows, start=1):
        _validate_stored_review(review, row_number, review_file)
        blinded_case_id = review.get("case_id")
        case_id = private_map.get(str(blinded_case_id), blinded_case_id)
        if case_id not in deterministic_by_case:
            raise AutomatedAuditError(f"Review references unknown case_id {case_id!r}.")
        normalized = dict(review)
        normalized["case_id"] = case_id
        reviews_by_case[case_id].append(normalized)

    dispositions: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for case_id in sorted(deterministic_by_case):
        deterministic = deterministic_by_case[case_id]
        case_reviews = sorted(reviews_by_case.get(case_id, []), key=lambda row: row["packet_id"])
        disposition, reasons, flags = _disposition(deterministic, case_reviews)
        row = {
            "case_id": case_id,
            "disposition": disposition,
            "reasons": reasons,
            "flags": flags,
            "review_packet_ids": [review["packet_id"] for review in case_reviews],
        }
        dispositions.append(row)
        if any(flags.values()):
            disagreements.append(
                {
                    "case_id": case_id,
                    "disposition": disposition,
                    "flags": flags,
                    "reviews": case_reviews,
                }
            )

    normalized_reviews_path = output_dir / REVIEWS_FILENAME
    disagreements_path = output_dir / DISAGREEMENTS_FILENAME
    dispositions_path = output_dir / DISPOSITIONS_FILENAME
    _write_jsonl(normalized_reviews_path, sorted(review_rows, key=lambda row: row["packet_id"]))
    _write_jsonl(disagreements_path, disagreements)
    _write_jsonl(dispositions_path, (_compact_disposition(row) for row in dispositions))
    counts = Counter(row["disposition"] for row in dispositions)
    report = {
        "report_type": "automated_audit_final",
        "report_version": 3,
        "created_at_utc": _utc_now(),
        "policy": {
            "relabeling_allowed": False,
            "external_confirmed_allowed": False,
            "deterministic_integrity_error": "exclude",
            "deterministic_label_disagreement": "diagnostic",
            "deterministic_temporal_leakage": "exclude",
            "ai_construct_concern_or_uncertainty": "diagnostic",
            "suspected_temporal_leakage": "exclude_pending_rerender",
        },
        "inputs": {
            "manifest": {"path": str(manifest_file), "sha256": _sha256_file(manifest_file)},
            "deterministic_dispositions": {
                "path": str(deterministic_file),
                "sha256": _sha256_file(deterministic_file),
            },
            "codex_reviews": {"path": str(review_file), "sha256": input_review_sha256},
        },
        "counts": {
            "cases": len(dispositions),
            "reviews": len(review_rows),
            "disagreements": len(disagreements),
            "by_disposition": dict(sorted(counts.items())),
        },
        "validation": {
            "complete_unique_disposition_coverage": len(dispositions) == len(deterministic_by_case),
            "selection_eligible_disposition": "include",
        },
        "outputs": {
            "reviews": {
                "path": str(normalized_reviews_path),
                "sha256": _sha256_file(normalized_reviews_path),
            },
            "disagreements": {
                "path": str(disagreements_path),
                "sha256": _sha256_file(disagreements_path),
            },
            "dispositions": {
                "path": str(dispositions_path),
                "sha256": _sha256_file(dispositions_path),
            },
        },
    }
    _write_json(output_dir / FINAL_JSON_FILENAME, report)
    markdown = [
        "# Automated Audit Finalization",
        "",
        "This report combines deterministic gates with Codex-assisted review. AI review never changes benchmark labels",
        "and cannot establish `EXTERNAL_CONFIRMED`.",
        "",
        "## Counts",
        "",
        f"- Cases: {len(dispositions)}",
        f"- AI reviews: {len(review_rows)}",
        f"- Cases with disagreement or concern flags: {len(disagreements)}",
    ]
    for disposition in ("include", "diagnostic", "exclude_pending_rerender", "exclude"):
        markdown.append(f"- `{disposition}`: {counts[disposition]}")
    markdown.extend(
        [
            "",
            "## Policy",
            "",
            "Deterministic integrity errors and deterministic temporal leakage are excluded. Deterministic label",
            "disagreements and AI-only construct concerns or uncertainty are diagnostic. AI-suspected temporal",
            "leakage is excluded pending rerender. These",
            "automated decisions are error-discovery dispositions, not human annotation or semantic ground truth.",
            "",
        ]
    )
    (output_dir / FINAL_MD_FILENAME).write_text("\n".join(markdown), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and finalize Codex-assisted automated audit review.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("review")
    review.add_argument("--manifest", required=True)
    review.add_argument("--construct-packets", required=True)
    review.add_argument("--temporal-packets", required=True)
    review.add_argument("--model")
    review.add_argument("--batch-size", type=int, default=10)
    review.add_argument("--workers", type=int, default=1)
    review.add_argument("--retries", type=int, default=2)
    review.add_argument("--timeout-seconds", type=float, default=600)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--deterministic-dispositions", required=True)
    finalize.add_argument("--reviews")
    args = parser.parse_args(argv)
    if args.command == "review":
        report = run_codex_reviews(
            manifest_path=args.manifest,
            construct_packets_path=args.construct_packets,
            temporal_packets_path=args.temporal_packets,
            model=args.model,
            batch_size=args.batch_size,
            workers=args.workers,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        report = finalize_audit(
            manifest_path=args.manifest,
            deterministic_dispositions_path=args.deterministic_dispositions,
            reviews_path=args.reviews,
        )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
