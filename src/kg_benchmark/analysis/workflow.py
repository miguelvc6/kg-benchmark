from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from jsonschema import Draft202012Validator

from kg_benchmark.analysis.statistics import (
    AnalysisStatisticsError,
    cluster_estimate,
    holm_adjust,
    paired_cluster_contrast,
)
from kg_benchmark.matrix.workflow import _load_matrix, _resolve, matrix_status
from kg_benchmark.selection.extensible import group_key_for_record, stratum_for_record
from rescore_run import rescore_run

EVALUATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
A_BOX_ENDPOINTS = {
    "accepted_repair": (True, lambda trace: trace.get("accepted")),
    "functional_success": (False, lambda trace: trace.get("metrics", {}).get("functional_success")),
    "exact_action_match": (False, lambda trace: trace.get("metrics", {}).get("a_box_exact_action_match")),
    "exact_value_match": (False, lambda trace: trace.get("metrics", {}).get("a_box_exact_value_match")),
    "regression_pass": (False, lambda trace: trace.get("metrics", {}).get("a_box_regression_pass")),
    "auditability_complete": (False, lambda trace: trace.get("metrics", {}).get("auditability_complete")),
    "provenance_supported": (False, lambda trace: trace.get("metrics", {}).get("provenance_supported")),
}
T_BOX_ENDPOINTS = {
    "tbox_patch_schema_decision_match_rate": (True, "schema_decision_match"),
    "tbox_patch_taxonomy_code_exact_match_rate": (True, "taxonomy_code_exact_match"),
    "tbox_patch_repair_op_exact_match_rate": (False, "repair_op_exact_match"),
    "tbox_patch_qualifier_property_match_rate": (False, "qualifier_property_match"),
    "tbox_patch_evidence_level_exact_match_rate": (False, "evidence_level_exact_match"),
}


class AnalysisWorkflowError(ValueError):
    """Raised when replay or paper result packaging fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _artifact(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisWorkflowError(f"Required artifact is missing: {path}")
    result: dict[str, Any] = {
        "path": os.path.relpath(path.resolve(), relative_to.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha_file(path),
    }
    if path.suffix == ".jsonl":
        result["records"] = _jsonl_count(path)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisWorkflowError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisWorkflowError(f"Expected a JSON object in {path}.")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise AnalysisWorkflowError(f"Cannot read JSONL artifact {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisWorkflowError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise AnalysisWorkflowError(f"Expected an object at {path}:{line_number}")
            yield row


@lru_cache(maxsize=None)
def _validator(path: Path) -> Draft202012Validator:
    return Draft202012Validator(_load_json(path))


def _validate(value: Any, schema_path: Path, label: str) -> None:
    errors = sorted(_validator(schema_path).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        location = ".".join(str(part) for part in errors[0].path)
        raise AnalysisWorkflowError(
            f"{label} fails {schema_path.name}{f' at {location}' if location else ''}: {errors[0].message}"
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _verify_fingerprint(record: dict[str, Any], *, expected_path: Path | None = None) -> Path:
    path = expected_path.resolve() if expected_path is not None else Path(str(record.get("path") or "")).resolve()
    size = record.get("bytes", record.get("size_bytes"))
    if not path.is_file() or path.stat().st_size != size or _sha_file(path) != record.get("sha256"):
        raise AnalysisWorkflowError(f"Artifact fingerprint mismatch: {path}")
    return path


def _group_cells(matrix: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in matrix["cells"]:
        grouped[cell["execution_group_id"]].append(cell)
    for cells in grouped.values():
        if {cell["task"] for cell in cells} != {"repair_proposal", "track_diagnosis"}:
            raise AnalysisWorkflowError("Every analysis execution group must contain both logical task cells.")
        if len({(cell["model_id"], cell["population"], cell["prompt_regime"], cell["context_bundle"]) for cell in cells}) != 1:
            raise AnalysisWorkflowError("Execution-group logical dimensions disagree.")
    return dict(grouped)


def _verify_evaluation(
    *,
    matrix_dir: Path,
    group_id: str,
    cells: list[dict[str, Any]],
    evaluation_id: str,
    population_count: int,
    schema_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    group_dir = matrix_dir / "executions" / group_id
    manifest_path = group_dir / "evaluations" / evaluation_id / "evaluation_manifest.json"
    manifest = _load_json(manifest_path)
    if schema_root is not None:
        _validate(manifest, schema_root / "evaluation-replay.schema.json", "evaluation replay")
    exemplar = cells[0]
    context = exemplar["context_bundle"]
    if manifest.get("provider_calls") != 0:
        raise AnalysisWorkflowError(f"Evaluation {manifest_path} records provider calls.")
    if manifest.get("evaluation_id") != evaluation_id or manifest.get("selected_case_count") != population_count:
        raise AnalysisWorkflowError(f"Evaluation coverage or id mismatch: {manifest_path}")
    if manifest.get("ablation_bundles") != [context]:
        raise AnalysisWorkflowError(f"Evaluation context mismatch: {manifest_path}")
    source = manifest.get("source_artifacts")
    if not isinstance(source, dict):
        raise AnalysisWorkflowError(f"Evaluation has no source-artifact bindings: {manifest_path}")
    _verify_fingerprint(source["run_config"], expected_path=group_dir / "run_config.json")
    _verify_fingerprint(source["run_manifest"], expected_path=group_dir / "run_manifest.jsonl")
    outputs = manifest.get("outputs", {})
    combined_summary = outputs.get("combined_summary")
    if not isinstance(combined_summary, dict):
        raise AnalysisWorkflowError(f"Evaluation lacks its combined summary binding: {manifest_path}")
    _verify_fingerprint(combined_summary)
    output = outputs.get(context)
    if not isinstance(output, dict):
        raise AnalysisWorkflowError(f"Evaluation lacks context output bindings: {manifest_path}")
    for key in (
        "traces",
        "summary",
        "tbox_taxonomy_patch_traces",
        "tbox_taxonomy_patch_summary",
        "diagnosis_traces",
        "diagnosis_summary",
    ):
        record = output.get(key)
        if not isinstance(record, dict):
            raise AnalysisWorkflowError(f"Evaluation is missing required {key}: {manifest_path}")
        _verify_fingerprint(record)
    return manifest, manifest_path


def replay_matrix_evaluations(
    *,
    matrix_dir: Path,
    evaluation_id: str,
    generation_cache_path: Path,
    schema_root: Path = Path("schemas"),
    rescorer: Callable[..., dict[str, Any]] = rescore_run,
    completeness_check: Callable[..., dict[str, Any]] = matrix_status,
) -> dict[str, Any]:
    """Replay one evaluator version across every physical group without provider calls."""
    if not EVALUATION_ID_PATTERN.fullmatch(evaluation_id):
        raise AnalysisWorkflowError("evaluation_id contains unsupported characters.")
    matrix_dir = matrix_dir.resolve()
    schema_root = schema_root.resolve()
    status = completeness_check(
        matrix_dir=matrix_dir,
        generation_cache_path=generation_cache_path,
        schema_root=schema_root,
    )
    if status.get("complete") is not True:
        raise AnalysisWorkflowError("Evaluation replay requires a complete, independently verified matrix.")
    matrix, _, _ = _load_matrix(matrix_dir, schema_root)
    grouped = _group_cells(matrix)
    cases_path = _resolve(matrix_dir, matrix["dataset"]["cases"]["path"])
    world_path = _resolve(matrix_dir, matrix["dataset"]["world_state"]["path"])
    created = 0
    reused = 0
    evaluation_paths: list[str] = []
    for group_id, cells in sorted(grouped.items()):
        exemplar = cells[0]
        population = matrix["populations"][exemplar["population"]]
        population_path = _resolve(matrix_dir, population["artifact"]["path"])
        group_dir = matrix_dir / "executions" / group_id
        evaluation_path = group_dir / "evaluations" / evaluation_id / "evaluation_manifest.json"
        if not evaluation_path.is_file():
            rescorer(
                run_dir=group_dir,
                evaluation_id=evaluation_id,
                classified_path=cases_path,
                world_state_path=world_path,
                selection_manifest_path=population_path,
            )
            created += 1
        else:
            reused += 1
        _verify_evaluation(
            matrix_dir=matrix_dir,
            group_id=group_id,
            cells=cells,
            evaluation_id=evaluation_id,
            population_count=population["case_count"],
            schema_root=schema_root,
        )
        evaluation_paths.append(str(evaluation_path))
    return {
        "matrix_id": matrix["matrix_id"],
        "evaluation_id": evaluation_id,
        "provider_calls": 0,
        "execution_groups": len(grouped),
        "created_evaluations": created,
        "reused_evaluations": reused,
        "complete_evaluation_coverage": True,
        "evaluation_manifests": evaluation_paths,
    }


def _analysis_parameters(config: dict[str, Any]) -> tuple[int, float, int]:
    if config.get("manifest_type") != "paper_analysis_configuration" or config.get("status") != "frozen":
        raise AnalysisWorkflowError("Paper analysis requires the frozen analysis configuration.")
    bootstrap = config.get("inference", {}).get("bootstrap", {})
    expected = {
        "method": "percentile_cluster_bootstrap",
        "samples": 5000,
        "confidence_level": 0.95,
        "seed": 13,
    }
    if bootstrap != expected:
        raise AnalysisWorkflowError("Analysis does not match the frozen 5,000-sample seed-13 bootstrap.")
    if config.get("pool_models") is not False:
        raise AnalysisWorkflowError("Paper analysis cannot pool models.")
    inference = config.get("inference", {})
    if inference.get("paired_binary_test") != "exact_mcnemar":
        raise AnalysisWorkflowError("Paper analysis requires exact McNemar tests.")
    if inference.get("multiplicity_correction") != "holm":
        raise AnalysisWorkflowError("Paper analysis requires Holm multiplicity correction.")
    return expected["samples"], expected["confidence_level"], expected["seed"]


def _case_metadata(cases_path: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for record in _iter_jsonl(cases_path):
        case_id = record.get("id")
        if not isinstance(case_id, str) or case_id not in selected_ids:
            continue
        stratum = stratum_for_record(record)
        if stratum is None:
            raise AnalysisWorkflowError(f"Selected case has no paper stratum: {case_id}")
        metadata[case_id] = {
            "stratum": stratum,
            "track": "T_BOX" if stratum == "TBOX" else "A_BOX",
            "group_key": group_key_for_record(record),
        }
    missing = sorted(selected_ids - set(metadata))
    if missing:
        raise AnalysisWorkflowError(f"Dataset is missing selected analysis cases: {missing[:5]}")
    return metadata


def _population_payloads(
    matrix: dict[str, Any], matrix_dir: Path, metadata: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for name, record in matrix["populations"].items():
        payload = _load_json(_resolve(matrix_dir, record["artifact"]["path"]))
        if payload.get("name") != name or payload.get("case_count") != len(payload.get("selected_case_ids", [])):
            raise AnalysisWorkflowError(f"Population metadata is inconsistent: {name}")
        expected_groups = [metadata[case_id]["group_key"] for case_id in payload["selected_case_ids"]]
        if payload.get("selected_group_keys") != expected_groups:
            raise AnalysisWorkflowError(f"Population group keys do not align with case order: {name}")
        payload["__artifact_sha256"] = record["artifact"]["sha256"]
        values[name] = payload
    return values


def _analysis_scope_and_roles(
    *, config: dict[str, Any], populations: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, str | None]]:
    confirmatory = config["confirmatory_population"]
    calibration = config["calibration_population"]
    extension_names = sorted(set(populations) - {confirmatory, calibration})
    if not extension_names:
        roles = {
            name: "confirmatory" if name == confirmatory else "azure_calibration"
            for name in populations
        }
        return "confirmatory_and_calibration", roles
    roles: dict[str, str | None] = {name: None for name in populations}
    for name in extension_names:
        payload = populations[name]
        parent = payload.get("parent")
        if not isinstance(parent, dict) or parent.get("relationship") != "nested_extension_of_parent" or parent.get("nesting_proven") is not True:
            raise AnalysisWorkflowError(f"Extension population lacks a nested-parent proof: {name}")
        parent_name = parent.get("name")
        if not isinstance(parent_name, str) or parent_name not in populations:
            raise AnalysisWorkflowError(f"Extension parent must be included in the same matrix: {name}")
        parent_payload = populations[parent_name]
        provenance_parent = payload.get("provenance", {}).get("parent_population")
        if not isinstance(provenance_parent, dict):
            raise AnalysisWorkflowError(f"Extension does not hash-bind its parent population: {name}")
        if provenance_parent.get("sha256") != parent_payload.get("__artifact_sha256"):
            raise AnalysisWorkflowError(f"Extension parent hash does not match the included parent: {name}")
        parent_ids = parent_payload["selected_case_ids"]
        extension_ids = payload["selected_case_ids"]
        for stratum in ("IC-L", "IC-G", "IC-E-elim", "TBOX"):
            parent_prefix = [case_id for case_id in parent_ids if metadata[case_id]["stratum"] == stratum]
            extension_order = [case_id for case_id in extension_ids if metadata[case_id]["stratum"] == stratum]
            if extension_order[: len(parent_prefix)] != parent_prefix:
                raise AnalysisWorkflowError(f"Extension does not preserve the parent prefix for {stratum}: {name}")
        roles[name] = "extension"
    return "extension", roles


def _condition_key(cell: dict[str, Any]) -> tuple[str, str]:
    return cell["prompt_regime"], cell["context_bundle"]


def _trace_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_iter_jsonl(Path(record["path"])))


def _unique_traces(rows: list[dict[str, Any]], expected_ids: set[str], label: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in by_id:
            raise AnalysisWorkflowError(f"{label} contains an invalid or duplicate case id.")
        by_id[case_id] = row
    if set(by_id) != expected_ids:
        raise AnalysisWorkflowError(
            f"{label} coverage differs: missing={sorted(expected_ids - set(by_id))[:5]} "
            f"unexpected={sorted(set(by_id) - expected_ids)[:5]}"
        )
    return by_id


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _value_delta_f1(trace: dict[str, Any]) -> float | None:
    if trace.get("metrics", {}).get("gold_has_value_delta") is not True:
        return None
    detail = trace.get("metric_detail", {})
    predicted = int(detail.get("value_pred", 0))
    gold = int(detail.get("value_gold", 0))
    true_positive = int(detail.get("value_tp", 0))
    return 2 * true_positive / (predicted + gold) if predicted + gold else 0.0


def _add_observation(
    observations: dict[tuple[Any, ...], list[dict[str, Any]]],
    *,
    dimensions: tuple[Any, ...],
    case_id: str,
    metadata: dict[str, dict[str, Any]],
    value: Any,
) -> None:
    numeric = _numeric(value)
    if numeric is None:
        return
    observations[dimensions].append(
        {"case_id": case_id, "cluster_key": metadata[case_id]["group_key"], "value": numeric}
    )


def _collect_group_observations(
    *,
    matrix_dir: Path,
    matrix: dict[str, Any],
    group_id: str,
    cells: list[dict[str, Any]],
    evaluation: dict[str, Any],
    population: dict[str, Any],
    role: str,
    metadata: dict[str, dict[str, Any]],
    model: dict[str, Any],
    observations: dict[tuple[Any, ...], list[dict[str, Any]]],
    diagnosis_rows: list[dict[str, Any]],
) -> None:
    del matrix_dir, matrix, group_id
    exemplar = cells[0]
    context = exemplar["context_bundle"]
    regime = exemplar["prompt_regime"]
    output = evaluation["outputs"][context]
    selected_ids = set(population["selected_case_ids"])
    abox_ids = {case_id for case_id in selected_ids if metadata[case_id]["track"] == "A_BOX"}
    tbox_ids = selected_ids - abox_ids
    abox = _unique_traces(_trace_rows(output["traces"]), abox_ids, "A-box evaluation traces")
    tbox = _unique_traces(
        _trace_rows(output["tbox_taxonomy_patch_traces"]), tbox_ids, "T-box evaluation traces"
    )
    diagnosis = _unique_traces(
        _trace_rows(output["diagnosis_traces"]), selected_ids, "diagnosis evaluation traces"
    )
    prefix = (role, exemplar["population"], exemplar["model_id"], model["provider"])
    condition = (regime, context)
    for case_id, trace in abox.items():
        stratum = metadata[case_id]["stratum"]
        for endpoint, (primary, extractor) in A_BOX_ENDPOINTS.items():
            dimensions = (*prefix, "a_box_repair", "A_BOX", stratum, endpoint, primary, *condition)
            _add_observation(
                observations,
                dimensions=dimensions,
                case_id=case_id,
                metadata=metadata,
                value=extractor(trace),
            )
    for case_id, trace in tbox.items():
        for endpoint, (primary, key) in T_BOX_ENDPOINTS.items():
            dimensions = (*prefix, "t_box_repair", "T_BOX", "TBOX", endpoint, primary, *condition)
            _add_observation(
                observations,
                dimensions=dimensions,
                case_id=case_id,
                metadata=metadata,
                value=trace.get("metrics", {}).get(key),
            )
        dimensions = (
            *prefix,
            "t_box_repair",
            "T_BOX",
            "TBOX",
            "tbox_patch_value_delta_f1_when_applicable",
            False,
            *condition,
        )
        _add_observation(
            observations,
            dimensions=dimensions,
            case_id=case_id,
            metadata=metadata,
            value=_value_delta_f1(trace),
        )
    for locus, locus_ids in (("ALL", selected_ids), ("A_BOX", abox_ids), ("T_BOX", tbox_ids)):
        if not locus_ids:
            continue
        stratum = locus
        dimensions = (*prefix, "track_diagnosis", locus, stratum, "accuracy", True, *condition)
        for case_id in sorted(locus_ids):
            _add_observation(
                observations,
                dimensions=dimensions,
                case_id=case_id,
                metadata=metadata,
                value=diagnosis[case_id].get("exact_track_match"),
            )
    diagnosis_rows.append(
        _diagnosis_summary_row(
            role=role,
            population=exemplar["population"],
            model_id=exemplar["model_id"],
            provider=model["provider"],
            regime=regime,
            context=context,
            traces=list(diagnosis.values()),
        )
    )


def _diagnosis_summary_row(
    *,
    role: str,
    population: str,
    model_id: str,
    provider: str,
    regime: str,
    context: str,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = ("A_BOX", "T_BOX")
    confusion: Counter[tuple[str, str]] = Counter()
    for trace in traces:
        predicted = trace.get("predicted_track")
        confusion[(trace.get("historical_track"), predicted if predicted in {*labels, "AMBIGUOUS"} else "__MISSING__")] += 1
    f1_values: list[float] = []
    per_locus: dict[str, Any] = {}
    for label in labels:
        truth = sum(count for (actual, _), count in confusion.items() if actual == label)
        predicted = sum(count for (_, guess), count in confusion.items() if guess == label)
        tp = confusion[(label, label)]
        f1 = 2 * tp / (truth + predicted) if truth + predicted else None
        if f1 is not None:
            f1_values.append(f1)
        per_locus[label] = {
            "support": truth,
            "precision": tp / predicted if predicted else None,
            "recall": tp / truth if truth else None,
            "f1": f1,
        }
    total = len(traces)
    return {
        "role": role,
        "population": population,
        "model_id": model_id,
        "provider": provider,
        "prompt_regime": regime,
        "context_bundle": context,
        "case_count": total,
        "accuracy": sum(bool(row.get("exact_track_match")) for row in traces) / total,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "false_locus_rate": sum(
            row.get("predicted_track") in labels and row.get("predicted_track") != row.get("historical_track")
            for row in traces
        )
        / total,
        "ambiguous_prediction_rate": sum(row.get("predicted_track") == "AMBIGUOUS" for row in traces) / total,
        "confusion_matrix": {
            actual: {guess: confusion[(actual, guess)] for guess in (*labels, "AMBIGUOUS", "__MISSING__")}
            for actual in labels
        },
        "per_locus": per_locus,
    }


def _relevant_count(
    *, population: dict[str, Any], metadata: dict[str, dict[str, Any]], task: str, stratum: str
) -> int:
    ids = population["selected_case_ids"]
    if task == "a_box_repair":
        return sum(metadata[case_id]["stratum"] == stratum for case_id in ids)
    if task == "t_box_repair":
        return sum(metadata[case_id]["stratum"] == "TBOX" for case_id in ids)
    if stratum == "ALL":
        return len(ids)
    return sum(metadata[case_id]["track"] == stratum for case_id in ids)


def _estimate_rows(
    *,
    analysis_id: str,
    observations: dict[tuple[Any, ...], list[dict[str, Any]]],
    populations: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    samples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimensions, values in sorted(observations.items()):
        role, population, model_id, provider, task, locus, stratum, endpoint, primary, regime, context = dimensions
        total = _relevant_count(population=populations[population], metadata=metadata, task=task, stratum=stratum)
        cluster_unit = "event_group" if locus == "ALL" else ("qid_property" if locus == "A_BOX" else "property_revision")
        try:
            estimate = cluster_estimate(
                values,
                samples=samples,
                confidence_level=confidence_level,
                seed=seed,
            )
        except AnalysisStatisticsError as exc:
            raise AnalysisWorkflowError(str(exc)) from exc
        rows.append(
            {
                "analysis_id": analysis_id,
                "role": role,
                "population": population,
                "model_id": model_id,
                "provider": provider,
                "task": task,
                "locus": locus,
                "stratum": stratum,
                "endpoint": endpoint,
                "primary": primary,
                "prompt_regime": regime,
                "context_bundle": context,
                "applicable_case_count": len(values),
                "excluded_case_count": total - len(values),
                "cluster_unit": cluster_unit,
                "estimate": estimate,
            }
        )
    return rows


def _contrast_rows(
    *,
    analysis_id: str,
    observations: dict[tuple[Any, ...], list[dict[str, Any]]],
    config: dict[str, Any],
    samples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, Any]]:
    bases: dict[tuple[Any, ...], dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(dict)
    for dimensions, values in observations.items():
        *base, primary, regime, context = dimensions
        if primary:
            bases[tuple(base)][(regime, context)] = values
    raw: list[dict[str, Any]] = []
    for base, conditions in sorted(bases.items()):
        role, population, model_id, provider, task, locus, stratum, endpoint = base
        family_id = "|".join(str(value) for value in base)
        for contrast in config["primary_contrasts"]:
            left = contrast["left"]
            right = contrast["right"]
            left_key = (left["prompt_regime"], left["context_bundle"])
            right_key = (right["prompt_regime"], right["context_bundle"])
            if left_key not in conditions or right_key not in conditions:
                raise AnalysisWorkflowError(f"Primary contrast lacks a condition in family {family_id}.")
            try:
                result = paired_cluster_contrast(
                    conditions[left_key],
                    conditions[right_key],
                    samples=samples,
                    confidence_level=confidence_level,
                    seed=seed,
                )
            except AnalysisStatisticsError as exc:
                raise AnalysisWorkflowError(str(exc)) from exc
            raw.append(
                {
                    "analysis_id": analysis_id,
                    "family_id": family_id,
                    "role": role,
                    "population": population,
                    "model_id": model_id,
                    "provider": provider,
                    "task": task,
                    "locus": locus,
                    "stratum": stratum,
                    "endpoint": endpoint,
                    "contrast_id": contrast["contrast_id"],
                    "left": left,
                    "right": right,
                    **result,
                    "p_value": result["mcnemar"]["p_value_exact_two_sided"],
                }
            )
    adjusted: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_family[row["family_id"]].append(row)
    expected_ids = {row["contrast_id"] for row in config["primary_contrasts"]}
    for family_id, family in sorted(by_family.items()):
        if len(family) != 4 or {row["contrast_id"] for row in family} != expected_ids:
            raise AnalysisWorkflowError(f"Holm family does not contain exactly four predeclared contrasts: {family_id}")
        holm_input = [{"index": index, "p_value": row["p_value"]} for index, row in enumerate(family)]
        for holm_row in holm_adjust(holm_input):
            row = family[holm_row["index"]]
            row["holm_adjusted_p_value"] = holm_row["holm_adjusted_p_value"]
            row["holm_reject_at_0_05"] = holm_row["holm_reject_at_0_05"]
            row.pop("p_value")
            adjusted.append(row)
    return sorted(adjusted, key=lambda row: (row["family_id"], row["contrast_id"]))


def _format_interval(value: dict[str, Any]) -> str:
    return f"{value['estimate']:.3f} [{value['ci_lower']:.3f}, {value['ci_upper']:.3f}]"


def _table_for_role(role: str, estimates: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> str:
    title = role.replace("_", " ").title()
    lines = [f"# {title}", "", "Primary endpoint estimates; models and repair loci are not pooled.", ""]
    primary = [row for row in estimates if row["role"] == role and row["primary"]]
    if not primary:
        return "\n".join([*lines, "No results in this role.", ""])
    lines.extend(
        [
            "| Population | Model | Task | Stratum | Endpoint | Regime | Context | n | Case micro (95% CI) | Event macro (95% CI) |",
            "|---|---|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in primary:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["population"],
                    row["model_id"],
                    row["task"],
                    row["stratum"],
                    row["endpoint"],
                    row["prompt_regime"],
                    row["context_bundle"],
                    str(row["applicable_case_count"]),
                    _format_interval(row["estimate"]["case_micro"]),
                    _format_interval(row["estimate"]["event_cluster_macro"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired primary contrasts",
            "",
            "Effects are right minus left. Holm adjustment is within each model × task × stratum × endpoint family.",
            "",
            "| Population | Model | Task | Stratum | Endpoint | Contrast | n | Micro difference (95% CI) | Exact p | Holm p |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in contrasts:
        if row["role"] != role:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["population"],
                    row["model_id"],
                    row["task"],
                    row["stratum"],
                    row["endpoint"],
                    row["contrast_id"],
                    str(row["pair_count"]),
                    _format_interval(row["case_micro_difference"]),
                    f"{row['mcnemar']['p_value_exact_two_sided']:.4g}",
                    f"{row['holm_adjusted_p_value']:.4g}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _diagnosis_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Track Diagnosis",
        "",
        "Diagnosis is scored independently and never routes repair proposals.",
        "",
        "| Role | Population | Model | Regime | Context | n | Accuracy | Macro-F1 | False locus | Ambiguous |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["role"],
                    row["population"],
                    row["model_id"],
                    row["prompt_regime"],
                    row["context_bundle"],
                    str(row["case_count"]),
                    f"{row['accuracy']:.3f}",
                    f"{row['macro_f1']:.3f}" if row["macro_f1"] is not None else "n/a",
                    f"{row['false_locus_rate']:.3f}",
                    f"{row['ambiguous_prediction_rate']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_paper_results(
    *,
    matrix_dir: Path,
    evaluation_id: str,
    generation_cache_path: Path,
    analysis_config_path: Path = Path("paper/analysis.json"),
    output_root: Path = Path("results"),
    schema_root: Path = Path("schemas"),
    repo_root: Path = Path("."),
    completeness_check: Callable[..., dict[str, Any]] = matrix_status,
) -> dict[str, Any]:
    """Aggregate immutable evaluation traces into a compact, hash-bound paper result package."""
    if not EVALUATION_ID_PATTERN.fullmatch(evaluation_id):
        raise AnalysisWorkflowError("evaluation_id contains unsupported characters.")
    matrix_dir = matrix_dir.resolve()
    schema_root = schema_root.resolve()
    analysis_config_path = analysis_config_path.resolve()
    output_root = output_root.resolve()
    repo_root = repo_root.resolve()
    status = completeness_check(
        matrix_dir=matrix_dir,
        generation_cache_path=generation_cache_path,
        schema_root=schema_root,
    )
    if status.get("complete") is not True:
        raise AnalysisWorkflowError("Paper analysis requires a complete, independently verified matrix.")
    matrix, _, matrix_path = _load_matrix(matrix_dir, schema_root)
    config = _load_json(analysis_config_path)
    samples, confidence_level, seed = _analysis_parameters(config)
    grouped = _group_cells(matrix)
    model_by_id = {model["model_id"]: model for model in matrix["models"]}
    selected_ids: set[str] = set()
    for population in matrix["populations"].values():
        payload = _load_json(_resolve(matrix_dir, population["artifact"]["path"]))
        selected_ids.update(payload["selected_case_ids"])
    cases_path = _resolve(matrix_dir, matrix["dataset"]["cases"]["path"])
    metadata = _case_metadata(cases_path, selected_ids)
    populations = _population_payloads(matrix, matrix_dir, metadata)
    scope, roles = _analysis_scope_and_roles(config=config, populations=populations, metadata=metadata)

    if scope == "confirmatory_and_calibration":
        configured = _load_json(_resolve(matrix_dir, matrix["model_configuration"]["path"]))["models"]
        expected = {model["model_id"] for model in configured}
        if set(model_by_id) != expected:
            raise AnalysisWorkflowError("Base paper analysis requires every configured model.")
        for model in configured:
            if model["population"] not in populations:
                raise AnalysisWorkflowError(f"Base paper analysis is missing {model['population']}.")
        for cells in grouped.values():
            exemplar = cells[0]
            model = model_by_id[exemplar["model_id"]]
            if exemplar["population"] != model["population"]:
                raise AnalysisWorkflowError("Base paper analysis forbids cross-population model cells.")
            role = roles[exemplar["population"]]
            if (role == "confirmatory" and model["provider"] != "ollama") or (
                role == "azure_calibration" and model["provider"] != "azure"
            ):
                raise AnalysisWorkflowError("Base paper model/provider role is inconsistent.")

    evaluations: dict[str, tuple[dict[str, Any], Path]] = {}
    for group_id, cells in sorted(grouped.items()):
        exemplar = cells[0]
        evaluations[group_id] = _verify_evaluation(
            matrix_dir=matrix_dir,
            group_id=group_id,
            cells=cells,
            evaluation_id=evaluation_id,
            population_count=matrix["populations"][exemplar["population"]]["case_count"],
            schema_root=schema_root,
        )
    workflow_path = Path(__file__).resolve()
    statistics_path = workflow_path.with_name("statistics.py")
    dependency_lock_path = repo_root / "uv.lock"
    basis = {
        "matrix_sha256": _sha_file(matrix_path),
        "analysis_configuration_sha256": _sha_file(analysis_config_path),
        "analysis_workflow_sha256": _sha_file(workflow_path),
        "analysis_statistics_sha256": _sha_file(statistics_path),
        "dependency_lock_sha256": _sha_file(dependency_lock_path),
        "evaluation_id": evaluation_id,
        "evaluation_manifests": [_sha_file(path) for _, path in evaluations.values()],
        "scope": scope,
    }
    analysis_id = f"analysis_{_sha_value(basis)[:20]}"
    observations: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    diagnosis_rows: list[dict[str, Any]] = []
    for group_id, cells in sorted(grouped.items()):
        exemplar = cells[0]
        role = roles[exemplar["population"]]
        if role is None:
            continue
        evaluation, _ = evaluations[group_id]
        _collect_group_observations(
            matrix_dir=matrix_dir,
            matrix=matrix,
            group_id=group_id,
            cells=cells,
            evaluation=evaluation,
            population=populations[exemplar["population"]],
            role=role,
            metadata=metadata,
            model=model_by_id[exemplar["model_id"]],
            observations=observations,
            diagnosis_rows=diagnosis_rows,
        )
    estimates = _estimate_rows(
        analysis_id=analysis_id,
        observations=observations,
        populations=populations,
        metadata=metadata,
        samples=samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    contrasts = _contrast_rows(
        analysis_id=analysis_id,
        observations=observations,
        config=config,
        samples=samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    estimate_schema = schema_root / "analysis-estimate.schema.json"
    contrast_schema = schema_root / "analysis-contrast.schema.json"
    for row in estimates:
        _validate(row, estimate_schema, "analysis estimate")
    for row in contrasts:
        _validate(row, contrast_schema, "analysis contrast")
    for row in diagnosis_rows:
        _validate(row, schema_root / "analysis-diagnosis.schema.json", "diagnosis summary")

    result_dir = output_root / analysis_id
    if result_dir.exists():
        verified = verify_paper_results(result_dir=result_dir, schema_root=schema_root)
        return {"result_dir": str(result_dir), "manifest": _load_json(result_dir / "manifest.json"), "verification": verified}
    temporary = output_root / f".{analysis_id}.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        estimates_path = temporary / "estimates.jsonl"
        contrasts_path = temporary / "contrasts.jsonl"
        diagnosis_path = temporary / "diagnosis.jsonl"
        summary_path = temporary / "summary.json"
        _write_jsonl(estimates_path, estimates)
        _write_jsonl(contrasts_path, contrasts)
        _write_jsonl(
            diagnosis_path,
            sorted(
                diagnosis_rows,
                key=lambda row: (
                    row["role"],
                    row["population"],
                    row["model_id"],
                    row["prompt_regime"],
                    row["context_bundle"],
                ),
            ),
        )
        roles_present = sorted({row["role"] for row in estimates})
        summary = {
            "manifest_type": "paper_analysis_summary",
            "manifest_version": 1,
            "analysis_id": analysis_id,
            "scope": scope,
            "matrix_id": matrix["matrix_id"],
            "evaluation_id": evaluation_id,
            "provider_calls": 0,
            "roles": roles_present,
            "populations": sorted({row["population"] for row in estimates}),
            "models": sorted({row["model_id"] for row in estimates}),
            "tasks": sorted({row["task"] for row in estimates}),
            "estimate_rows": len(estimates),
            "primary_estimate_rows": sum(row["primary"] for row in estimates),
            "contrast_rows": len(contrasts),
            "holm_families": len({row["family_id"] for row in contrasts}),
            "diagnosis_rows": len(diagnosis_rows),
            "interpretation": {
                "confirmatory": "headline local-model results on the frozen main population",
                "azure_calibration": "paired calibration reference; no cross-model significance tests",
                "extension": "separate nested extension; never replaces confirmatory results",
            },
        }
        _write_json(summary_path, summary)
        _validate(summary, schema_root / "analysis-summary.schema.json", "analysis summary")
        tables = temporary / "tables"
        tables.mkdir()
        table_paths = {
            "table_confirmatory": tables / "confirmatory.md",
            "table_azure_calibration": tables / "azure-calibration.md",
            "table_extensions": tables / "extensions.md",
            "table_diagnosis": tables / "diagnosis.md",
        }
        table_paths["table_confirmatory"].write_text(
            _table_for_role("confirmatory", estimates, contrasts), encoding="utf-8"
        )
        table_paths["table_azure_calibration"].write_text(
            _table_for_role("azure_calibration", estimates, contrasts), encoding="utf-8"
        )
        table_paths["table_extensions"].write_text(
            _table_for_role("extension", estimates, contrasts), encoding="utf-8"
        )
        table_paths["table_diagnosis"].write_text(_diagnosis_table(diagnosis_rows), encoding="utf-8")
        output_paths = {
            "summary": summary_path,
            "estimates": estimates_path,
            "contrasts": contrasts_path,
            "diagnosis": diagnosis_path,
            **table_paths,
        }
        manifest = {
            "manifest_type": "paper_analysis_results",
            "manifest_version": 1,
            "analysis_id": analysis_id,
            "scope": scope,
            "matrix_id": matrix["matrix_id"],
            "evaluation_id": evaluation_id,
            "provider_calls": 0,
            "analysis_configuration": _artifact(analysis_config_path, relative_to=temporary),
            "analysis_code": {
                "workflow": _artifact(workflow_path, relative_to=temporary),
                "statistics": _artifact(statistics_path, relative_to=temporary),
            },
            "dependency_lock": _artifact(dependency_lock_path, relative_to=temporary),
            "matrix": _artifact(matrix_path, relative_to=temporary),
            "evaluation_manifests": [
                _artifact(path, relative_to=temporary) for _, path in sorted(evaluations.values(), key=lambda item: str(item[1]))
            ],
            "outputs": {name: _artifact(path, relative_to=temporary) for name, path in output_paths.items()},
            "validation": {
                "complete_matrix": True,
                "complete_evaluation_coverage": True,
                "no_provider_calls": True,
                "models_not_pooled": True,
                "loci_not_combined": True,
                "roles_separated": True,
                "four_contrasts_per_holm_family": True,
                "extensions_do_not_replace_confirmatory": True,
            },
        }
        _validate(manifest, schema_root / "analysis-result-manifest.schema.json", "analysis manifest")
        _write_json(temporary / "manifest.json", manifest)
        output_root.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, result_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verification = verify_paper_results(result_dir=result_dir, schema_root=schema_root)
    return {"result_dir": str(result_dir), "manifest": _load_json(result_dir / "manifest.json"), "verification": verification}


def verify_paper_results(*, result_dir: Path, schema_root: Path = Path("schemas")) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    schema_root = schema_root.resolve()
    manifest_path = result_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    _validate(manifest, schema_root / "analysis-result-manifest.schema.json", "analysis manifest")
    checked = 0
    for record in [
        manifest["analysis_configuration"],
        *manifest["analysis_code"].values(),
        manifest["dependency_lock"],
        manifest["matrix"],
        *manifest["evaluation_manifests"],
        *manifest["outputs"].values(),
    ]:
        path = _resolve(result_dir, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha_file(path) != record["sha256"]:
            raise AnalysisWorkflowError(f"Result artifact fingerprint mismatch: {path}")
        if path.suffix == ".jsonl" and _jsonl_count(path) != record.get("records"):
            raise AnalysisWorkflowError(f"Result JSONL record count mismatch: {path}")
        checked += 1
    for row in _iter_jsonl(_resolve(result_dir, manifest["outputs"]["estimates"]["path"])):
        _validate(row, schema_root / "analysis-estimate.schema.json", "analysis estimate")
    contrasts = list(_iter_jsonl(_resolve(result_dir, manifest["outputs"]["contrasts"]["path"])))
    for row in contrasts:
        _validate(row, schema_root / "analysis-contrast.schema.json", "analysis contrast")
    families = Counter(row["family_id"] for row in contrasts)
    if any(count != 4 for count in families.values()):
        raise AnalysisWorkflowError("A stored Holm family does not contain four contrasts.")
    for row in _iter_jsonl(_resolve(result_dir, manifest["outputs"]["diagnosis"]["path"])):
        _validate(row, schema_root / "analysis-diagnosis.schema.json", "diagnosis summary")
    summary = _load_json(_resolve(result_dir, manifest["outputs"]["summary"]["path"]))
    _validate(summary, schema_root / "analysis-summary.schema.json", "analysis summary")
    canonical = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if manifest_path.read_text(encoding="utf-8") != canonical:
        raise AnalysisWorkflowError("Result manifest is not in byte-reproducible canonical form.")
    return {
        "valid": True,
        "analysis_id": manifest["analysis_id"],
        "scope": manifest["scope"],
        "checked_artifacts": checked,
        "manifest_sha256": _sha_file(manifest_path),
        "manifest_byte_reproduced": True,
        "provider_calls": 0,
    }
