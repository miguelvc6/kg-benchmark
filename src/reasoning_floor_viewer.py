#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

import streamlit as st

from guardian.reasoning import PromptBundle
from guardian.reasoning_floor_viewer_data import (
    BundleDebugData,
    CaseDebugRecord,
    build_case_prompt_debug,
    discover_run_directories,
    extract_response_content,
    list_run_bundles,
    load_bundle_debug_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reports-root", default="reports/reasoning_floor")
    parser.add_argument("--classified-benchmark", default=None)
    parser.add_argument("--world-state", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


@st.cache_data(show_spinner=False)
def load_bundle_data_cached(
    reports_root: str,
    run_dir: str,
    bundle_name: str,
    classified_benchmark: Optional[str],
    world_state: Optional[str],
) -> BundleDebugData:
    return load_bundle_debug_data(
        reports_root=reports_root,
        run_dir=run_dir,
        bundle_name=bundle_name,
        classified_benchmark=classified_benchmark,
        world_state=world_state,
    )


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Reasoning Floor Debugger", layout="wide")
    _apply_page_style()
    _initialize_sidebar_state(args)

    st.title("Reasoning Floor Debugger")
    st.caption("Read-only browser for reasoning-floor runs, predictions, prompts, and evaluation traces.")

    reports_root = st.sidebar.text_input("Reports root", key="reports_root_input")
    run_dirs = discover_run_directories(reports_root)
    if not run_dirs:
        st.error(f"No reasoning-floor runs found under {reports_root}.")
        return

    run_names = [path.name for path in run_dirs]
    selected_run_name = st.sidebar.selectbox("Run", options=run_names, key="selected_run_name")
    run_dir = next(path for path in run_dirs if path.name == selected_run_name)

    bundle_names = list_run_bundles(run_dir)
    if not bundle_names:
        st.error(f"No ablation bundle directories found in {run_dir}.")
        return
    selected_bundle = st.sidebar.selectbox("Ablation bundle", options=bundle_names, key="selected_bundle_name")

    with st.sidebar.expander("Input Overrides"):
        classified_benchmark = st.text_input(
            "Classified benchmark",
            key="classified_benchmark_override",
            help="Optional override for live evaluation and prompt reconstruction.",
        ).strip()
        world_state = st.text_input(
            "World state",
            key="world_state_override",
            help="Optional override for live evaluation and prompt reconstruction.",
        ).strip()

    if st.sidebar.button("Update View", use_container_width=True):
        load_bundle_data_cached.clear()
        st.rerun()

    bundle_data = load_bundle_data_cached(
        reports_root,
        str(run_dir),
        selected_bundle,
        classified_benchmark or None,
        world_state or None,
    )

    st.sidebar.markdown("### Filters")
    case_query = st.sidebar.text_input("Case id contains", key="case_query_filter").strip().lower()
    track_options = sorted({row.historical_track for row in bundle_data.case_rows if row.historical_track})
    selected_tracks = st.sidebar.multiselect(
        "Historical track",
        track_options,
        default=track_options,
        key="historical_track_filter",
    )
    parse_status_options = sorted({row.proposal_parse_status for row in bundle_data.case_rows})
    selected_parse_status = st.sidebar.multiselect(
        "Proposal parse status",
        parse_status_options,
        default=parse_status_options,
        key="proposal_parse_status_filter",
    )
    proposal_type_options = sorted({row.proposal_type for row in bundle_data.case_rows if row.proposal_type})
    selected_proposal_types = st.sidebar.multiselect(
        "Proposal type",
        proposal_type_options,
        default=proposal_type_options,
        key="proposal_type_filter",
    )
    accepted_filter = st.sidebar.selectbox(
        "Accepted",
        options=["all", "accepted", "rejected", "unknown"],
        key="accepted_filter",
    )

    filtered_cases = filter_case_rows(
        bundle_data.case_rows,
        case_query=case_query,
        tracks=set(selected_tracks),
        parse_statuses=set(selected_parse_status),
        proposal_types=set(selected_proposal_types),
        accepted_filter=accepted_filter,
    )
    if not filtered_cases:
        st.warning("No cases match the current filters.")
        return

    render_run_header(bundle_data)

    case_ids = [row.case_id for row in filtered_cases]
    selected_case_key = f"selected_case_id::{selected_run_name}::{selected_bundle}"
    current_case_id = st.session_state.get(selected_case_key)
    if current_case_id not in case_ids:
        st.session_state[selected_case_key] = case_ids[0]
        current_case_id = case_ids[0]
    selected_index = case_ids.index(current_case_id)

    nav_cols = st.columns([1, 1, 4])
    with nav_cols[0]:
        if st.button("Previous", disabled=selected_index == 0, use_container_width=True):
            st.session_state[selected_case_key] = case_ids[selected_index - 1]
            st.rerun()
    with nav_cols[1]:
        if st.button("Next", disabled=selected_index >= len(case_ids) - 1, use_container_width=True):
            st.session_state[selected_case_key] = case_ids[selected_index + 1]
            st.rerun()
    with nav_cols[2]:
        selected_case_id = st.selectbox(
            "Case",
            options=case_ids,
            index=selected_index,
            key=selected_case_key,
        )

    selected_case = next(row for row in filtered_cases if row.case_id == selected_case_id)
    prompt_debug = build_case_prompt_debug(bundle_data, selected_case_id)

    with st.expander("Filtered cases", expanded=False):
        st.dataframe([case_table_row(row) for row in filtered_cases], use_container_width=True, hide_index=True)

    tabs = st.tabs(["Overview", "Inputs", "Track Prediction", "Repair Proposal", "Evaluation", "Raw JSON"])
    with tabs[0]:
        render_overview_tab(bundle_data, selected_case)
    with tabs[1]:
        render_inputs_tab(prompt_debug)
    with tabs[2]:
        render_track_prediction_tab(selected_case)
    with tabs[3]:
        render_repair_proposal_tab(selected_case)
    with tabs[4]:
        render_evaluation_tab(selected_case)
    with tabs[5]:
        render_raw_json_tab(bundle_data, selected_case, prompt_debug)


def render_run_header(bundle_data: BundleDebugData) -> None:
    run_info = bundle_data.run_info or {}
    left, right = st.columns([3, 2])
    with left:
        st.subheader(bundle_data.run_dir.name)
        st.caption(
            " / ".join(
                str(part)
                for part in (
                    run_info.get("provider"),
                    run_info.get("model"),
                    bundle_data.bundle_name,
                )
                if part
            )
        )
    with right:
        st.caption(
            f"Evaluation: {bundle_data.summary_source or bundle_data.traces_source} | "
            f"Classified input: {bundle_data.input_sources.get('classified_benchmark')} | "
            f"World state: {bundle_data.input_sources.get('world_state')}"
        )

    summary = bundle_data.bundle_summary or {}
    counts = summary.get("counts") if isinstance(summary, dict) else {}
    overall = summary.get("overall_metrics") if isinstance(summary, dict) else {}
    usage = bundle_data.usage_summary
    total_cases = counts.get("cases") if isinstance(counts, dict) else None
    if not isinstance(total_cases, int):
        total_cases = len(bundle_data.case_rows)
    proposal_present_rate = None
    if isinstance(counts, dict) and total_cases:
        proposal_present = counts.get("proposal_present")
        if isinstance(proposal_present, int):
            proposal_present_rate = proposal_present / total_cases

    metric_cols = st.columns(6)
    metric_cols[0].metric("Cases", format_int(total_cases))
    metric_cols[1].metric("Proposal present", format_ratio(proposal_present_rate))
    metric_cols[2].metric("Accepted rate", format_ratio(overall.get("accepted_rate")))
    metric_cols[3].metric("Functional success", format_ratio(overall.get("functional_success_rate")))
    metric_cols[4].metric("Track accuracy", format_ratio(overall.get("track_diagnosis_accuracy")))
    metric_cols[5].metric("Total tokens", format_int(usage.get("total_tokens")))

    detail_cols = st.columns([2, 2, 2, 3])
    detail_cols[0].metric("Mean tokens/call", format_float(usage.get("mean_total_tokens_per_call"), 1))
    detail_cols[1].metric("Calls", format_int(usage.get("call_count")))
    detail_cols[2].metric("Elapsed seconds", format_float(usage.get("elapsed_seconds"), 1))
    detail_cols[3].metric("Estimated cost", format_cost(usage.get("estimated_cost_usd")))

    parse_counts = bundle_data.parse_status_counts
    if parse_counts:
        with st.expander("Parse-status breakdown", expanded=False):
            st.dataframe(parse_counts, use_container_width=True, hide_index=True)


def render_overview_tab(bundle_data: BundleDebugData, case: CaseDebugRecord) -> None:
    record = case.record or {}
    trace = case.trace or {}
    diagnosis = trace.get("track_diagnosis") if isinstance(trace.get("track_diagnosis"), dict) else {}

    hero_cols = st.columns(6)
    hero_cols[0].metric("Case id", case.case_id)
    hero_cols[1].metric("Historical track", case.historical_track or "n/a")
    hero_cols[2].metric("Proposal type", case.proposal_type or "n/a")
    hero_cols[3].metric("Proposal parse", case.proposal_parse_status)
    hero_cols[4].metric("Diagnosis parse", case.diagnosis_parse_status)
    hero_cols[5].metric("Accepted", format_bool(case.accepted))

    detail_cols = st.columns(4)
    detail_cols[0].metric("QID", value_or_na(record.get("qid")))
    detail_cols[1].metric("Property", value_or_na(record.get("property")))
    detail_cols[2].metric("Class", value_or_na((record.get("classification") or {}).get("class")))
    detail_cols[3].metric("Subtype", value_or_na((record.get("classification") or {}).get("subtype")))

    secondary_cols = st.columns(4)
    secondary_cols[0].metric("Track exact match", format_bool(diagnosis.get("exact_track_match")))
    secondary_cols[1].metric("Proposal present", format_bool(trace.get("proposal_present")))
    secondary_cols[2].metric("Proposal valid", format_bool(trace.get("proposal_valid")))
    secondary_cols[3].metric("Executable", format_bool(trace.get("proposal_executable")))

    st.markdown("#### Metadata")
    st.json(
        {
            "case_id": case.case_id,
            "bundle": bundle_data.bundle_name,
            "labels_en": record.get("labels_en"),
            "violation_context": record.get("violation_context"),
            "persistence_check": record.get("persistence_check"),
            "proposal_manifest": case.proposal_manifest,
            "diagnosis_manifest": case.diagnosis_manifest,
        },
        expanded=False,
    )


def render_inputs_tab(prompt_debug: Any) -> None:
    if prompt_debug.error:
        st.warning(prompt_debug.error)
    prompt_cols = st.columns(2)
    with prompt_cols[0]:
        st.markdown("#### Track diagnosis input")
        render_prompt_bundle(prompt_debug.diagnosis_prompt)
    with prompt_cols[1]:
        st.markdown("#### Repair proposal input")
        render_prompt_bundle(prompt_debug.proposal_prompt)

    with st.expander("World-state entry", expanded=False):
        if prompt_debug.world_state_entry is None:
            st.info("No world-state entry was required or available for this bundle.")
        else:
            st.json(prompt_debug.world_state_entry, expanded=False)


def render_track_prediction_tab(case: CaseDebugRecord) -> None:
    trace = case.trace or {}
    diagnosis = trace.get("track_diagnosis") if isinstance(trace.get("track_diagnosis"), dict) else {}
    manifest = case.diagnosis_manifest or {}
    normalized = case.diagnosis_normalized
    raw_content = extract_response_content(case.diagnosis_raw)

    cols = st.columns(5)
    cols[0].metric(
        "Predicted track",
        value_or_na((normalized or {}).get("predicted_track") or diagnosis.get("predicted_track")),
    )
    cols[1].metric("Confidence", value_or_na((normalized or {}).get("confidence") or diagnosis.get("confidence")))
    cols[2].metric("Exact match", format_bool(diagnosis.get("exact_track_match")))
    cols[3].metric("Ambiguous", format_bool(diagnosis.get("ambiguous_prediction")))
    cols[4].metric("Parse status", case.diagnosis_parse_status)

    render_usage_block(manifest.get("usage"))
    if diagnosis.get("rationale"):
        st.markdown("#### Rationale")
        st.write(diagnosis.get("rationale"))
    elif normalized and normalized.get("rationale"):
        st.markdown("#### Rationale")
        st.write(normalized.get("rationale"))

    st.markdown("#### Assistant output")
    if raw_content:
        render_text_or_json(raw_content)
    else:
        st.info("No raw track-diagnosis response content was found.")

    st.markdown("#### Normalized diagnosis")
    if normalized:
        st.json(normalized, expanded=False)
    else:
        st.info("No normalized diagnosis row is available.")


def render_repair_proposal_tab(case: CaseDebugRecord) -> None:
    trace = case.trace or {}
    manifest = case.proposal_manifest or {}
    raw_content = extract_response_content(case.proposal_raw)
    parser_error = manifest.get("parser_error")

    cols = st.columns(6)
    cols[0].metric("Proposal type", case.proposal_type or "n/a")
    cols[1].metric("Parse status", case.proposal_parse_status)
    cols[2].metric("Accepted", format_bool(trace.get("accepted")))
    cols[3].metric("Valid", format_bool(trace.get("proposal_valid")))
    cols[4].metric("Executable", format_bool(trace.get("proposal_executable")))
    cols[5].metric("Functional success", format_ratio((trace.get("metrics") or {}).get("functional_success")))

    render_usage_block(manifest.get("usage"))
    if parser_error:
        st.error(parser_error)

    st.markdown("#### Normalized repair proposal")
    if case.proposal_normalized:
        st.json(case.proposal_normalized, expanded=False)
    else:
        st.info("No normalized repair proposal row is available for this case.")

    st.markdown("#### Assistant output")
    if raw_content:
        render_text_or_json(raw_content)
    else:
        st.info("No raw proposal response content was found.")

    comparison = trace.get("comparison")
    details = trace.get("details")
    if comparison or details:
        with st.expander("Proposal comparison details", expanded=False):
            if comparison:
                st.json(comparison, expanded=False)
            if details:
                st.json(details, expanded=False)


def render_evaluation_tab(case: CaseDebugRecord) -> None:
    if not case.trace:
        st.info("No evaluation trace is available for this case.")
        return
    trace = case.trace
    metrics = trace.get("metrics") or {}
    diagnosis = trace.get("track_diagnosis") or {}

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accepted", format_bool(trace.get("accepted")))
    metric_cols[1].metric("Historical agreement", format_ratio(metrics.get("exact_historical_agreement")))
    metric_cols[2].metric("Provenance completeness", format_ratio(metrics.get("provenance_completeness")))
    metric_cols[3].metric("Info preservation", format_float(metrics.get("information_preservation"), 2))

    diagnosis_cols = st.columns(4)
    diagnosis_cols[0].metric("Track exact match", format_bool(diagnosis.get("exact_track_match")))
    diagnosis_cols[1].metric("Predicted track", value_or_na(diagnosis.get("predicted_track")))
    diagnosis_cols[2].metric("Historical track", value_or_na(diagnosis.get("historical_track")))
    diagnosis_cols[3].metric("Confidence", value_or_na(diagnosis.get("confidence")))

    st.markdown("#### Trace")
    st.json(trace, expanded=False)


def render_raw_json_tab(bundle_data: BundleDebugData, case: CaseDebugRecord, prompt_debug: Any) -> None:
    sections = [
        ("Classified record", case.record),
        ("Proposal manifest", case.proposal_manifest),
        ("Diagnosis manifest", case.diagnosis_manifest),
        ("Proposal raw record", case.proposal_raw),
        ("Diagnosis raw record", case.diagnosis_raw),
        ("Normalized proposal", case.proposal_normalized),
        ("Normalized diagnosis", case.diagnosis_normalized),
        ("Evaluation trace", case.trace),
        ("Prompt reconstruction", {
            "proposal_prompt": prompt_bundle_to_dict(prompt_debug.proposal_prompt),
            "diagnosis_prompt": prompt_bundle_to_dict(prompt_debug.diagnosis_prompt),
            "error": prompt_debug.error,
        }),
        ("Run summary", bundle_data.run_summary),
    ]
    for title, payload in sections:
        with st.expander(title, expanded=False):
            if payload is None:
                st.info("Not available.")
            else:
                st.json(payload, expanded=False)


def render_prompt_bundle(prompt_bundle: Optional[PromptBundle]) -> None:
    if prompt_bundle is None:
        st.info("Prompt reconstruction unavailable.")
        return
    st.caption(f"{prompt_bundle.prompt_name} | bundle={prompt_bundle.ablation_bundle}")
    st.markdown("**System prompt**")
    st.code(prompt_bundle.system_prompt, language="text")
    st.markdown("**Response format**")
    st.json(prompt_bundle.response_format, expanded=False)
    st.markdown("**User prompt**")
    render_text_or_json(prompt_bundle.prompt)


def render_usage_block(usage: Optional[dict[str, Any]]) -> None:
    usage = usage if isinstance(usage, dict) else {}
    cols = st.columns(5)
    cols[0].metric("Prompt tokens", format_int(usage.get("prompt_tokens")))
    cols[1].metric("Completion tokens", format_int(usage.get("completion_tokens")))
    cols[2].metric("Total tokens", format_int(usage.get("total_tokens")))
    cols[3].metric("Elapsed seconds", format_float(usage.get("elapsed_seconds"), 2))
    cols[4].metric("Estimated cost", format_cost(usage.get("estimated_cost_usd")))


def filter_case_rows(
    case_rows: list[CaseDebugRecord],
    *,
    case_query: str,
    tracks: set[str],
    parse_statuses: set[str],
    proposal_types: set[str],
    accepted_filter: str,
) -> list[CaseDebugRecord]:
    filtered = []
    for row in case_rows:
        if case_query and case_query not in row.case_id.lower():
            continue
        if tracks and row.historical_track not in tracks:
            continue
        if parse_statuses and row.proposal_parse_status not in parse_statuses:
            continue
        if proposal_types and row.proposal_type not in proposal_types:
            continue
        if accepted_filter == "accepted" and row.accepted is not True:
            continue
        if accepted_filter == "rejected" and row.accepted is not False:
            continue
        if accepted_filter == "unknown" and row.accepted is not None:
            continue
        filtered.append(row)
    return filtered


def case_table_row(row: CaseDebugRecord) -> dict[str, Any]:
    trace = row.trace or {}
    diagnosis = trace.get("track_diagnosis") if isinstance(trace.get("track_diagnosis"), dict) else {}
    return {
        "case_id": row.case_id,
        "track": row.historical_track,
        "proposal_type": row.proposal_type,
        "proposal_parse_status": row.proposal_parse_status,
        "diagnosis_parse_status": row.diagnosis_parse_status,
        "accepted": row.accepted,
        "track_exact_match": diagnosis.get("exact_track_match"),
    }


def prompt_bundle_to_dict(prompt_bundle: Optional[PromptBundle]) -> Optional[dict[str, Any]]:
    if prompt_bundle is None:
        return None
    return {
        "ablation_bundle": prompt_bundle.ablation_bundle,
        "prompt_name": prompt_bundle.prompt_name,
        "system_prompt": prompt_bundle.system_prompt,
        "response_format": prompt_bundle.response_format,
        "prompt": prompt_bundle.prompt,
    }


def render_text_or_json(value: str) -> None:
    try:
        parsed = json.loads(value)
    except Exception:
        st.code(value, language="text")
        return
    st.json(parsed, expanded=False)


def format_ratio(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{100 * float(value):.1f}%"
    return str(value)


def format_float(value: Any, digits: int) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def format_int(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    return str(value)


def format_cost(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (int, float)):
        return f"${float(value):.4f}"
    return str(value)


def format_bool(value: Any) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def value_or_na(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return str(value)


def _initialize_sidebar_state(args: argparse.Namespace) -> None:
    defaults = {
        "reports_root_input": args.reports_root,
        "classified_benchmark_override": args.classified_benchmark or "",
        "world_state_override": args.world_state or "",
        "case_query_filter": "",
        "historical_track_filter": [],
        "proposal_parse_status_filter": [],
        "proposal_type_filter": [],
        "accepted_filter": "all",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(237, 197, 63, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(33, 126, 93, 0.12), transparent 24%),
                linear-gradient(180deg, #fbf8ef 0%, #f4efe1 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #19352f 0%, #22443c 100%);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
            color: #f8f5ec;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="input"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] input {
            color: #17342d !important;
            -webkit-text-fill-color: #17342d !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #fffaf0;
            border-color: rgba(25, 53, 47, 0.24);
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: #dfeadf;
            color: #17342d;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div {
            color: #17342d;
        }
        [role="listbox"] *,
        [role="option"] * {
            color: #17342d !important;
        }
        [data-testid="stSidebar"] .stButton button {
            background: #ecd487;
            color: #17342d;
            border: 1px solid rgba(25, 53, 47, 0.24);
            font-weight: 600;
        }
        .stMetric {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(25, 53, 47, 0.12);
            border-radius: 14px;
            padding: 0.4rem 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
