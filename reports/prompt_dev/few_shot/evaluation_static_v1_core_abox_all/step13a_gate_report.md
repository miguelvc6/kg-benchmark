# Step 13A A-box Static Few-shot Gate Report

Status: `BLOCKED`

The full A-box static few-shot run was started with local Ollama `gpt-oss:120b`, `PROMPT_DEV_VERSION=prompt_dev_v5_tbox_taxonomy_patch`, `static_diverse_kshot` with `--example-count 4`, `track_filter=A_BOX`, and contexts `logic_only,local_graph`.

The harness rendered 8,408 prompt records in memory, then stopped before writing `prompt_dev_rendered_prompts.jsonl` or making model calls because `few_shot_leakage_scan.json` failed.

## Gate Evidence

- Leakage scan: `FAIL` (3 hard matches).
- Overlap scan: `PASS` (core case overlap `0`, core T-box revision overlap `0`).
- Schema outputs: not evaluated; no model responses were generated.
- Request/parse error gates: not evaluated; no model requests were sent.
- Accepted/exact-value improvement gate: not evaluated.
- TypeC hallucination/empty-op behavior: not evaluated.
- Token increase: not evaluated.

## Leakage Matches

- `repair_Q10857009_2354085271` in `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` matched forbidden term `popularity`.
- `repair_Q10857009_2354085271` in `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` matched forbidden term `popularity`.
- `repair_Q65968866_2355096295` in `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` matched forbidden term `popularity`.

Targeted inspection found at least one hard match comes from an ordinary visible value description: `violation_context.value_descriptions_en[0]` for `repair_Q10857009_2354085271` includes "gained broad popularity amongst consumers". The scanner still treats `popularity` as hard-forbidden, so this run cannot pass the requested leakage gate without a scanner policy change and rerun.

## Frozen Zero-shot Baseline

Baseline source: `reports/reasoning_floor/ollama_v4_spec_only_oracle_core/20260616T075205_ollama_gpt_oss_120b`. It contains the same core selection under oracle mode with 4,204 A-box rows per context. Since the few-shot run was blocked before inference, all requested deltas are `null` in the JSON report.

## Verdict

`BLOCKED`
