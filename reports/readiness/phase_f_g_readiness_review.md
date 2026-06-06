# Phase F/G Readiness Review

Created: 2026-06-06

## Overall Verdict

READY WITH WARNINGS.

The repository is ready to continue Phase F prompt development and Phase G dry-run preparation. It is not yet ready to freeze prompts or start paid/main Phase G scoring: the current dev inference run is a small zero-shot candidate run, one endpoint request failed, track diagnosis is only a sanity pass, and T-box semantic-family quality is still weak.

## Blocking Issues

None found after patching the Phase G prompt-visible case-id path and updating stale runbook references.

## Non-Blocking Warnings

- Phase F prompt freeze is premature. The current run covers only `hybrid_json_nl`, `zero_shot`, `logic_only` and `local_graph`, oracle repair proposals, and 24 balanced dev cases.
- One prompt-dev endpoint request failed in the local-graph repair matrix: 1/24 for that matrix, 1/96 overall.
- Track diagnosis sanity is modest: 13/24 for logic-only and 16/24 for local-graph.
- Qwen raw outputs often include `<think>` text, but parser extraction recovered valid JSON in all non-request-error cases.
- T-box proposal parsing is stable, but semantic-family quality remains weak in the current run.
- The rendered-prompt JSONL keeps raw benchmark IDs in internal metadata. This is acceptable for artifacts, but model-visible message text must remain the leakage boundary.

## Files Inspected

- `docs-technical/00-implementation_plan.md`
- `docs-technical/Prompt_Development.md`
- `docs-technical/Reasoning_Floor.md`
- `docs-technical/Paper_Execution_Plan.md`
- `reports/prompt_dev/evaluation_prompt_dev_v1/rendered_prompts/prompt_dev_render_summary.json`
- `reports/prompt_dev/evaluation_prompt_dev_v1/rendered_prompts/prompt_dev_prompt_review.md`
- `reports/prompt_dev/evaluation_prompt_dev_v1/rendered_prompts/prompt_dev_rendered_prompts.jsonl`
- `reports/prompt_dev/evaluation_prompt_dev_v1/prompt_dev_evaluation_summary.json`
- `reports/prompt_dev/evaluation_prompt_dev_v1/prompt_dev_evaluation_comparison.md`
- `reports/manual_audit/audit_phase_d_v1_summary.md`
- `reports/manual_audit/audit_phase_d_v1_policy.md`
- `reports/benchmark_selection/dev_prompt_v1_seed_13.json`
- `reports/benchmark_selection/core_v1_seed_13.json`
- `reports/non_llm_baselines/core_v1_phase_e/run_config.json`
- `src/prompt_dev.py`
- `src/lib/prompt_dev.py`
- `scripts/prompt_dev_templates.py`
- `src/guardian/reasoning.py`
- `src/guardian/prompts.py`
- `src/guardian/model_provider.py`
- `src/reasoning_floor.py`
- `tests/test_prompt_dev.py`
- `tests/test_model_provider.py`
- `tests/test_reasoning_floor.py`
- `tests/test_track_parser.py`
- `tests/test_tbox_parser.py`
- `tests/test_patch_parser.py`

## Commands Run

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest tests/test_prompt_dev.py tests/test_model_provider.py tests/test_reasoning_floor.py tests/test_track_parser.py tests/test_tbox_parser.py tests/test_patch_parser.py
```

Result: 103 passed.

```bash
rg -n "paper_eval_tbox_cap_100|/home/mvazquez|LLM inference not run|minimal_case/evaluation_summary" docs-technical/Paper_Execution_Plan.md docs-technical/00-implementation_plan.md docs-technical/Reasoning_Floor.md
```

Result: no stale matches.

Structured artifact and leakage checks were also run against the rendered prompt JSONL, prompt review Markdown, manifests, and evaluation summary.

## Artifact Consistency

| Area | Status | Evidence |
|---|---:|---|
| Phase D audit policy | PASS | Summary and policy files exist; audit status is ready. |
| Dev/core manifests | PASS | Dev selected 600 cases; core selected 4,800 cases. No dev/core case overlap and no T-box revision overlap found in audit. |
| Core scoring split | PASS | Core has 3,818 `main_score_case_ids` and 982 `diagnostic_case_ids`. |
| Dev prompt split | PASS | Dev has 481 main-style and 119 diagnostic/challenge cases; dev is documented for tuning only. |
| Rendered prompts | PASS | 96 rendered prompts, 24 eval cases, 4 matrix rows, 0 skipped. |
| Track balance | PASS | Render summary has 12 A_BOX and 12 T_BOX cases. |
| Task coverage | PASS | `track_diagnosis`: 48, `a_box_repair`: 24, `t_box_repair`: 24. |
| Evaluation rows | PASS | 96 evaluated prompts, 95 normalized, 1 request_error, 0 parse_error. |
| Per-matrix counts | PASS | Each matrix records normalized, parse_error, request_error, skipped, by historical track, by task, and by context. |
| Comparison Markdown | PASS | `prompt_dev_evaluation_comparison.md` points to the same evaluation run output. |
| Phase E baselines | PASS | `reports/non_llm_baselines/core_v1_phase_e/` exists with run config, summaries, traces, and baseline-specific evaluation outputs. |

## Prompt Leakage Checklist

| Check | Phase F prompt-dev | Phase G reasoning-floor |
|---|---:|---:|
| Raw `repair_` / `reform_` IDs absent from model-visible text | PASS | PASS after patch |
| Neutral visible case IDs used | PASS | PASS after patch |
| Visible IDs mapped back before normalization/evaluation | PASS | PASS after patch |
| `sitelinks_count` absent from local-graph prompt text | PASS | PASS |
| Classification labels and historical track absent from prompt text | PASS | PASS |
| `repair_target`, `persistence_check`, and build metadata absent from prompt text | PASS | PASS |
| Local graph suppresses target-property L3 edges | PASS | PASS |
| T-box context follows pre-reform/compact temporal policy | PASS | PASS |
| T-box prompt forbids copying report QIDs into schema signatures without visible support | PASS | PASS after prompt wording patch |
| Provenance prompt avoids inviting hallucinated `HISTORY` without visible revision evidence | PASS | PASS |
| Few-shot without core manifest fails unless explicitly risk-accepted | PASS | N/A |

## Prompt-Dev Evaluation Snapshot

| Matrix | Normalized | Parse Error | Request Error | Key Metric |
|---|---:|---:|---:|---|
| `logic_only` track diagnosis | 24 | 0 | 0 | accuracy 0.5417 |
| `logic_only` oracle repair | 24 | 0 | 0 | T-box target constraint match 0.9167; A-box exact value match 0.3333 |
| `local_graph` track diagnosis | 24 | 0 | 0 | accuracy 0.6667 |
| `local_graph` oracle repair | 23 | 0 | 1 | T-box target constraint match 0.9167; A-box exact value match 0.6667 |

## Patches Made During Review

- `src/guardian/reasoning.py`: added Phase G neutral prompt-visible case IDs, persisted `visible_case_id_map` in run config, included visible IDs in run manifests/raw rows, and mapped visible IDs back to benchmark IDs before diagnosis/proposal normalization.
- `src/guardian/prompts.py`: removed raw `repair_case`/`reform_case` example IDs from model-visible examples, strengthened JSON-only/no-chain-of-thought instructions, and made the T-box report-QID anti-copy rule explicit.
- `tests/test_reasoning_floor.py`: extended static-provider coverage for neutral ID mapping, raw-output visible IDs, and normalized raw benchmark IDs.
- `docs-technical/00-implementation_plan.md`, `docs-technical/Paper_Execution_Plan.md`, and `docs-technical/Reasoning_Floor.md`: updated stale Phase F/G status, current core/dev manifest references, scoring split guidance, and Phase G command/runbook language.

## Phase F Next-Run Recommendation

Run one larger balanced dev-only candidate before freezing prompts:

- `--max-cases 72` minimum, preferably 96 if endpoint stability is acceptable.
- Keep `hybrid_json_nl`, `zero_shot`, `logic_only,local_graph`, and oracle repair proposals as the next controlled candidate.
- Pass the core manifest even for zero-shot runs, so leakage guards and run metadata are exercised consistently.
- Postpone few-shot until zero-shot parse/request stability is confirmed.
- Test abstention only after the zero-shot prompt has stable parsing and endpoint behavior.

Suggested go/no-go thresholds before Phase G main scoring:

- Proposal parse error rate <= 2 percent, with no T-box nested-object extraction regression.
- Request error rate <= 1 percent after retry/backoff.
- Local-graph track diagnosis macro-F1 or balanced accuracy >= 0.65.
- A-box/T-box proposal normalization >= 98 percent.
- No prompt leakage in rendered prompt text.
- Token budgets stay within endpoint limits with comfortable margin.

## Phase G Dry-Run Recommendation

Before any main scoring run, execute a tiny no-credit/static-provider or existing test-harness dry run that exercises:

- `oracle` repair mode.
- `diagnosis_routed` repair mode.
- Resume from `--resume-run-dir`.
- Synthetic skipped rows for `AMBIGUOUS`.
- Run-config capture of prompt version, context bundle, model/provider endpoint key, sample manifest, selected IDs, and `visible_case_id_map`.

Then run Phase G on `reports/benchmark_selection/core_v1_seed_13.json`, and keep headline scoring filtered to `main_score_case_ids`; diagnostic/challenge cases should be reported separately.
