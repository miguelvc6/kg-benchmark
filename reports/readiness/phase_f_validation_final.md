# Final Phase F Validation

Generated: 2026-06-10T06:41:45Z  
Commit: `ffd5665123897681b3d15ff8948b1834b826493e`

## Overall Verdict

**READY_FOR_PHASE_G_DRY_RUN** for the main `prompt_dev_v4_spec_only` no-abstain prompt.

The previous script blocker is fixed and Phase G prompts now follow the v4 spec-only policy more closely. This is still not a full-main approval because Phase G 16/64 dry-run artifacts are not present locally for validation.

## Separate Verdicts

| Area | Verdict |
|---|---|
| Phase F main prompt | READY_FOR_PHASE_G_DRY_RUN |
| Phase F abstention branch | BLOCKED / experimental only |
| v3 scaffolded prompt | Diagnostic ablation only |
| Phase G oracle dry run | Ready with warnings |
| Phase G main full evaluation | Not approved until dry-run artifacts are validated |

## Gate Summary From Existing v4 Holdout

| Gate | Status | Evidence |
|---|---|---|
| Completion | pass | 384/384 prompts evaluated |
| Request errors | pass | 0 request errors |
| Leakage | pass | 0 raw repair/reform IDs, hidden TypeA/B/C, or sitelinks_count |
| Spec-only | pass | Disallowed scaffold terms absent in Phase F v4 artifacts |
| Parse | warning | 6/192 proposal parse errors; 6/384 all prompts |

## Fixes Completed

- **BLOCKER_CLOSED** Phase F wrapper CRLF syntax blocker fixed  
  Evidence: Line endings normalized; bash -n scripts/run_phase_f_v4_ollama_holdout.sh passes.  
  Impact: Phase F wrapper is runnable again.
- **HIGH_CLOSED** Phase G wrapper now refuses accidental full-core runs  
  Evidence: No MAX_CASES exits status 2 unless ALLOW_FULL_CORE_RUN=1; bash -n passes.  
  Impact: Dry-run safety restored for wrapper invocation.
- **HIGH_CLOSED** Reasoning-floor prompts aligned to v4 spec-only policy  
  Evidence: Concrete A-box example and T-box scaffolded template examples removed from src/guardian/prompts.py; tests assert absence of scaffold wording and anchor IDs.  
  Impact: Phase G prompt contract now matches the clean v4 policy more closely.
- **HIGH_CLOSED** Stale scripts/Ollama_VM_Runbook.md reconciled  
  Evidence: scripts/Ollama_VM_Runbook.md is now a thin pointer to docs-technical/Ollama_VM_Runbook.md.  
  Impact: Avoids conflicting VM setup instructions.
- **MEDIUM_CLOSED** Track-diagnosis comparison rows now mark proposal metrics n/a  
  Evidence: prompt-dev comparison renderer emits n/a for Functional/Audit on track_diagnosis rows and n/a for Track acc on repair rows; existing v4 comparison markdown regenerated.  
  Impact: Reduces misleading zero-valued proposal metrics for track-only matrices.
- **INFO** Repository line-ending guard added  
  Evidence: .gitattributes pins *.sh, scripts/*.md, and docs-technical/*.md to LF.  
  Impact: Reduces CRLF regression risk.
- **INFO** Required non-inference tests pass  
  Evidence: 110 passed in 13.40s.  
  Impact: Current patched code paths pass the requested gate.

## Remaining Warnings

- **HIGH** Phase G dry-run artifacts are still absent locally  
  Evidence: reports/reasoning_floor contains no local run artifacts to validate.  
  Impact: Full Phase G main oracle remains unapproved until 16/64 dry-run artifacts exist and are reviewed.
- **HIGH** v4 no-abstain proposal parse rate remains warning-level in existing artifacts  
  Evidence: spec_only_holdout_report parse_gate: 6/192 proposal prompts = 3.125%; 6/384 all prompts = 1.5625%; request errors 0.  
  Impact: Acceptable for bounded dry-run monitoring; not a claim of high-performance or perfect parse stability.
- **BLOCKER** Abstention branch remains experimental and not Phase-G-main ready  
  Evidence: Existing abstention artifacts have 48/96 and 50/96 repair proposal parse errors; docs now state first-class parser/evaluator support is required.  
  Impact: Do not use abstention in Phase G main.

## Verification

```bash
bash -n scripts/run_phase_f_v4_ollama_holdout.sh
bash -n scripts/run_phase_g_ollama_oracle.sh
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest tests/test_prompt_dev.py tests/test_model_provider.py tests/test_reasoning_floor.py tests/test_track_parser.py tests/test_tbox_parser.py tests/test_patch_parser.py
```

Result: **110 passed** and both shell syntax checks passed.

## Do Not Do

- Do not tune prompts against core or dev failures.
- Do not use the abstention branch for Phase G main.
- Do not run full core until dry-run artifacts are validated.
