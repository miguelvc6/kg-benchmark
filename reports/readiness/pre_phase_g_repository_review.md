# Pre-Phase-G Repository Review

Generated: 2026-06-10T06:41:45Z  
Commit: `ffd5665123897681b3d15ff8948b1834b826493e`

## Overall Verdict

**READY_FOR_PHASE_G_DRY_RUN**

The script blockers and high-priority prompt-policy/runbook issues have been fixed. The repository is ready for the next bounded oracle dry run, but **not** for full Phase G main oracle scoring because Phase G 16/64 dry-run artifacts are still missing locally and unvalidated.

## Separate Verdicts

| Area | Verdict |
|---|---|
| Phase F main prompt | READY_FOR_PHASE_G_DRY_RUN |
| Phase F abstention branch | BLOCKED / experimental only |
| Phase G oracle dry run | READY_WITH_WARNINGS |
| Phase G diagnosis_routed | NOT READY AS MAIN; ablation only |
| Phase G main full evaluation | NOT READY until dry-run artifacts are validated |

## Fixed Blockers And High Warnings

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

## Command Safety

- `scripts/run_phase_g_ollama_oracle.sh` now refuses to run full selected core unless `ALLOW_FULL_CORE_RUN=1` is set.
- `PROPOSAL_TRACK_MODE` still defaults to `oracle`; `diagnosis_routed` remains opt-in ablation behavior.
- The wrapper passes `--oracle-diagnosis-mode skip` only for oracle mode.

## Verification Commands Run

```bash
bash -n scripts/run_phase_f_v4_ollama_holdout.sh
bash -n scripts/run_phase_g_ollama_oracle.sh
env -i PATH="$PATH" HOME="$HOME" bash scripts/run_phase_g_ollama_oracle.sh
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest tests/test_prompt_dev.py tests/test_model_provider.py tests/test_reasoning_floor.py tests/test_track_parser.py tests/test_tbox_parser.py tests/test_patch_parser.py
```

Results:

- Shell syntax checks passed.
- Full-core guard returned status 2 with a clear refusal message before inference.
- Test suite: **110 passed in 13.40s**.

## Exact Next Commands

Run the bounded oracle dry run on the VM or local Ollama environment:

```bash
MAX_CASES=16 OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_dry_run_16 \
  bash scripts/run_phase_g_ollama_oracle.sh
```

If the 16-case dry run is clean and copied back/validated:

```bash
MAX_CASES=64 OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_g1_64 \
  bash scripts/run_phase_g_ollama_oracle.sh
```

Only after dry-run validation and explicit approval:

```bash
ALLOW_FULL_CORE_RUN=1 OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_core \
  bash scripts/run_phase_g_ollama_oracle.sh
```

## Do Not Do

- Do not run full core yet.
- Do not inspect or tune on core results before prompt freeze.
- Do not run diagnosis-routed as the main Phase G mode.
- Do not use abstention branch as main scoring.
- Do not promote v3 scaffolded to the main prompt.
