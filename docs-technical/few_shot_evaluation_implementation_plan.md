# Few-Shot Evaluation Implementation Plan

**Project:** WikidataRepairEval / `kg-benchmark`  
**Scope:** Phase G5 few-shot ablations for A-box repair and T-box taxonomy-patch repair.  
**Status target:** Implement few-shot infrastructure and run safe, auditable few-shot evaluations without changing the already frozen zero-shot oracle baseline.

---

## 0. Executive Decisions

Few-shot must be treated as a **separate ablation**, not a replacement for the main zero-shot reasoning-floor result.

### Main evaluation remains

```text
A-box prompt: prompt_dev_v4_spec_only
T-box prompt: prompt_dev_v5_tbox_taxonomy_patch
Track mode: oracle
Examples: zero_shot
Contexts: logic_only, local_graph
```

### Few-shot evaluation should be organized as

```text
Phase G5-A: static diverse few-shot, paper-facing ablation
Phase G5-B: visible-similarity dynamic few-shot, exploratory ablation
Phase G5-C: hidden-metadata matched few-shot, oracle-metadata appendix only
Phase G5-D: diagnosis-routed few-shot rescue, dev-only unless gates pass
```

### Non-goals

Do **not**:

- tune the zero-shot prompt further;
- use core cases as examples;
- select examples from the same evaluation case pool;
- use hidden subtype/action/information-condition metadata for paper-facing example retrieval;
- mix strict-signature T-box examples with taxonomy-patch T-box examples;
- use diagnosis-routed as a main condition unless a diagnosis gate is passed;
- use abstention until parser/evaluator support is implemented and validated.

---

## 1. Define Few-Shot Governance

### Objective

Create a governance document that fixes the scientific interpretation of few-shot experiments before any few-shot inference is run.

### Required artifact

```text
docs-technical/Few_Shot_Evaluation.md
```

### Required content

The document must define:

1. **Few-shot role**
   - Zero-shot oracle remains the main reasoning-floor result.
   - Few-shot is Phase G5 ablation.
   - Static diverse support examples are the first paper-facing few-shot condition.
   - Dynamic visible-similarity retrieval is exploratory unless proven clean.
   - Hidden-metadata matched retrieval is oracle-metadata appendix only.

2. **Example-source policy**
   - Examples must come from `reports/benchmark_selection/dev_prompt_v1_seed_13.json` unless a later document explicitly defines a new support pool.
   - Core cases must never be examples.
   - Core T-box revision groups must never be examples.
   - Raw `repair_...` / `reform_...` IDs must never appear in model-visible prompt text.

3. **Prompt-validity policy**
   - Few-shot inputs must obey the same leakage boundary as zero-shot inputs.
   - Example outputs may contain the correct answer for the example, but not hidden benchmark fields.
   - Example selection logic must be declared.

4. **Few-shot condition taxonomy**
   - `static_diverse_kshot`
   - `visible_similarity_kshot`
   - `matched_metadata_oracle_kshot`
   - `diagnosis_static_kshot`

### Passing satisfaction criteria

- `docs-technical/Few_Shot_Evaluation.md` exists.
- It explicitly states that few-shot is not the main zero-shot floor.
- It forbids core examples.
- It distinguishes paper-facing static examples from hidden-metadata oracle examples.
- It references A-box v4 and T-box taxonomy-patch v5 as the task schemas to use.

---

## 2. Add Explicit Evaluation-Manifest vs Example-Manifest Separation

### Objective

Prevent accidental use of evaluation cases as few-shot examples.

### Required code changes

Extend prompt-dev CLI and library options to distinguish:

```text
--eval-manifest / --dev-manifest
--example-manifest
--core-manifest
--support-set-manifest
```

Recommended semantics:

```text
--eval-manifest:
  cases to evaluate/render.

--example-manifest:
  candidate pool for selecting examples.

--core-manifest:
  forbidden final evaluation cases and groups.

--support-set-manifest:
  fixed preselected support examples. If supplied, selection is static and does not query the example pool except for validation.
```

Backward compatibility:

- Keep `--dev-manifest` as an alias for `--eval-manifest`.
- In reports, record the field as `eval_manifest`, not only `dev_manifest`.

### Required code touchpoints

Likely files:

```text
src/prompt_dev.py
src/lib/prompt_dev.py
tests/test_prompt_dev.py
docs-technical/Prompt_Development.md
docs-technical/Few_Shot_Evaluation.md
```

### Passing satisfaction criteria

- CLI accepts `--example-manifest`.
- CLI accepts `--support-set-manifest`.
- Existing zero-shot commands still work.
- Reports distinguish `eval_manifest` and `example_manifest`.
- If `example_policy != zero_shot` and no `example_manifest` or `support_set_manifest` is provided, the command fails unless `--allow-core-example-risk` is explicitly passed.
- Tests verify that core cases cannot become examples by default.

---

## 3. Define Static Support-Set Manifest Schema

### Objective

Create an auditable, immutable representation of static examples.

### Required artifact

```text
schemas/few_shot_support_set.schema.json
```

### Recommended schema shape

```json
{
  "manifest_type": "few_shot_support_set",
  "manifest_version": "static_support_v1",
  "created_at_utc": "...",
  "source_manifest": "reports/benchmark_selection/dev_prompt_v1_seed_13.json",
  "blocked_manifest": "reports/benchmark_selection/core_v1_seed_13.json",
  "selection_policy": "static_diverse",
  "support_sets": {
    "a_box_repair": [
      {
        "raw_case_id": "repair_...",
        "visible_example_id": "example_a_000001",
        "role": "a_box_clean_rule",
        "task_schema": "a_box_v4_spec_only",
        "notes": "internal only"
      }
    ],
    "t_box_repair": [
      {
        "raw_case_id": "reform_...",
        "visible_example_id": "example_t_000001",
        "role": "tbox_taxonomy_cq_plus",
        "task_schema": "tbox_taxonomy_patch_v1",
        "gold_version": "tbox_taxonomy_patch_gold_core_v1 or dev equivalent",
        "notes": "internal only"
      }
    ],
    "track_diagnosis": [
      {
        "raw_case_id": "...",
        "visible_example_id": "example_d_000001",
        "role": "diagnosis_a_box_or_t_box",
        "task_schema": "track_diagnosis_v1",
        "notes": "internal only"
      }
    ]
  },
  "blocked_overlaps": {
    "core_case_overlap": 0,
    "core_qid_overlap": 0,
    "core_tbox_revision_overlap": 0
  }
}
```

### Passing satisfaction criteria

- Schema exists and validates a generated support manifest.
- Support set manifest contains raw IDs only as internal metadata.
- Support set manifest contains neutral visible example IDs.
- Roles are internal and never rendered into model-visible prompts.
- Schema separates A-box, T-box taxonomy-patch, and diagnosis support sets.

---

## 4. Implement Static Support-Set Generator

### Objective

Select fixed, diverse examples from dev only.

### Required script

```text
src/select_few_shot_support.py
```

### Required inputs

```bash
--classified-benchmark data/04_classified_benchmark.jsonl
--dev-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json
--core-manifest reports/benchmark_selection/core_v1_seed_13.json
--output-dir reports/prompt_dev/few_shot/static_support_v1
--seed 13
```

### Required outputs

```text
reports/prompt_dev/few_shot/static_support_v1/static_support_manifest.json
reports/prompt_dev/few_shot/static_support_v1/support_selection_report.md
reports/prompt_dev/few_shot/static_support_v1/support_selection_report.json
```

### Recommended static support sets

#### A-box static set

Start with **3-shot** as the paper-facing default:

```text
A1: clean rule / constraint rejection / target-required claim
A2: format or literal normalization/pruning
A3: local-evidence case
```

Optional 4th example:

```text
A4: diagnostic / non-visible evidence case
```

Do not include hidden labels in model-visible examples. Internal roles may use labels.

#### T-box taxonomy-patch static set

Use **4-shot**:

```text
T1: CAUSAL_SCHEMA_REPAIR + CQ_PLUS / CONSTRAINT_QUALIFIER_ADD
T2: CAUSAL_SCHEMA_REPAIR + CQ_MINUS or CQ_REPLACE
T3: NO_CAUSAL_SCHEMA_REPAIR with repairs=[]
T4: OTHER_TBOX_UPDATE or FAMILY_ONLY / OPERATION_VISIBLE case
```

Do not include `CLASS_HIERARCHY_ADD` or `EXCEPTION_ADD` examples unless real dev gold exists for them.

#### Diagnosis static set

Only for diagnosis-routed rescue experiments:

```text
D1: A_BOX
D2: T_BOX
optional D3: AMBIGUOUS, only if a principled dev example exists
```

### Selection constraints

Hard exclusions:

```text
same raw case as evaluation case
core case IDs
core T-box revision keys
duplicate focus QID inside support set, if avoidable
duplicate property inside support set, if avoidable
```

For static examples, same property with some core case may be impossible to avoid because the support set is fixed and core is large. The support report must disclose this. For dynamic retrieval, same property should remain excluded by default.

### Passing satisfaction criteria

- Generator creates all required artifacts.
- A-box support examples exist and are valid.
- T-box taxonomy-patch support examples exist and are valid.
- Support examples all come from dev manifest.
- `core_case_overlap = 0`.
- `core_tbox_revision_overlap = 0`.
- Report includes counts by track, class/subtype, T-box taxonomy code, property, QID, and T-box revision key.
- Report explicitly states whether static support examples share properties with any core cases.

---

## 5. Generate Example Output Payloads Safely

### Objective

Ensure few-shot example outputs use the correct answer schema without exposing forbidden fields.

### Required behavior

For each support example:

#### A-box

Expected output must be canonical A-box v4 proposal:

```json
{
  "case_id": "example_a_000001",
  "target": {"qid": "Q...", "pid": "P..."},
  "ops": [...],
  "rationale": "...",
  "provenance": [...],
  "uncertainty": {...}
}
```

#### T-box taxonomy-patch

Expected output must be taxonomy-patch proposal:

```json
{
  "case_id": "example_t_000001",
  "schema_decision": "...",
  "target": {"pid": "P...", "constraint_type_qid": "Q..."},
  "repairs": [...],
  "rationale": "...",
  "provenance": [...],
  "uncertainty": {...}
}
```

#### Diagnosis

Expected output must be track diagnosis:

```json
{
  "case_id": "example_d_000001",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "...",
  "rationale": "..."
}
```

### Forbidden in example inputs and outputs

```text
repair_target
classification
persistence_check
truth_source
truth_tokens
selection_stratum
group_key
selected_case_ids
case_annotations
raw repair_/reform_ IDs
core/dev labels such as DEV_ or CORE_
hidden historical_track
```

### Passing satisfaction criteria

- All support outputs parse under the intended parser.
- A-box examples use A-box schema only.
- T-box examples use taxonomy-patch schema only.
- Diagnosis examples use diagnosis schema only.
- No strict-signature T-box example appears in a taxonomy-patch run.
- Leakage scan over rendered example blocks passes.

---

## 6. Add Static Few-Shot Policy to Prompt-Dev

### Objective

Make static support examples available as a first-class example policy.

### Required policy name

```text
static_diverse_kshot
```

or explicit fixed variants:

```text
static_diverse_2shot
static_diverse_3shot
static_diverse_4shot
```

Recommendation:

- Implement `static_diverse_kshot` with `--example-count`.
- Use support-set manifest roles to choose the first `k` examples for each task.

### Required behavior

For repair proposal:

```text
A-box query -> A-box examples only
T-box taxonomy query -> T-box taxonomy examples only
```

For track diagnosis:

```text
Diagnosis query -> diagnosis examples only
```

Do not mix A-box and T-box repair examples in repair proposal prompts because the output schemas differ.

### Passing satisfaction criteria

- `example_policy=static_diverse_kshot` renders examples from support-set manifest.
- A-box repair prompt never contains T-box taxonomy example outputs.
- T-box taxonomy prompt never contains A-box `ops` example outputs.
- Track diagnosis prompt never contains repair proposal examples.
- If support-set manifest is missing required examples, the render/evaluate command fails with a clear error.
- Existing zero-shot behavior is unchanged.

---

## 7. Add Few-Shot Leakage and Overlap Scans

### Objective

Prevent subtle leakage via example selection and rendering.

### Required artifact per run

```text
few_shot_leakage_scan.json
few_shot_overlap_report.json
few_shot_overlap_report.md
```

### Required checks

#### Model-visible text scan

Check system and user prompt text for:

```text
repair_target
classification
persistence_check
truth_source
truth_tokens
selection_stratum
group_key
selected_case_ids
case_annotations
sitelinks_count
popularity
raw repair_ / reform_ IDs
DEV_
CORE_
historical_track
TypeA
TypeB
TypeC
```

Exception: visible Wikidata label/description text may contain ordinary words such as “classification”; such matches must be manually classified as benign.

#### Example overlap scan

Check:

```text
example raw case id overlaps evaluation raw case id
example qid overlaps evaluation qid
example property overlaps evaluation property
example T-box revision overlaps evaluation T-box revision
example appears in core manifest
example appears in selected core T-box group
```

For static support sets, property overlap may be allowed if disclosed. For dynamic retrieval, property overlap should fail unless `--allow-same-property-examples` is passed.

### Passing satisfaction criteria

- Leakage scan exists for every few-shot render/evaluation run.
- Core case overlap is 0.
- Core T-box revision overlap is 0.
- Raw IDs absent from model-visible text.
- Hidden fields absent from model-visible text.
- Any benign text matches are documented.

---

## 8. Add Few-Shot Reporting

### Objective

Compare few-shot to zero-shot without mixing tasks.

### Required output per run

```text
few_shot_run_config.json
few_shot_delta_vs_zero_shot.md
few_shot_delta_vs_zero_shot.json
```

### Required comparisons

#### A-box

Compare against zero-shot oracle baseline by:

```text
overall A-box accepted
TypeA accepted
TypeB accepted
TypeC accepted
A-box exact value
A-box exact action
A-box regression pass
overdelete rate
empty ops rate
constraint/type-QID-as-value rate
parse error rate
token usage
```

#### T-box taxonomy-patch

Compare against zero-shot taxonomy-patch baseline by:

```text
family-level success
schema-decision match
taxonomy-code match
taxonomy-level success
constraint-family F1
repair-op F1
value-delta F1 when applicable
value-delta claimed when gold absent
family-only when value-delta gold present
out-of-current-gold operation false positive rate
parse error rate
token usage
```

#### Diagnosis, if run

Compare against zero-shot neutral diagnosis:

```text
balanced accuracy
A-box recall
T-box recall
AMBIGUOUS rate
wrong-route rate
parse error rate
```

### Passing satisfaction criteria

- Reports clearly separate A-box and T-box.
- Reports do not aggregate A-box and T-box into one headline.
- Reports distinguish static few-shot from dynamic retrieval.
- Reports include token/cost/latency overhead.
- Reports state whether the few-shot result is paper-facing or exploratory.

---

## 9. Add Tests

### Objective

Lock in safety and correctness before running inference.

### Required test areas

Add or extend tests in:

```text
tests/test_prompt_dev.py
tests/test_few_shot_support.py
tests/test_tbox_taxonomy_patch_parser.py
tests/test_tbox_taxonomy_patch_evaluator.py
```

### Required test cases

1. `example_manifest` is separate from `eval_manifest`.
2. Few-shot without `example_manifest` or `support_set_manifest` fails unless explicitly allowed.
3. Core cases cannot be selected as examples.
4. Core T-box revision groups cannot be selected as examples.
5. Static support examples render with neutral IDs.
6. Raw `repair_` / `reform_` IDs are absent from model-visible example blocks.
7. Hidden fields are absent from example input/output blocks.
8. A-box examples use A-box schema.
9. T-box examples use taxonomy-patch schema.
10. Mixed-schema examples are rejected.
11. Existing zero-shot tests still pass.
12. Existing T-box taxonomy-patch tests still pass.

### Required command

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest \
  tests/test_prompt_dev.py \
  tests/test_model_provider.py \
  tests/test_reasoning_floor.py \
  tests/test_track_parser.py \
  tests/test_tbox_parser.py \
  tests/test_patch_parser.py \
  tests/test_tbox_taxonomy_patch_parser.py \
  tests/test_tbox_taxonomy_patch_evaluator.py \
  tests/test_few_shot_support.py
```

### Passing satisfaction criteria

- All tests pass.
- Test suite includes at least one failure test for accidental core example leakage.
- Test suite includes at least one T-box taxonomy-patch few-shot rendering test.

---

## 10. Render Static Few-Shot Review Pack

### Objective

Review prompts manually before model inference.

### Required command shape

```bash
PROMPT_DEV_VERSION=prompt_dev_v5_tbox_taxonomy_patch \
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/prompt_dev.py render \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --eval-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --example-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --support-set-manifest reports/prompt_dev/few_shot/static_support_v1/static_support_manifest.json \
  --output-dir reports/prompt_dev/few_shot/rendered_static_v1_holdout96 \
  --max-cases 96 \
  --sample-strategy manifest_order \
  --representations hybrid_json_nl \
  --example-policies static_diverse_kshot \
  --example-count 4 \
  --context-bundles logic_only,local_graph \
  --tasks repair_proposal \
  --repair-track-modes oracle
```

### Required artifacts

```text
reports/prompt_dev/few_shot/rendered_static_v1_holdout96/prompt_dev_rendered_prompts.jsonl
reports/prompt_dev/few_shot/rendered_static_v1_holdout96/prompt_dev_prompt_review.md
reports/prompt_dev/few_shot/rendered_static_v1_holdout96/prompt_dev_render_summary.json
reports/prompt_dev/few_shot/rendered_static_v1_holdout96/few_shot_leakage_scan.json
reports/prompt_dev/few_shot/rendered_static_v1_holdout96/few_shot_overlap_report.md
reports/prompt_dev/few_shot/rendered_static_v1_holdout96/few_shot_overlap_report.json
```

### Passing satisfaction criteria

- Rendered prompts contain examples.
- Example blocks are schema-correct.
- Leakage scan passes.
- Overlap scan passes.
- Prompt review confirms A-box queries receive only A-box examples.
- Prompt review confirms T-box queries receive only T-box taxonomy-patch examples.
- Prompt length is within model context limits.

---

## 11. Run Static Few-Shot Dev Holdout Evaluation

### Objective

Validate static few-shot on dev before core.

### Required command shape

```bash
PROMPT_DEV_VERSION=prompt_dev_v5_tbox_taxonomy_patch \
OLLAMA_MODEL=gpt-oss:120b \
OLLAMA_TIMEOUT_SECONDS=300 \
OLLAMA_MAX_RETRIES=0 \
OLLAMA_TEMPERATURE=0 \
OLLAMA_CONTEXT_LENGTH=16384 \
OLLAMA_MAX_OUTPUT_TOKENS=2048 \
OLLAMA_KEEP_ALIVE=30m \
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/prompt_dev.py evaluate \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --eval-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --example-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --support-set-manifest reports/prompt_dev/few_shot/static_support_v1/static_support_manifest.json \
  --output-dir reports/prompt_dev/few_shot/evaluation_static_v1_holdout96 \
  --model-endpoint ollama \
  --model gpt-oss:120b \
  --max-cases 96 \
  --sample-strategy manifest_order \
  --representations hybrid_json_nl \
  --example-policies static_diverse_kshot \
  --example-count 4 \
  --context-bundles logic_only,local_graph \
  --tasks repair_proposal \
  --repair-track-modes oracle \
  --no-progress
```

### Required artifacts

```text
reports/prompt_dev/few_shot/evaluation_static_v1_holdout96/prompt_dev_evaluation_summary.json
reports/prompt_dev/few_shot/evaluation_static_v1_holdout96/prompt_dev_evaluation_comparison.md
reports/prompt_dev/few_shot/evaluation_static_v1_holdout96/few_shot_delta_vs_zero_shot.md
reports/prompt_dev/few_shot/evaluation_static_v1_holdout96/few_shot_delta_vs_zero_shot.json
```

### Passing satisfaction criteria

- Request error rate <= 1%.
- Proposal parse error rate does not regress beyond an agreed threshold, default <= 4%.
- Leakage scan passes.
- A-box and T-box outputs normalize under the correct schema.
- T-box taxonomy metrics are emitted for T-box examples.
- Delta report compares static few-shot to zero-shot baseline.
- If performance drops significantly while token cost rises significantly, stop before core canary and analyze failure modes.

---

## 12. Run Static Few-Shot Core Canary

### Objective

Test core readiness without full cost.

### Recommended sample

```text
128 A-box rows
128 T-box rows
logic_only + local_graph
oracle mode
```

Use either a small fixed canary manifest or `--track-filter` runs.

### Required output

```text
reports/prompt_dev/few_shot/evaluation_static_v1_core_canary256/
```

### Passing satisfaction criteria

- 100% prompt rendering.
- Request error rate <= 1%.
- Proposal parse error rate <= 4%.
- Leakage scan passes.
- No core examples appear in example blocks.
- A-box and T-box schemas remain separated.
- Token usage fits compute budget.
- Delta vs zero-shot canary is reported.

---

## 13. Run Full Static Few-Shot Phase G5 Ablation

### Objective

Run the paper-facing few-shot ablation after dev and canary gates pass.

### Recommended scope

Start with:

```text
core main-score subset only
oracle mode
logic_only + local_graph
static diverse examples
```

Then optionally extend to full selected core including diagnostics.

### Required output

```text
reports/reasoning_floor/few_shot_static_v1_oracle_core/
reports/analysis/few_shot_static_v1_core_results.md
reports/analysis/few_shot_static_v1_core_results.json
```

### Passing satisfaction criteria

- Full run completes or resumes cleanly.
- Request errors and parse errors are within gates.
- Leakage/overlap reports pass.
- A-box and T-box are reported separately.
- T-box taxonomy-patch metrics use the taxonomy metric family, not strict-signature headlines.
- Token/cost overhead is reported.
- Result is labeled as Phase G5 few-shot ablation, not main zero-shot floor.

---

## 14. Implement Visible-Similarity Dynamic Retrieval

### Objective

Test whether dynamic retrieval improves performance without hidden metadata.

### Allowed retrieval features

Only use model-visible features:

```text
property label
property description
violation report text
constraint labels visible in prompt
qualifier property labels visible in prompt
visible value datatype or string shape
rendered prompt text lexical similarity
```

### Forbidden retrieval features

Do not use:

```text
classification class/subtype
TypeA/TypeB/TypeC
repair_target
gold target values
taxonomy_code
schema_decision gold
information condition
selection_stratum
truth_source/truth_tokens
core labels
```

### Required implementation

Add policy:

```text
visible_similarity_kshot
```

Recommended simple retrieval first:

```text
TF-IDF or BM25 over sanitized rendered example inputs
balanced by task schema
hard exclusions applied before scoring
```

For T-box taxonomy patch, do not select by gold taxonomy code unless this is labeled as oracle metadata retrieval.

### Passing satisfaction criteria

- Retrieval index is built only from sanitized visible example inputs.
- Retrieval code has tests proving hidden metadata is not used.
- Dynamic examples exclude core cases and core groups.
- Dynamic examples exclude same QID and same T-box revision.
- Same-property examples are excluded by default.
- Dev holdout evaluation report exists before any core run.
- If dynamic retrieval is run on core, it is labeled exploratory unless separately approved.

---

## 15. Optional Hidden-Metadata Matched Oracle Ablation

### Objective

Measure the upper-bound benefit of oracle example matching.

### Policy name

```text
matched_metadata_oracle_kshot
```

### Allowed only if clearly labeled

This policy may use:

```text
track
constraint family
subtype/action
information condition
value datatype
popularity bucket
```

but must be labeled:

```text
oracle-metadata example retrieval
not paper-facing main condition
appendix / exploratory
```

### Passing satisfaction criteria

- Hidden-metadata retrieval is impossible to confuse with paper-facing static or visible-similarity retrieval.
- Reports clearly label it as oracle-metadata ablation.
- It is not used as the main few-shot result.
- It never uses core cases as examples.

---

## 16. Optional Diagnosis Few-Shot Rescue

### Objective

Test whether few-shot examples improve diagnosis-routed viability.

### Preconditions

The zero-shot diagnosis result for `gpt-oss:120b` failed, with low T-box recall and below-gate balanced accuracy. Diagnosis-routed must remain disabled unless few-shot diagnosis passes gates.

### Recommended experiment

```text
diagnosis_static_2shot:
  one A_BOX example
  one T_BOX example
```

Use neutral diagnosis context bundles:

```text
diagnosis_minimal
diagnosis_logic_neutral
diagnosis_local_neutral
```

### Gates

Proceed to a routed dev canary only if:

```text
balanced_accuracy >= 0.65
T-box recall >= 0.50
AMBIGUOUS rate <= 0.20
parse error rate <= 0.04
request error rate <= 0.01
```

Consider Phase G diagnosis-routed ablation only if:

```text
balanced_accuracy >= 0.70
T-box recall >= 0.65
```

### Passing satisfaction criteria

- Diagnosis examples come from dev only.
- Diagnosis examples do not leak hidden labels beyond the example output label.
- Diagnosis-only evaluation passes the gates before any routed canary.
- If gates fail, diagnosis-routed remains stopped for this model.

---

## 17. Final Few-Shot Readiness Review

### Objective

Gate full few-shot core runs.

### Required artifact

```text
reports/readiness/few_shot_readiness_review.md
reports/readiness/few_shot_readiness_review.json
```

### Required checks

- Governance doc exists.
- Example-source separation is implemented.
- Static support set exists and validates.
- Leakage scan passes.
- Overlap scan passes.
- Dev holdout few-shot run completed.
- Core canary completed.
- Tests pass.
- Few-shot result is labeled as ablation.
- Dynamic/hidden-metadata policies are separated from paper-facing static few-shot.

### Verdict options

```text
READY_FOR_STATIC_FEW_SHOT_CORE
READY_FOR_VISIBLE_SIMILARITY_DEV_ONLY
READY_FOR_ORACLE_METADATA_ABLATION_DEV_ONLY
BLOCKED
```

### Passing satisfaction criteria

- Review returns `READY_FOR_STATIC_FEW_SHOT_CORE` before full static core run.
- Any unresolved leakage or support-set issue returns `BLOCKED`.
- Diagnosis-routed few-shot is not approved unless diagnosis gates pass.

---

## 18. Paper Reporting Plan

### Objective

Ensure few-shot results are presented honestly.

### Required reporting language

Use separate table sections:

```text
Main zero-shot oracle result
Static few-shot oracle ablation
Visible-similarity retrieval ablation, if run
Oracle-metadata retrieval ablation, if run
Diagnosis-routed rescue experiment, if run
```

### Required metrics

#### A-box

```text
accepted / exact value
exact action
regression pass
TypeA / TypeB / TypeC split
parse errors
empty ops
overdelete
constraint-QID-as-value
```

#### T-box taxonomy patch

```text
family-level success
schema-decision match
taxonomy-code match
taxonomy-level success
constraint-family F1
repair-op F1
value-delta F1 when applicable
value-delta false positive
family-only when value-delta gold present
out-of-current-gold operation false positives
```

#### Cost

```text
prompt tokens
completion tokens
latency
throughput
context-window failures
```

### Passing satisfaction criteria

- Few-shot is never merged into a single headline with zero-shot.
- A-box and T-box are not merged into a single undifferentiated score.
- T-box strict-signature metrics are not used as taxonomy-patch headline metrics.
- Claims explicitly state whether examples are static, visible-similarity, or oracle-metadata selected.

---

## Implementation Dependency Graph

```text
Step 1  Governance
  ↓
Step 2  Eval/example manifest separation
  ↓
Step 3  Support-set schema
  ↓
Step 4  Static support-set generator
  ↓
Step 5  Safe example output payloads
  ↓
Step 6  Static few-shot prompt policy
  ↓
Step 7  Leakage/overlap scans
  ↓
Step 8  Reporting
  ↓
Step 9  Tests
  ↓
Step 10 Static render review
  ↓
Step 11 Dev holdout evaluation
  ↓
Step 12 Core canary
  ↓
Step 13 Full static few-shot ablation
```

Optional branches:

```text
Step 14 visible-similarity dynamic retrieval
Step 15 hidden-metadata oracle ablation
Step 16 diagnosis few-shot rescue
```

---

## Final Success Definition

The few-shot system is ready for full Phase G5 static core evaluation only when all are true:

```text
examples are selected from dev only
core example overlap is impossible by construction
support set is frozen and versioned
A-box examples use A-box schema
T-box examples use taxonomy-patch schema
model-visible text has no hidden metadata
dev holdout few-shot passes stability gates
core canary passes stability gates
few-shot reports are clearly labeled as ablation
```
