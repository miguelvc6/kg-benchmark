# Implementation Plan: Ferranti-Style T-Box Taxonomy Patch Task

**Artifact status:** implementation plan  
**Target repo:** `miguelvc6/kg-benchmark`  
**Scope:** replace the current strict T-box `signature_after` reconstruction task with a complete, gold-extractable, Ferranti-style T-box repair-taxonomy task while keeping the existing A-box task unchanged.

---

## 0. Design Commitments

This plan is based on the following constraints.

1. **Every T-box instance must be representable.** No selected T-box record may be left without a gold answer under the new schema.
2. **Every output operation must be explained in the prompt.** The prompt may define fields and operations, but it must not encode hidden benchmark labels or dev-derived repair recipes.
3. **Gold answers must be mechanically extracted from the dataset.** The evaluator must not rely on manual labels or LLM interpretation to define the target.
4. **Existing strict signature reconstruction remains available as a diagnostic.** Do not delete or reinterpret prior `signature_after` metrics; demote them from headline T-box metrics to strict historical reconstruction diagnostics.
5. **A-box remains unchanged.** The current `prompt_dev_v4_spec_only` A-box task remains the main A-box prompt.
6. **No new model inference until gold extraction, parser, evaluator, and tests pass.**
7. **No core-result-driven prompt tuning.** Core results may motivate evaluator/reporting corrections, but not prompt optimization.

---

## Step 1 — Create the Taxonomy Mapping Specification

### Goal

Define a local T-box taxonomy derived from Ferranti et al.’s Wikidata repair taxonomy and adapted to the benchmark’s available artifacts.

Ferranti-style T-box repair components to support:

| Taxonomy code | Operation name | Meaning |
|---|---|---|
| `C_MINUS` | `CONSTRAINT_REMOVE` | Remove a property-constraint statement. |
| `C_D` | `CONSTRAINT_DEPRECATE` | Deprecate or deactivate a constraint statement by rank/status. |
| `C_PLUS` | `CONSTRAINT_ADD` | Add a property-constraint statement. Engineering extension for complete diff coverage. |
| `C_REPLACE` | `CONSTRAINT_TYPE_REPLACE` | Replace one constraint family/type with another. Engineering extension; may also be represented as remove+add. |
| `CQ_PLUS` | `CONSTRAINT_QUALIFIER_ADD` | Add a qualifier value to a constraint definition. |
| `CQ_MINUS` | `CONSTRAINT_QUALIFIER_REMOVE` | Remove a qualifier value from a constraint definition. |
| `CQ_REPLACE` | `CONSTRAINT_QUALIFIER_REPLACE` | Replace a qualifier value, equivalent to a remove+add pair on the same qualifier property. |
| `SUBCLASS_PLUS` | `CLASS_HIERARCHY_ADD` | Add a subclass relation that resolves the violation via class hierarchy. |
| `E_PLUS` | `EXCEPTION_ADD` | Add an exception to the constraint. |
| `OTHER` | `OTHER_TBOX_UPDATE` | Schema-level repair not covered by the listed operations. |

### Required files

Create:

```text
docs-technical/TBox_Taxonomy_Patch_Task.md
```

The document must include:

- the taxonomy table above;
- why `C_PLUS`, `C_REPLACE`, and `OTHER` are engineering extensions;
- how coincidental and unknown T-box cases are represented;
- how this task differs from the old strict `signature_after` task;
- which operations are expected to be extractable from current dataset fields;
- which operations require additional mining support, especially `CLASS_HIERARCHY_ADD`.

### Passing satisfaction criteria

- [ ] The taxonomy document exists.
- [ ] Every operation in the proposed answer schema is defined in plain language.
- [ ] The document explicitly states that strict `signature_after` reconstruction is diagnostic, not the new headline T-box task.
- [ ] The document explicitly states which Ferranti-derived operations are currently fully supported by dataset artifacts.
- [ ] The document explicitly states how unsupported-but-theoretically-valid operations are handled.
- [ ] The document contains no hidden prompt-time labels such as `TypeA`, `TypeB`, or `TypeC` as model-visible instructions.

---

## Step 2 — Define the New T-Box Patch Answer Schema

### Goal

Define a JSON schema for a T-box repair patch that is expressive enough to represent all selected T-box records.

### Proposed schema

Create:

```text
schemas/tbox_taxonomy_patch_proposal.schema.json
```

Schema shape:

```json
{
  "case_id": "<copy input id exactly>",
  "schema_decision": "CAUSAL_SCHEMA_REPAIR | NO_CAUSAL_SCHEMA_REPAIR | UNCLEAR_SCHEMA_EVIDENCE",
  "target": {
    "pid": "P...",
    "constraint_type_qid": "Q..."
  },
  "repairs": [
    {
      "repair_op": "CONSTRAINT_REMOVE | CONSTRAINT_DEPRECATE | CONSTRAINT_ADD | CONSTRAINT_TYPE_REPLACE | CONSTRAINT_QUALIFIER_ADD | CONSTRAINT_QUALIFIER_REMOVE | CONSTRAINT_QUALIFIER_REPLACE | CLASS_HIERARCHY_ADD | EXCEPTION_ADD | OTHER_TBOX_UPDATE",
      "taxonomy_code": "C_MINUS | C_D | C_PLUS | C_REPLACE | CQ_PLUS | CQ_MINUS | CQ_REPLACE | SUBCLASS_PLUS | E_PLUS | OTHER",
      "constraint_type_qid": "Q...",
      "qualifier_property_id": "P... or null",
      "added_values": ["Q..." , "P..." , "<literal>", 123],
      "removed_values": ["Q..." , "P..." , "<literal>", 123],
      "old_value": "Q... | P... | <literal> | 123 | null",
      "new_value": "Q... | P... | <literal> | 123 | null",
      "rank_after": "normal | preferred | deprecated | null",
      "snaktype_after": "VALUE | SOMEVALUE | NOVALUE | null",
      "evidence_level": "FAMILY_ONLY | OPERATION_VISIBLE | VALUE_DELTA_VISIBLE"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [
    {
      "kind": "KG | OTHER",
      "node_id": "Q... or P... or null",
      "snippet": "<visible evidence>"
    }
  ],
  "uncertainty": {
    "confidence": 0.0,
    "notes": "<short uncertainty note>"
  }
}
```

### Schema constraints

- `case_id`, `schema_decision`, `target`, `repairs`, `rationale`, `provenance`, and `uncertainty` are required.
- `target.pid` and `target.constraint_type_qid` are required unless `schema_decision` is `UNCLEAR_SCHEMA_EVIDENCE` and no visible family is available; if relaxed, the exception must be explicitly documented.
- `repairs` may be empty only when:
  - `schema_decision = NO_CAUSAL_SCHEMA_REPAIR`, or
  - `schema_decision = UNCLEAR_SCHEMA_EVIDENCE`.
- If `schema_decision = CAUSAL_SCHEMA_REPAIR`, at least one repair should normally be present.
- `taxonomy_code` must match `repair_op`.
- `added_values` and `removed_values` must default to empty lists, not null.
- Placeholder identifiers such as `Q...`, `P...`, `Q0`, empty strings, or `"none"` must be invalid in parsed outputs.
- `evidence_level` is required for each repair.

### Passing satisfaction criteria

- [ ] JSON schema file exists.
- [ ] Schema validates at least one example for every supported operation.
- [ ] Schema rejects placeholder identifiers.
- [ ] Schema allows `repairs = []` for `NO_CAUSAL_SCHEMA_REPAIR` and `UNCLEAR_SCHEMA_EVIDENCE`.
- [ ] Schema rejects `repairs = []` for `CAUSAL_SCHEMA_REPAIR`, unless the project explicitly allows family-only causal repair with empty operation list.
- [ ] Every enum value is documented in `docs-technical/TBox_Taxonomy_Patch_Task.md`.
- [ ] Schema can be referenced from parser tests.

---

## Step 3 — Implement Deterministic Gold Extraction

### Goal

Create a deterministic extractor that maps every selected T-box benchmark record to the new patch schema.

### Required files

Create:

```text
src/lib/tbox_taxonomy_patch_gold.py
src/derive_tbox_taxonomy_patch_gold.py
tests/test_tbox_taxonomy_patch_gold.py
```

### Gold extraction inputs

Use only benchmark-internal historical artifacts:

```text
record.id
record.property
record.track
record.classification.class
record.classification.subtype
record.classification.diagnostics
record.repair_target.kind
record.repair_target.property_revision_id
record.repair_target.constraint_delta.signature_before
record.repair_target.constraint_delta.old_constraints
record.repair_target.constraint_delta.signature_after
record.repair_target.constraint_delta.new_constraints
record.repair_target.constraint_delta.changed_constraint_types
record.qid
record.violation_context
```

Do not use model outputs.

### Extraction algorithm

#### 3.1 Normalize signatures

Normalize each constraint signature into a family-indexed form:

```python
{
  constraint_type_qid: {
    "snaktypes": set(...),
    "ranks": set(...),
    "qualifiers": {
      qualifier_pid: set(normalized_values)
    }
  }
}
```

Value normalization must match the existing strict T-box parser where possible:

- normalize QIDs;
- normalize PIDs;
- normalize numeric/literal atoms;
- sort sets deterministically.

#### 3.2 Diff before/after

For each constraint family:

- before absent, after present → `CONSTRAINT_ADD` / `C_PLUS`;
- before present, after absent → `CONSTRAINT_REMOVE` / `C_MINUS`;
- same family present in both but qualifier values changed → one or more `CONSTRAINT_QUALIFIER_*` repairs;
- rank changed to deprecated/inactive → `CONSTRAINT_DEPRECATE` / `C_D`;
- snaktype/rank changed but not deprecation → `OTHER_TBOX_UPDATE` or dedicated rank/snaktype repair with `taxonomy_code = OTHER`;
- family appears replaced by a different family and the diff is clearly paired → optional `CONSTRAINT_TYPE_REPLACE`; otherwise encode remove+add.

#### 3.3 Qualifier value delta rules

For each qualifier property:

```text
added_values = after_values - before_values
removed_values = before_values - after_values
```

- only added values → `CONSTRAINT_QUALIFIER_ADD`;
- only removed values → `CONSTRAINT_QUALIFIER_REMOVE`;
- both added and removed on same qualifier property → `CONSTRAINT_QUALIFIER_REPLACE`;
- multiple qualifier properties changed → multiple repair entries unless a single entry can represent all without ambiguity.

#### 3.4 Exception additions

Detect `EXCEPTION_ADD` when:

- a constraint qualifier corresponding to exception-to-constraint is added, and
- the added value includes the violating/focus item or an explicitly represented exception value.

If exception qualifier detection is not currently possible, record this in the gold summary and route such cases through `OTHER_TBOX_UPDATE` until support is added.

#### 3.5 Class hierarchy additions

Detect `CLASS_HIERARCHY_ADD` only if the dataset contains class-hierarchy deltas relevant to the repair.

If the current benchmark does not mine class-hierarchy deltas:

- do not hallucinate them;
- mark `CLASS_HIERARCHY_ADD` as supported by schema but not currently extractable;
- ensure no selected T-box record requires it for coverage;
- if some selected record appears to require it, mark as blocker.

#### 3.6 Schema decision extraction

Set:

```text
CAUSAL_SCHEMA_REPAIR
```

for causal T-box repair subtypes such as:

- `RELAXATION_SET_EXPANSION`;
- `RESTRICTION_SET_CONTRACTION`;
- `RELAXATION_RANGE_WIDENED`;
- `RESTRICTION_RANGE_NARROWED`;
- `SCHEMA_UPDATE` when classified as causal.

Set:

```text
NO_CAUSAL_SCHEMA_REPAIR
```

for `COINCIDENTAL_SCHEMA_CHANGE`.

Set:

```text
UNCLEAR_SCHEMA_EVIDENCE
```

for unknown, weak, missing, or ambiguous T-box causality.

#### 3.7 Evidence level gold

For each repair:

```text
VALUE_DELTA_VISIBLE
```

if concrete added/removed values are present in the gold patch.

```text
OPERATION_VISIBLE
```

if the operation/family is extractable but concrete values are not.

```text
FAMILY_ONLY
```

if only the target constraint family is extractable.

### CLI command

The extractor should support:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/derive_tbox_taxonomy_patch_gold.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --selection-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --out-jsonl reports/gold/tbox_taxonomy_patch_gold_core_v1.jsonl \
  --out-summary reports/gold/tbox_taxonomy_patch_gold_core_v1_summary.json \
  --require-coverage
```

Also support dev validation:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/derive_tbox_taxonomy_patch_gold.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --selection-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --out-jsonl reports/gold/tbox_taxonomy_patch_gold_dev_holdout_v1.jsonl \
  --out-summary reports/gold/tbox_taxonomy_patch_gold_dev_holdout_v1_summary.json \
  --require-coverage
```

### Gold summary requirements

The summary JSON must include:

```text
selected_records
selected_tbox_records
gold_extracted
unsupported_count
unsupported_case_ids
by_schema_decision
by_repair_op
by_taxonomy_code
by_constraint_type_qid
by_qualifier_property_id
by_evidence_level
value_delta_available_count
empty_repairs_count
class_hierarchy_delta_supported
exception_delta_supported
```

### Passing satisfaction criteria

- [ ] Gold extractor runs on dev holdout.
- [ ] Gold extractor runs on core manifest.
- [ ] `unsupported_count == 0` for core selected T-box records.
- [ ] Every selected T-box record has a gold patch.
- [ ] Summary reports all required distributions.
- [ ] Unit tests cover all operation types that appear in the selected core.
- [ ] If `CLASS_HIERARCHY_ADD` or `EXCEPTION_ADD` cannot be extracted from current data, this is explicitly documented and no selected T-box record is incorrectly forced into those operations.
- [ ] Running with `--require-coverage` exits nonzero if any T-box record lacks a gold patch.

---

## Step 4 — Implement the T-Box Taxonomy Patch Parser

### Goal

Parse and normalize model outputs for the new T-box task without breaking the existing strict T-box parser.

### Required files

Create:

```text
src/guardian/tbox_taxonomy_patch_parser.py
tests/test_tbox_taxonomy_patch_parser.py
```

### Parser responsibilities

The parser must normalize:

```text
case_id
schema_decision
target.pid
target.constraint_type_qid
repairs[*].repair_op
repairs[*].taxonomy_code
repairs[*].constraint_type_qid
repairs[*].qualifier_property_id
repairs[*].added_values
repairs[*].removed_values
repairs[*].old_value
repairs[*].new_value
repairs[*].rank_after
repairs[*].snaktype_after
repairs[*].evidence_level
rationale
provenance
uncertainty
```

### Validation rules

- Reject invalid PIDs/QIDs.
- Reject placeholders (`P...`, `Q...`, `Q0`, empty values).
- Reject unknown enum values.
- Ensure `taxonomy_code` matches `repair_op`.
- Ensure `target.constraint_type_qid` and `repairs[*].constraint_type_qid` are constraint-family QIDs when an allowed set is supplied.
- Allow empty `added_values`/`removed_values`.
- Allow empty `repairs` only under allowed `schema_decision` values.
- Canonicalize repair list ordering.
- Preserve metadata only if explicitly allowed; do not let model-supplied hidden metadata affect scoring.

### Passing satisfaction criteria

- [ ] Parser accepts valid examples for every operation.
- [ ] Parser rejects placeholder identifiers.
- [ ] Parser rejects invalid enum values.
- [ ] Parser rejects `CAUSAL_SCHEMA_REPAIR` with empty `repairs` unless explicitly allowed by design.
- [ ] Parser accepts `NO_CAUSAL_SCHEMA_REPAIR` with empty `repairs`.
- [ ] Parser canonicalization is deterministic.
- [ ] Existing strict `tbox_parser.py` tests still pass.
- [ ] Existing A-box parser tests still pass.

---

## Step 5 — Implement T-Box Patch Evaluation

### Goal

Evaluate model outputs against the extracted taxonomy-patch gold.

### Required modifications

Update or add evaluator code in:

```text
src/guardian/evaluator.py
src/guardian/tbox_taxonomy_patch_evaluator.py
tests/test_tbox_taxonomy_patch_evaluator.py
```

### Evaluation layers

#### 5.1 Contract and parse layer

Metrics:

```text
tbox_patch_parse_rate
tbox_patch_contract_valid_rate
tbox_patch_parse_error_rate
```

#### 5.2 Target layer

Metrics:

```text
tbox_patch_target_pid_match_rate
tbox_patch_primary_constraint_family_match_rate
tbox_patch_any_changed_family_hit_rate
tbox_patch_constraint_family_precision
tbox_patch_constraint_family_recall
tbox_patch_constraint_family_f1
```

#### 5.3 Decision layer

Metrics:

```text
tbox_patch_schema_decision_match_rate
tbox_patch_no_causal_schema_repair_match_rate
tbox_patch_unclear_schema_evidence_match_rate
```

#### 5.4 Taxonomy operation layer

Metrics:

```text
tbox_patch_repair_op_exact_match_rate
tbox_patch_taxonomy_code_exact_match_rate
tbox_patch_repair_op_precision
tbox_patch_repair_op_recall
tbox_patch_repair_op_f1
```

#### 5.5 Value-delta layer

Only where gold value deltas exist:

```text
tbox_patch_qualifier_property_match_rate
tbox_patch_added_values_precision
tbox_patch_added_values_recall
tbox_patch_added_values_f1
tbox_patch_removed_values_precision
tbox_patch_removed_values_recall
tbox_patch_removed_values_f1
tbox_patch_value_delta_f1_when_applicable
```

#### 5.6 Evidence-level calibration

Metrics:

```text
tbox_patch_evidence_level_exact_match_rate
tbox_patch_value_delta_claimed_when_gold_absent_rate
tbox_patch_family_only_when_value_delta_gold_present_rate
```

#### 5.7 Composite success metrics

Define:

```text
tbox_patch_family_level_success
```

Passes if:

- target PID matches;
- at least one gold changed/target family is hit.

```text
tbox_patch_decision_level_success
```

Passes if:

- family-level success;
- `schema_decision` matches.

```text
tbox_patch_taxonomy_level_success
```

Passes if:

- decision-level success;
- taxonomy operation set matches or overlaps according to defined multi-edit matching.

```text
tbox_patch_value_delta_success
```

Passes if:

- taxonomy-level success;
- qualifier and added/removed value deltas match when applicable.

#### 5.8 Strict signature diagnostics

Keep existing metrics as diagnostics:

```text
strict_tbox_exact_action_match_rate
strict_tbox_exact_signature_match_rate
strict_tbox_signature_jaccard
```

Do not use them as the main T-box headline for taxonomy-patch runs.

### Denominator policy

Every metric must report:

```text
numerator
applicable_denominator
total_tbox_rows
applicability_coverage
```

No T-box metric with small applicability should be shown without coverage.

### Passing satisfaction criteria

- [ ] Evaluator can score a synthetic example for every operation.
- [ ] Evaluator reports numerator and denominator for every new metric.
- [ ] Evaluator separates all-core, main-score, and diagnostic subsets.
- [ ] Evaluator does not use strict `signature_after` exactness as the default T-box headline for taxonomy-patch runs.
- [ ] Existing strict T-box evaluation still works for old outputs.
- [ ] Tests cover multi-edit gold and prediction matching.
- [ ] Tests cover empty-repair `NO_CAUSAL_SCHEMA_REPAIR` and `UNCLEAR_SCHEMA_EVIDENCE`.

---

## Step 6 — Add Prompt Template for the New T-Box Task

### Goal

Define a model-visible prompt that asks for the taxonomy-patch schema and explains every operation without leaking benchmark metadata.

### Required modifications

Update:

```text
scripts/prompt_dev_templates.py
src/guardian/prompts.py
docs-technical/Prompt_Development.md
docs-technical/Reasoning_Floor.md
```

Add prompt version:

```text
prompt_dev_v5_tbox_taxonomy_patch
```

Add reasoning-floor prompt:

```text
reasoning_floor_t_box_taxonomy_patch_zero_shot
```

### Prompt requirements

The prompt must define:

- `schema_decision`;
- `target.pid`;
- `target.constraint_type_qid`;
- `repair_op`;
- `taxonomy_code`;
- `qualifier_property_id`;
- `added_values`;
- `removed_values`;
- `old_value`;
- `new_value`;
- `rank_after`;
- `snaktype_after`;
- `evidence_level`;
- `rationale`;
- `provenance`;
- `uncertainty`.

### Prompt safety rules

The prompt must state:

- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary item/type values.
- Do not copy report-violation QIDs into value deltas unless visibly presented as schema values.
- Use empty added/removed lists when concrete changed values are not visible.
- Do not construct a full post-repair `signature_after`.
- Do not use hidden benchmark classes, subtypes, historical labels, or `repair_target`.
- Do not use raw case-id prefixes.

### Prompt operation definitions

The prompt must include concise definitions for all operations:

```text
CONSTRAINT_REMOVE
CONSTRAINT_DEPRECATE
CONSTRAINT_ADD
CONSTRAINT_TYPE_REPLACE
CONSTRAINT_QUALIFIER_ADD
CONSTRAINT_QUALIFIER_REMOVE
CONSTRAINT_QUALIFIER_REPLACE
CLASS_HIERARCHY_ADD
EXCEPTION_ADD
OTHER_TBOX_UPDATE
```

### Passing satisfaction criteria

- [ ] Rendered prompt includes definitions for every operation.
- [ ] Rendered prompt includes the full output contract.
- [ ] Rendered prompt does not include hidden TypeA/TypeB/TypeC labels.
- [ ] Rendered prompt does not include `classification`, `repair_target`, `persistence_check`, popularity, raw `repair_`/`reform_` IDs, or temporal-policy labels.
- [ ] A-box prompt remains unchanged.
- [ ] Track diagnosis remains disabled for main evaluation.
- [ ] Rendered prompt review passes leakage scan.

---

## Step 7 — Integrate New T-Box Task into Prompt Dev and Reasoning Floor

### Goal

Allow mixed A-box v4 and T-box taxonomy-patch evaluation in the existing pipeline.

### Required behavior

For oracle repair mode:

```text
A_BOX records -> existing A-box v4 spec-only prompt and A-box parser/evaluator
T_BOX records -> new T-box taxonomy-patch prompt and parser/evaluator
```

### Required modifications

Update as needed:

```text
src/lib/prompt_dev.py
src/reasoning_floor.py
src/guardian/reasoning.py
src/guardian/evaluator.py
```

Add explicit output paths for T-box taxonomy patch proposals, for example:

```text
t_box_taxonomy_patch_proposals.jsonl
```

Do not overwrite old strict:

```text
t_box_proposals.jsonl
```

unless a compatibility layer is explicitly implemented.

### Run config requirements

Run config must record:

```text
tbox_task_version: tbox_taxonomy_patch_v1
abox_task_version: prompt_dev_v4_spec_only
prompt_version: prompt_dev_v5_tbox_taxonomy_patch
strict_tbox_signature_diagnostic: enabled/disabled
```

### Passing satisfaction criteria

- [ ] Mixed A-box/T-box prompt-dev rendering works.
- [ ] Mixed A-box/T-box reasoning-floor generation works.
- [ ] Output files clearly distinguish strict T-box proposals from taxonomy-patch proposals.
- [ ] Run config records new task version.
- [ ] Resume behavior works without duplicating rows.
- [ ] Existing strict T-box pipeline still works for old runs.
- [ ] Existing A-box pipeline is unchanged.

---

## Step 8 — Add Tests and Static Validation

### Goal

Prevent regressions before any model inference.

### Required tests

Add or update:

```text
tests/test_tbox_taxonomy_patch_gold.py
tests/test_tbox_taxonomy_patch_parser.py
tests/test_tbox_taxonomy_patch_evaluator.py
tests/test_prompt_dev.py
tests/test_reasoning_floor.py
```

### Required test cases

Gold extractor:

- qualifier add;
- qualifier remove;
- qualifier replace;
- constraint remove;
- constraint deprecate;
- constraint add;
- no causal schema repair;
- unclear schema evidence;
- unknown qualifier property fallback to `OTHER`.

Parser:

- valid payload with one repair;
- valid payload with multiple repairs;
- valid no-causal empty repair;
- invalid placeholder QID/PID;
- invalid enum;
- mismatched repair_op/taxonomy_code.

Evaluator:

- exact taxonomy match;
- partial family match;
- wrong schema decision;
- wrong repair op;
- value-delta precision/recall cases;
- small-denominator coverage reporting.

Prompt:

- no leakage;
- every operation definition present;
- A-box prompt unchanged.

### Required command

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest \
  tests/test_prompt_dev.py \
  tests/test_model_provider.py \
  tests/test_reasoning_floor.py \
  tests/test_track_parser.py \
  tests/test_tbox_parser.py \
  tests/test_patch_parser.py \
  tests/test_tbox_taxonomy_patch_gold.py \
  tests/test_tbox_taxonomy_patch_parser.py \
  tests/test_tbox_taxonomy_patch_evaluator.py
```

### Passing satisfaction criteria

- [ ] All tests pass.
- [ ] Gold extractor coverage test passes on synthetic cases.
- [ ] Parser rejects invalid placeholders.
- [ ] Evaluator reports denominators and coverage.
- [ ] No existing A-box tests regress.
- [ ] No existing strict T-box tests regress.

---

## Step 9 — Derive and Freeze Gold Patch Artifacts

### Goal

Generate gold patch files for dev and core before running LLM inference.

### Required artifacts

```text
reports/gold/tbox_taxonomy_patch_gold_dev_holdout_v1.jsonl
reports/gold/tbox_taxonomy_patch_gold_dev_holdout_v1_summary.json
reports/gold/tbox_taxonomy_patch_gold_core_v1.jsonl
reports/gold/tbox_taxonomy_patch_gold_core_v1_summary.json
```

### Required commands

Dev:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/derive_tbox_taxonomy_patch_gold.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --selection-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --out-jsonl reports/gold/tbox_taxonomy_patch_gold_dev_holdout_v1.jsonl \
  --out-summary reports/gold/tbox_taxonomy_patch_gold_dev_holdout_v1_summary.json \
  --require-coverage
```

Core:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/derive_tbox_taxonomy_patch_gold.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --selection-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --out-jsonl reports/gold/tbox_taxonomy_patch_gold_core_v1.jsonl \
  --out-summary reports/gold/tbox_taxonomy_patch_gold_core_v1_summary.json \
  --require-coverage
```

### Passing satisfaction criteria

- [ ] Dev gold extraction succeeds.
- [ ] Core gold extraction succeeds.
- [ ] `unsupported_count == 0` for both.
- [ ] Summary includes all required distributions.
- [ ] Artifacts are committed before inference.
- [ ] Any use of `OTHER_TBOX_UPDATE` is quantified and reviewed.
- [ ] If `OTHER_TBOX_UPDATE` dominates, schema refinement is required before prompt evaluation.

---

## Step 10 — Dev-Only LLM Validation

### Goal

Validate the new T-box task on dev before any core rerun.

### Run configuration

Use:

```text
A-box: existing v4 spec-only
T-box: taxonomy-patch v1
track mode: oracle
diagnosis_routed: disabled
abstention: disabled
contexts: logic_only, local_graph
model: same local model as G3, e.g. gpt-oss:120b
```

### Suggested command shape

```bash
PROMPT_DEV_VERSION=prompt_dev_v5_tbox_taxonomy_patch \
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/prompt_dev.py evaluate \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --dev-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_holdout96_zero_shot \
  --model-endpoint ollama \
  --max-cases 96 \
  --sample-strategy manifest_order \
  --representations hybrid_json_nl \
  --example-policies zero_shot \
  --context-bundles logic_only,local_graph \
  --tasks repair_proposal \
  --repair-track-modes oracle
```

### Dev validation gates

Minimum gates:

- [ ] request error rate = 0 or below predeclared threshold;
- [ ] parse error rate ≤ 3%;
- [ ] no prompt leakage;
- [ ] A-box metrics do not regress unexpectedly;
- [ ] T-box taxonomy-patch contract-valid rate ≥ 95%;
- [ ] T-box family-level success is non-null and interpretable;
- [ ] no operation enum dominates due to prompt misunderstanding unless justified by gold distribution;
- [ ] `OTHER_TBOX_UPDATE` prediction rate is reviewed;
- [ ] T-box metrics report denominators correctly.

### Passing satisfaction criteria

- [ ] Dev evaluation completes.
- [ ] Dev summary generated.
- [ ] T-box taxonomy metrics are non-null.
- [ ] Failure modes are interpretable.
- [ ] No full core inference has been run.

---

## Step 11 — Core T-Box-Only Dry Run

### Goal

Validate the new T-box task on a bounded core subset before rerunning all T-box rows.

### Run configuration

Use core T-box only if the runner supports track filtering. If not, add explicit track filtering.

Recommended:

```text
MAX_CASES=64
TRACK_FILTER=T_BOX
ABLATION_BUNDLES=logic_only,local_graph
PROPOSAL_TRACK_MODE=oracle
```

### Required checks

- parse error rate;
- request error rate;
- run config records new T-box task version;
- generated proposal files are separated from strict T-box outputs;
- evaluator summary includes new metrics;
- no model-visible leakage.

### Passing satisfaction criteria

- [ ] Core T-box dry run completes.
- [ ] New T-box metrics are computed.
- [ ] No output overwrites old G3 artifacts.
- [ ] Request errors below threshold.
- [ ] Parse errors below threshold.
- [ ] No leakage.
- [ ] Summary distinguishes strict-signature diagnostics from taxonomy-patch headline metrics.

---

## Step 12 — Decide Whether to Rerun Core T-Box

### Goal

Decide whether to run the taxonomy-patch task over all core T-box rows.

### Decision options

```text
RUN_CORE_TBOX_TAXONOMY_PATCH
FIX_SCHEMA_OR_EVALUATOR
ABANDON_TBOX_TAXONOMY_PATCH
```

### Run full core T-box only if:

- [ ] gold extraction coverage is 100%;
- [ ] dev validation passed;
- [ ] core dry run passed;
- [ ] evaluator metrics are interpretable;
- [ ] no prompt validity violations were found;
- [ ] compute budget is acceptable;
- [ ] old strict G3 artifacts are preserved.

### Passing satisfaction criteria

- [ ] Decision recorded in a Markdown/JSON report.
- [ ] If proceeding, exact command is recorded.
- [ ] If not proceeding, blockers are listed.

---

## Step 13 — Reporting and Paper Integration

### Goal

Update reporting to distinguish between old strict T-box reconstruction and new taxonomy-patch T-box repair reasoning.

### Required reports

Create:

```text
reports/analysis/tbox_taxonomy_patch_metric_design.md
reports/analysis/tbox_taxonomy_patch_gold_coverage.md
reports/analysis/tbox_taxonomy_patch_dev_validation.md
```

If core rerun is completed:

```text
reports/analysis/tbox_taxonomy_patch_core_results.md
reports/analysis/tbox_taxonomy_patch_core_results.json
```

### Paper-facing metric hierarchy

Use as T-box headline:

```text
tbox_patch_family_level_success
tbox_patch_schema_decision_match_rate
tbox_patch_taxonomy_code_match_rate
tbox_patch_value_delta_f1_when_applicable
```

Use as diagnostics:

```text
strict_exact_signature_match_rate
strict_exact_historical_agreement_rate
signature_after_jaccard
```

### Passing satisfaction criteria

- [ ] Reports explain why the old strict `signature_after` metric is diagnostic.
- [ ] Reports explain the Ferranti-derived taxonomy mapping.
- [ ] Reports include numerator, denominator, and applicability for every T-box metric.
- [ ] Reports separate main-score and diagnostic cases.
- [ ] Reports separate A-box and T-box tasks.
- [ ] Reports do not compare old and new T-box scores as if they were identical tasks.

---

## Step 14 — Governance and Versioning

### Goal

Prevent accidental mixing of strict-signature and taxonomy-patch T-box tasks.

### Required version labels

Use explicit labels:

```text
tbox_task_version = strict_signature_v1
tbox_task_version = taxonomy_patch_v1
prompt_version = prompt_dev_v5_tbox_taxonomy_patch
gold_version = tbox_taxonomy_patch_gold_core_v1
```

### Required artifact separation

Old strict G3 artifacts remain under their existing directories.

New taxonomy-patch outputs must use new directories, for example:

```text
reports/reasoning_floor/ollama_v5_tbox_taxonomy_patch_core_tbox_only/
```

### Passing satisfaction criteria

- [ ] No new run overwrites old strict G3 artifacts.
- [ ] Run config records `tbox_task_version`.
- [ ] Evaluation summaries record `gold_version`.
- [ ] Prompt version is recorded.
- [ ] Documentation warns against comparing strict-signature and taxonomy-patch scores as the same task.

---

## Final Acceptance Checklist

The migration is complete only when all of the following are true:

- [ ] Taxonomy mapping document exists and is reviewed.
- [ ] JSON schema exists and validates examples for every operation.
- [ ] Gold extraction succeeds on dev and core with `unsupported_count == 0`.
- [ ] Parser passes unit tests.
- [ ] Evaluator passes unit tests.
- [ ] Prompt explains every operation.
- [ ] Prompt passes leakage scan.
- [ ] A-box prompt remains unchanged.
- [ ] Dev holdout evaluation passes.
- [ ] Core T-box dry run passes.
- [ ] Decision report approves or rejects full core T-box rerun.
- [ ] Existing strict T-box metrics remain available as diagnostics.
- [ ] New T-box taxonomy-patch metrics have numerator, denominator, and applicability.
- [ ] Documentation is updated.
- [ ] No full core rerun occurs before all prior gates pass.

---

## Recommended Immediate Next Command

Begin with gold extraction, not prompting:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/derive_tbox_taxonomy_patch_gold.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --selection-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --out-jsonl reports/gold/tbox_taxonomy_patch_gold_dev_holdout_v1.jsonl \
  --out-summary reports/gold/tbox_taxonomy_patch_gold_dev_holdout_v1_summary.json \
  --require-coverage
```

If this cannot be implemented with full coverage, the new task is not ready for LLM inference.
