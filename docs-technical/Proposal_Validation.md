# Proposal Validation

Proposal validation is implemented in the `guardian` package.

## Runtime Modules

- `guardian.patch_parser`: A-box proposal normalization
- `guardian.tbox_parser`: T-box reform normalization

Both modules expose the same core surface:

- `PatchValidationError`
- `load_schema`
- `normalize_proposal`
- `canonicalize`

The track-diagnosis parser is implemented separately in `guardian.track_parser`; see [Track Diagnosis](./Track_Diagnosis.md).

## A-box Contract

The A-box contract is defined by [schemas/verified_repair_proposal.schema.json](/home/mvazquez/kg-benchmark/schemas/verified_repair_proposal.schema.json).

Current runtime behavior:

- normalizes QIDs and PIDs to canonical uppercase form
- normalizes ops to `SET`, `ADD`, `REMOVE`, or `DELETE_ALL`
- rewrites `REMOVE` without a value into `DELETE_ALL`
- validates ISO dates, ids, finite numeric values, and op cardinality
- coerces common provenance inputs into the canonical list form
- normalizes proposal-level `uncertainty` into `{"confidence": <0.0-1.0>, "notes": ...}` when present
- emits a deterministic `canonical_hash`

## T-box Contract

The T-box contract is defined by [schemas/tbox_reform_proposal.schema.json](/home/mvazquez/kg-benchmark/schemas/tbox_reform_proposal.schema.json).

Current runtime behavior:

- validates `target.pid` and `target.constraint_type_qid`
- restricts `proposal.action` to the supported first-wave schema-reform families
- normalizes `proposal.signature_after` into the same canonical shape used by Stage 2 `constraint_delta`
- when the caller supplies a case-local constraint-family inventory, rejects `target.constraint_type_qid` and `proposal.signature_after[*].constraint_qid` values that are not valid constraint-family QIDs for that task
- coerces common provenance inputs into the canonical list form
- normalizes proposal-level `uncertainty` into `{"confidence": <0.0-1.0>, "notes": ...}` when present
- emits a deterministic `canonical_hash`

Reasoning-floor runs now pass a strict allowlist assembled from:

- repo-known constraint-family QIDs
- the case's historical `repair_target.constraint_delta.changed_constraint_types`
- the property's current world-state constraint inventory

Malformed T-box outputs therefore surface as proposal `parse_error` rows in reasoning-floor manifests instead of being written as normalized T-box proposals.

## Provenance Compatibility

Normalized output schemas now also expose a top-level `uncertainty` object in addition to canonical `provenance`.

The runtime normalizers now accept several common input variants and convert them into that list form:

- string -> `[{"kind":"OTHER","snippet":"..."}]`
- singleton object with `url` -> inferred `WEB`
- singleton object with `node_id` or a `source` that looks like a `Q...`/`P...` id -> inferred `KG`
- singleton object with `revision_id` -> inferred `HISTORY`
- list entries without `kind` -> inferred from the same signals above

Malformed provenance entries are dropped individually when they cannot be normalized, instead of invalidating an otherwise-correct proposal.

The uncertainty normalizer also accepts several compatibility forms:

- bare numeric confidence such as `0.2` -> `{"confidence": 0.2}`
- confidence strings such as `"0.2"` -> `{"confidence": 0.2}`
- qualitative strings such as `"low"`, `"medium"`, or `"high"` -> deterministic numeric scores
- objects with `score` or `probability` instead of `confidence`

Legacy proposals that omit `rationale`, `provenance`, or `uncertainty` still normalize so historical artifacts remain inspectable, but evaluator acceptance now requires all three auditability components.

## Test Coverage

Current parser coverage includes:

- existing A-box parser tests in [tests/test_patch_parser.py](/home/mvazquez/kg-benchmark/tests/test_patch_parser.py)
- T-box parser tests in [tests/test_tbox_parser.py](/home/mvazquez/kg-benchmark/tests/test_tbox_parser.py)
- track-diagnosis parser tests in [tests/test_track_parser.py](/home/mvazquez/kg-benchmark/tests/test_track_parser.py)
