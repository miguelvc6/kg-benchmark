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

## A-box Contract

The A-box contract is defined by [schemas/verified_repair_proposal.schema.json](/home/mvazquez/kg-benchmark/schemas/verified_repair_proposal.schema.json).

Current runtime behavior:

- normalizes QIDs and PIDs to canonical uppercase form
- normalizes ops to `SET`, `ADD`, `REMOVE`, or `DELETE_ALL`
- rewrites `REMOVE` without a value into `DELETE_ALL`
- validates ISO dates, ids, finite numeric values, and op cardinality
- emits a deterministic `canonical_hash`

## T-box Contract

The T-box contract is defined by [schemas/tbox_reform_proposal.schema.json](/home/mvazquez/kg-benchmark/schemas/tbox_reform_proposal.schema.json).

Current runtime behavior:

- validates `target.pid` and `target.constraint_type_qid`
- restricts `proposal.action` to the supported first-wave schema-reform families
- normalizes `proposal.signature_after` into the same canonical shape used by Stage 2 `constraint_delta`
- emits a deterministic `canonical_hash`

## Test Coverage

Current parser coverage includes:

- existing A-box parser tests in [tests/test_patch_parser.py](/home/mvazquez/kg-benchmark/tests/test_patch_parser.py)
- T-box parser tests in [tests/test_tbox_parser.py](/home/mvazquez/kg-benchmark/tests/test_tbox_parser.py)
