# T-Box Taxonomy Patch Gold Coverage

Date: 2026-06-17

## Summary

Gold extraction has complete coverage for both the dev holdout and core manifests.

| Manifest | Selected records | Selected T-box records | Gold extracted | Unsupported | Value-delta available |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dev holdout | 96 | 48 | 48 | 0 | 33 |
| Core | 4800 | 596 | 596 | 0 | 213 |

The extractor does not use model outputs. It derives gold patches from benchmark-internal historical artifacts such as `repair_target.constraint_delta`, changed constraint families, before/after signatures, and violation context.

## Core Gold Distribution

| Field | Distribution |
| --- | --- |
| Schema decision | `CAUSAL_SCHEMA_REPAIR`: 296; `NO_CAUSAL_SCHEMA_REPAIR`: 300 |
| Repair operation | `CONSTRAINT_QUALIFIER_ADD`: 138; `CONSTRAINT_QUALIFIER_REMOVE`: 31; `CONSTRAINT_QUALIFIER_REPLACE`: 64; `OTHER_TBOX_UPDATE`: 73 |
| Taxonomy code | `CQ_PLUS`: 138; `CQ_MINUS`: 31; `CQ_REPLACE`: 64; `OTHER`: 73 |
| Evidence level | `FAMILY_ONLY`: 73; `OPERATION_VISIBLE`: 20; `VALUE_DELTA_VISIBLE`: 213 |
| Qualifier property | `P1793`: 19; `P2305`: 41; `P2306`: 46; `P2308`: 94; `P4155`: 2; `P424`: 31 |

## Dev Holdout Gold Distribution

| Field | Distribution |
| --- | --- |
| Schema decision | `CAUSAL_SCHEMA_REPAIR`: 36; `NO_CAUSAL_SCHEMA_REPAIR`: 12 |
| Repair operation | `CONSTRAINT_QUALIFIER_ADD`: 20; `CONSTRAINT_QUALIFIER_REMOVE`: 14; `CONSTRAINT_QUALIFIER_REPLACE`: 3; `OTHER_TBOX_UPDATE`: 1 |
| Taxonomy code | `CQ_PLUS`: 20; `CQ_MINUS`: 14; `CQ_REPLACE`: 3; `OTHER`: 1 |
| Evidence level | `FAMILY_ONLY`: 1; `OPERATION_VISIBLE`: 4; `VALUE_DELTA_VISIBLE`: 33 |
| Qualifier property | `P1793`: 2; `P2305`: 5; `P2306`: 15; `P2308`: 5; `P424`: 10 |

## Unsupported Operations

`CLASS_HIERARCHY_ADD` and `EXCEPTION_ADD` remain schema-supported operations but are not mined from the current gold artifacts:

- `class_hierarchy_delta_supported = false`
- `exception_delta_supported = false`
- `unsupported_count = 0`

No selected T-box record is incorrectly forced into these operations. Unsupported-but-theoretically-valid operations are represented by the schema and prompt contract, while current gold uses only operations supported by available historical artifacts.

## Task Separation

This report covers T-box taxonomy-patch gold only. It does not change A-box gold, A-box prompts, or A-box scoring. Strict T-box `signature_after` reconstruction remains a separate diagnostic task and is not the taxonomy-patch gold target.
