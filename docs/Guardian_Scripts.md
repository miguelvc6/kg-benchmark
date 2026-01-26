# Guardian Scripts

This document describes the lightweight, offline scripts under `guardian/`.
They provide deterministic validation, state modeling, and patch application for
WikidataRepairEval's "Guardian" harness.

## Overview

- `guardian/patch_parser.py`
  - Purpose: Parse, validate, and normalize repair proposals against the schema.
  - Inputs: JSON string or dict proposal.
  - Output: Normalized dataclasses with a canonical SHA256 hash.
  - Key behaviors:
    - Enforces Q/P ID formats and normalizes casing.
    - Normalizes values (QIDs, ISO dates, numbers, strings).
    - Canonical JSON serialization for hashing.
  - CLI:
    - `python -m guardian.patch_parser --in proposal.json --out normalized.json`

- `guardian/state.py`
  - Purpose: Provide a minimal, deterministic in-memory entity state.
  - Core types:
    - `EntityState` for `qid` + `claims` (pid -> list of values).
    - `Value` for typed value normalization and sorting.
  - Key behaviors:
    - Deterministic ordering of values across types.
    - Constructors for world-state and case-record overrides.
    - Utility methods for set/add/remove/delete and diffing.
  - Notes:
    - Pure offline; no I/O within the module.

- `guardian/apply_patch.py`
  - Purpose: Apply normalized patches to an `EntityState`.
  - Output: New state, op-by-op trace, action summary, and diff.
  - Key behaviors:
    - Applies SET/ADD/REMOVE/DELETE_ALL in order.
    - Classifies actions as INSERT/UPDATE/DELETE/NOOP or MIXED.
    - Produces deterministic trace entries and diff output.
  - CLI:
    - `python -m guardian.apply_patch --case case.json --world world_state.json --proposal proposal.json`

## Typical Flow

1. Validate and normalize a proposal with `guardian.patch_parser`.
2. Build a pre-repair `EntityState` from world state and case record with
   `guardian.state`.
3. Apply the proposal with `guardian.apply_patch` and consume the trace, action
   summary, and diff for evaluation metrics.
