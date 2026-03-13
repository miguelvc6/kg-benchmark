# Technical Documentation

This area contains repository-facing documentation: how the benchmark is built, which scripts produce which artifacts, and how the current code organizes data and decisions.

- [Pipeline Implementation](./Pipeline_Implementation.md): current stage-by-stage workflow, scripts, CLI entry points, caches, and outputs.
- [Artifact Schemas](./Artifact_Schemas.md): structure of the Stage 2, Stage 3, Stage 4, and Stage 5 artifacts.
- [Classifier Specification](./Classifier_Specification.md): current classification logic implemented in `src/classifier.py`.
- [Proposal Validation](./Proposal_Validation.md): A-box and T-box proposal normalization contracts and runtime modules.
- [Evaluation Harness](./Evaluation_Harness.md): benchmark scoring workflow, outputs, and metric semantics.
- [Reasoning Floor](./Reasoning_Floor.md): zero-shot baseline runner, ablation bundles, and provider interface.
- [Track Diagnosis](./Track_Diagnosis.md): separate A-box vs T-box diagnostic task and scoring outputs.
- [T-BOX Update Analysis](./TBox_Update_Analysis.md): streaming frequency analysis for property-level T-box revisions in Stage 4.
- [Paper Execution Plan](./Paper_Execution_Plan.md): terminal runbook for building the benchmark and paper artifacts end to end.
- [Conceptual Deviation Report](./Conceptual_Deviation_Report.md): mismatches between current conceptual claims and the implemented repository.

Conceptual rationale and research framing live in [docs-conceptual](../docs-conceptual/README.md).
