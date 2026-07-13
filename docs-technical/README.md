# Technical Documentation

This area contains repository-facing documentation: executable research controls, implementation details, artifact
contracts, and developer operations. Research rationale and paper claims live in
[docs-conceptual](../docs-conceptual/README.md).

## Current Protocol

- [Research Release Protocol](./Research_Release_Protocol.md): authoritative engineering protocol for building,
  freezing, registering, verifying, and releasing paper-eligible artifacts.
- [Confirmatory Release Runbook](./Confirmatory_Release_Runbook.md): Stage 0--4 lineage, isolated post-freeze acquisition,
  v2 audits, reserve/final selection, API subset construction, and failure recovery.
- [Automated Consistency Audit](./Automated_Consistency_Audit.md): deterministic and Codex-assisted error
  discovery workflow for full-data and rendered-prompt artifacts.
- [Pipeline Implementation](./Pipeline_Implementation.md): current stage-by-stage code paths and outputs.
- [Artifact Schemas](./Artifact_Schemas.md): current artifact contracts and schema status.
- [Artifact Acquisition](./Artifact_Acquisition.md): clean-clone retrieval and verification of published large files.
- [Benchmark Selection](./Benchmark_Selection.md): deterministic dev/core selection and leakage controls.
- [Evaluation Harness](./Evaluation_Harness.md): scoring workflow, output subsets, and metric semantics.
- [Reasoning Floor](./Reasoning_Floor.md): model runner, prompt/task selection, provenance capture, and resume behavior.
- [Model Execution Matrix](./Model_Execution.md): registered Ollama/Azure conditions, content-addressed generation
  reuse, extensible population planning, and versioned metric replay.

The [Paper Execution Plan](./Paper_Execution_Plan.md) is a historical pre-release runbook. It is retained to explain
legacy exploratory artifacts and must not be used to produce confirmatory or release-candidate results.

## Task And Experiment Documentation

- [Classifier Specification](./Classifier_Specification.md): Stage 4 classification logic.
- [Proposal Validation](./Proposal_Validation.md): A-box, strict T-box, taxonomy-patch T-box, and track-diagnosis
  normalization contracts.
- [Non-LLM Baselines](./Non_LLM_Baselines.md): deterministic track, symbolic TypeA, local lookup, and evaluator
  lower-bound baselines.
- [Prompt Development](./Prompt_Development.md): dev-only prompt rendering, evaluation, and freeze artifacts.
- [Few-Shot Evaluation](./Few_Shot_Evaluation.md): support-set and leakage governance for Phase G5 ablations.
- [Few-Shot Implementation Plan](./few_shot_evaluation_implementation_plan.md): implementation and execution gates for
  the few-shot workstream.
- [Track Diagnosis](./Track_Diagnosis.md): separate A-box versus T-box diagnosis and optional routed generation.
- [T-Box Taxonomy Patch Task](./TBox_Taxonomy_Patch_Task.md): taxonomy-patch answer contract and extraction boundaries.
- [T-Box Taxonomy Patch Governance](./TBox_Taxonomy_Patch_Governance.md): task versions, artifact separation, and score
  interpretation.
- [T-Box Taxonomy Patch Implementation Plan](./tbox_taxonomy_patch_implementation_plan.md): historical migration plan;
  use the task and governance documents for current behavior.
- [T-Box Update Analysis](./TBox_Update_Analysis.md): property-revision frequency analysis.

## Operations And Tools

- [LLM Endpoint Smoke Test](./LLM_Endpoint_Smoke_Test.md): provider connectivity checks.
- [Ollama VM Runbook](./Ollama_VM_Runbook.md): portable local-Ollama VM setup and Phase F/G commands.
- [Reasoning Floor Viewer](./Reasoning_Floor_Viewer.md): read-only Streamlit debugger for run artifacts.
- [Confidence Frequency Report](./Confidence_Frequency_Report.md): classification-confidence frequency report.

## Plans And Completion History

- [Implementation Plan](./00-implementation_plan.md): phased project plan. Its task lists are historical and are not the
  current release gate.
- [Phase A Completion](./implementation-plan-completions/00-phase_A_completion.md)
- [Phase B Completion](./implementation-plan-completions/00-phase_B_completion.md)
- [Phase C Completion](./implementation-plan-completions/00-phase_C_completion.md)
- [Phase D Manual Audit](./implementation-plan-completions/manual_audit_phase_D.md)
- [Phase E Completion](./implementation-plan-completions/00-phase_E_completion.md)
- [Conceptual Deviation Report](./Conceptual_Deviation_Report.md): current implemented boundary and remaining
  post-release research work.
