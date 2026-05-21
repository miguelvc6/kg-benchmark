# Conceptual Documentation

This area contains the benchmark's research-facing design: why the project exists, what it measures, and which methodological decisions define WikidataRepairEval.

Start with the two anchor documents:

- [Full Project Description](./00-full_project_description.md): complete conceptual description of WikidataRepairEval 1.0, including pipeline, taxonomy, classifier risks, dataset tiers, prompting, evaluation, and paper scope.
- [KG LLM Benchmark Narrative](./00-kg_llm_benchmark.md): paper-facing research narrative, hypotheses, tasks, context ablations, evaluation protocol, limitations, and positioning.

The remaining conceptual docs are shorter navigational slices that should agree with those anchors:

- [Scientific Vision](./Scientific_Vision.md): project thesis, research questions, and first-paper scope.
- [Benchmark Taxonomy](./Benchmark_Taxonomy.md): repair-locus and information-access labels.
- [Evaluation Framework](./Evaluation_Framework.md): evaluation goals, reasoning-floor design, and metric families.
- [Temporal Validity](./Temporal_Validity.md): leakage risks, persistence checks, and the target-property rule.

If a change is driven by repository implementation, CLI behavior, artifact layout, or engineering procedure, it belongs in [docs-technical](../docs-technical/README.md), not here.
