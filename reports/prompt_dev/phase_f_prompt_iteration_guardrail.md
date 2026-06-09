# Phase F Prompt Iteration Guardrail

- `prompt_dev_v1`, `prompt_dev_v2`, and `prompt_dev_v3` are dev-only prompt iterations.
- No core results have been inspected for these prompt iterations.
- Do not do more broad prompt rewrites without a measured failure mechanism from the answerability audit or diagnostic tasks.
- Acceptable future changes are limited to abstention, diagnostics, evaluator/reporting fixes, or one targeted `prompt_dev_v4` based on the answerability audit.
- Phase G main should use oracle mode first. `diagnosis_routed` is an ablation only unless track diagnosis improves substantially.
- Current diagnostic source run: `reports/prompt_dev/evaluation_prompt_dev_v4_spec_only_holdout96_ollama_zero_shot`.
