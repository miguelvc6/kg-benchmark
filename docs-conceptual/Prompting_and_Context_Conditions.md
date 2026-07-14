# Prompting and Context Conditions

The headline experiment crosses two tasks, two prompt regimes, and two context bundles. Repair proposal uses the known
repair locus; track diagnosis predicts A-box, T-box, or ambiguous independently and does not route proposals. The prompt
regimes are zero-shot and fixed static few-shot. The contexts are `logic_only` and `local_graph`.

Few-shot examples are selected from a published, audited support bank using deterministic prefixes. They use neutral
visible identifiers and task-valid outputs, and their event groups never appear in evaluated populations. Changing the
example count changes the prompt and defines a new request condition. Expanding only the evaluation population retains
the same request identity for already evaluated cases.

All prompts forbid hidden class/subtype labels, historical targets, post-repair target-property truth, selection labels,
and raw benchmark prefixes. No task uses tools, retrieval, semantic retries, or visible chain-of-thought.
