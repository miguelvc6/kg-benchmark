# Correct Usage and Pitfalls

This is the shortest practical guide for using the benchmark safely.

## Recommended Usage

1. Treat Stage 4 classified benchmark records as the canonical benchmark cases.
2. Treat Stage 3 world state as contextual support, not as the canonical historical before-state of the edited target property.
3. Reconstruct the target property's pre-repair state from `repair_target.old_value`, with fallback to `violation_context.value`.
4. Keep current world-state context for surrounding graph and constraints only.
5. Evaluate model outputs against the historical repaired target using the official evaluator when possible.

## Common Pitfalls

### Pitfall 1: Reading Stage 3 `L1_ego_node.properties[target_pid]` as the historical before-state

Why it is wrong:

- Stage 3 stores frozen present-day context, not the canonical historical target-property state.

Correct behavior:

- use benchmark repair metadata to synthesize the target property's pre-repair state

### Pitfall 2: Building local-graph prompts directly from raw world state

Why it is wrong:

- raw world state may expose post-repair target-property values

Correct behavior:

- keep current non-target graph context
- replace the target property with benchmark-derived pre-repair state

### Pitfall 3: Evaluating against current world-state target values

Why it is wrong:

- the benchmark is historical
- current target-property values are not the evaluation authority

Correct behavior:

- compare against the historical repaired target defined in the benchmark record

### Pitfall 4: Collapsing benchmark and protocol layers together

Why it is wrong:

- the benchmark defines the task
- the reasoning floor and later protocol runners define experimental conditions over that task

Correct behavior:

- describe clearly whether you are using benchmark artifacts only or benchmark plus protocol/runtime artifacts

## Safe External Reproduction Rule

If you are not using the repository's official prompt-construction and evaluation
code, reproduce these details explicitly in your own release notes or methods
section:

- target-property pre-repair reconstruction rule
- prompt-side separation of target property from surrounding world-state context
- evaluation target definition
- handling of temporal-validity issues
