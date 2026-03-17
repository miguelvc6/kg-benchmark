# Benchmark Invariants

These are the benchmark rules external users should treat as non-negotiable.

## Canonical Data Layers

- The benchmark is built from real historical Wikidata repair events.
- The historical repair record is the canonical source for the edited target property.
- The frozen world-state snapshot is contextual support, not the canonical source for the target property's historical before-state.

## Target-Property Temporal Policy

- When reconstructing a case, derive the focus target property's pre-repair state from `repair_target.old_value` when available.
- If `repair_target.old_value` is unavailable, fall back to `violation_context.value`.
- Do not treat `L1_ego_node.properties[target_pid]` from Stage 3 world state as the historical before-state.
- Do not expose post-repair target-property values to prompting code when evaluating historical repair behavior.

## Current-World Context Policy

- Current/frozen world state is allowed for surrounding graph context, labels, and constraints.
- The target property is the exception: it must be reconstructed from benchmark repair metadata rather than copied directly from Stage 3.
- If you build local-graph prompts, keep non-target neighborhood and constraint context current, but synthesize the target property from the benchmark record.

## Evaluation Policy

- Evaluate against the historical repaired target, not against the current world state's target-property values.
- Do not silently update benchmark ground truth to match later graph evolution.
- If a case no longer represents the same logical problem in the frozen present-day setting, treat it as a temporal-validity issue rather than relabeling it ad hoc.

## Protocol Separation

- The benchmark release and the reasoning-floor/protocol release are related but distinct.
- The benchmark defines the task.
- Prompt builders, baselines, and protocol runners are experimental layers over that task.
- A protocol implementation that bypasses these invariants can produce invalid benchmark conclusions even if its code runs successfully.

## External-Use Rule

- If you do not use the repository's official loaders, prompt builders, and evaluator, you must replicate these invariants exactly.
- If you cannot replicate them exactly, report your use as a derived task or derived prompt setting rather than as direct benchmark use.
