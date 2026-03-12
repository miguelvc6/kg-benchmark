# Scientific Vision: The Dynamics of Neuro-Symbolic Alignment

## Scope

WikidataRepairEval Phase 1 studies how large language models and knowledge graphs can participate in a stable repair loop rather than acting as isolated components.

- LLMs are flexible but prone to hallucination and knowledge staleness.
- Knowledge graphs are auditable but incomplete and brittle.
- A sustainable neuro-symbolic system needs a mediation layer that decides when model output is trustworthy enough to change structured knowledge.

This benchmark is the experimental apparatus for that claim.

## Repository Objective

This repository is the first minimum publishable unit of the broader research program.

Its purpose is to deliver the full Phase 1 benchmark foundation:

- benchmark data processing from historical repair discovery through experiment-ready artifacts
- evaluation procedures that can judge repair quality, validity, and alignment
- a "reasoning floor" baseline that measures what current LLMs can do before any Guardian protocol, tool use, memory, or learning intervention is introduced

The repository therefore does not exist only to build data. It exists to establish the benchmark, the evaluation frame, and the minimum baseline against which later Guardian-style systems can be compared.

## Phase 1 Completion Requirements

Phase 1 is conceptually complete only when the repository supports all three of the following:

1. End-to-end benchmark construction for the benchmark artifacts needed by downstream experiments.
2. Evaluation logic that can test success, failure, and safety under the benchmark's validity rules.
3. A reasoning-floor baseline that exposes the zero-shot performance ceiling of unaided models under the same benchmark conditions used in later phases.

These requirements define the minimum publishable unit because without all three, later claims about protocol gains would be underspecified:

- without the benchmark, there is no controlled task
- without the evaluation layer, there is no defensible measurement
- without the reasoning floor, there is no pre-intervention reference point

## The Guardian Hypothesis

The project assumes that a durable neuro-symbolic workflow requires a formal transaction protocol, described here as a "Guardian", between a stochastic model and a rigid knowledge graph.

Two hypotheses follow:

1. Symbolic rejection filters such as SHACL or OWL constraints are necessary to prevent semantic drift in generated knowledge.
2. Better semantic behavior comes from active grounding: verified repairs become the safe signal that can later be reused for training, evaluation, or feedback loops.

## Research Questions

### RQ1: Protocol Definition

What should count as a valid knowledge transaction?

The project treats a repair as more than a triple edit. A useful transaction must preserve proposal content, rationale, provenance, and uncertainty.

### RQ2: Information Gap

What information source is actually required to resolve a violation?

The benchmark distinguishes cases solvable from:

- internal logic alone
- local graph topology
- genuinely external information

This is the central motivation for the Type A / B / C taxonomy.

### RQ3: Verification Trade-off

How much human oversight can be replaced by automated constraint verification without damaging graph integrity?

The project therefore cares about acceptance quality, not just whether a model can produce a syntactically valid patch.

### RQ4: Loop Dynamics

If an LLM is trained on historically verified repairs, does it become more truthful, or does it simply overfit to benchmark-specific constraint logic?

### RQ5: The Reasoning Floor

What is the strongest performance current models can achieve under zero-shot conditions before any Guardian intervention is added?

This question matters because later protocol gains are only meaningful if they are measured against a clear pre-intervention baseline rather than against an undefined notion of model capability.

## Why This Benchmark Exists

Existing KG-repair resources tend to be fragmented, outdated, or focused only on whether a repair was produced. WikidataRepairEval is designed instead to answer what information was needed to produce a correct repair and whether that distinction can support more rigorous neuro-symbolic evaluation.

Key design commitments:

- Real historical Wikidata repair events rather than purely synthetic cases.
- Benchmark stratification by information necessity rather than by constraint family alone.
- Separate treatment of data repair and schema reform to make concept drift visible.
- Auditable popularity stratification so head and long-tail entities can be studied separately.

The conceptual taxonomy that supports these commitments is defined in [Benchmark Taxonomy](./Benchmark_Taxonomy.md).
