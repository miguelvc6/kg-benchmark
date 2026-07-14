# Benchmark Design

The dataset is reconstructed from Wikidata constraint-report candidates and historical entity/property revisions. Each
case records a pre-repair problem, the historically accepted repair, current persistence, and sanitized world-state
context. Classification and selection metadata remain hidden from model-visible prompts.

The fixed construct-review sample is selected by seed-13 SHA-256 ranking over case IDs and exposes no class, track,
subtype, or label columns. It is used only to discover possible implementation errors; exhaustive deterministic gates
and conservative dispositions, rather than the sampled reviewer, determine selection eligibility.

The final release contains the complete Stage 0–4 lineage, audit dispositions, independently eligible ordering, a fixed
few-shot support bank, the main 1,200 selection, and the nested Azure 600 selection. Only final disposition `include`
can enter evaluated populations. Each A-box QID/property group and each T-box property/revision group contributes at
most one evaluated case.

The support bank reserves 16 independent A-box and 16 independent T-box groups. Evaluation populations exclude the
entire bank. Initial prompts use four repair examples per locus and two diagnosis examples, while later prompt studies
may select longer deterministic prefixes without changing the dataset. Population quotas are configuration values; a
larger experiment extends the stable eligible prefixes and preserves every prior request identity.

Before the final 1,200 are sealed, the policy constructs a 1,440-case reserve (276 IC-L, 450 IC-G, 354 IC-E-elim, and
up to 360 T-box), renders every prompt, and replaces prompt-level failures using the predeclared stratum ordering. This
keeps prompt QA separate from relabeling and prevents quality fixes from changing the sampling rule after inspection.
