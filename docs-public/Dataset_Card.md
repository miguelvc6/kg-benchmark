# Dataset Card: WikidataRepairEval

## Release Status

`kg-benchmark` is a research repository for constructing and evaluating Wikidata repair cases. As of 2026-07-13,
the repository does **not** contain a complete, independently downloadable dataset release. The large data artifacts are
ignored by Git, their publication URLs and release checksums are unresolved, and the exact upstream source snapshot is
not bound to a tracked immutable manifest. Numbers below describe the current local build and tracked selection
manifests; they must not be cited as a released dataset version until a published distribution manifest replaces the
draft template in `release/artifact_distribution.template.json`.

## Summary

The benchmark reconstructs historical Wikidata repair events and pairs them with later, frozen graph context. It is
intended to test whether a system can choose an appropriate repair track and propose a historically aligned A-box value
repair or T-box constraint repair without leaking the target from the later snapshot.

Each classified case contains identifiers for the focus entity and property, constraint-report context, a historical
repair target, a reference to frozen world-state context, a repair track, an information-necessity class/subtype,
classifier confidence, and popularity metadata. The exact record contract is
`schemas/04_classified_benchmark.schema.json`.

## Current Artifact Inventory

| Layer | Intended role | Current release state |
| --- | --- | --- |
| Stage 2 | Historical repair record and target metadata | Required, not published |
| Stage 3 | Frozen world-state context | Required, not published |
| Stage 4 | Classified benchmark JSONL | Local build exists, not published |
| Stage 5 | Deterministic train/dev/test split metadata | Canonical release artifact not yet published |
| Core/dev selection manifests | Deterministic evaluation views over Stage 4 | Tracked in `reports/benchmark_selection/` |

The repository includes small synthetic files under `data_sample/` for software tests. They are not research examples
and must not be included in benchmark results.

## Snapshot and Provenance

The current Stage 4 build report records build time `2026-06-01T08:19:50Z` and names local Stage 2, Stage 3, and
popularity inputs. This is build metadata, not an upstream Wikidata snapshot identifier. The exact Wikidata endpoint or
dump, query/request definition, collection interval, response hashes, and Stage 2/3 checksums are not all preserved in a
tracked source manifest. The later world state is described in the project as a 2026 snapshot, but its exact snapshot
timestamp is unresolved.

Publication must bind all of the following in both the immutable release manifest and the distribution manifest:

- source commit and clean-worktree state
- exact upstream source URL or dump identifier and collection timestamp
- query/request definition and any retry, truncation, or persistence-failure accounting
- byte size and SHA-256 digest of every released file
- immutable download URL and per-artifact license

The draft distribution template leaves unknown values as `null` or empty lists. Those values are explicit release
blockers, not permission to substitute the current live Wikidata state.

## Current Local-Build Statistics

The local `reports/classifier_stats.json` reports 535,570 Stage 4 records:

| Dimension | Count | Share |
| --- | ---: | ---: |
| A-box track | 78,976 | 14.7% |
| T-box track | 456,594 | 85.3% |
| Type A | 48,085 | 9.0% |
| Type B | 6,059 | 1.1% |
| Type C | 24,832 | 4.6% |
| T-box class | 456,594 | 85.3% |
| High confidence | 72,245 | 13.5% |
| Medium confidence | 169,109 | 31.6% |
| Low confidence | 294,216 | 54.9% |

These distributions show that the full artifact is dominated by T-box and low-confidence cases. Full-artifact aggregate
scores would therefore be poor summaries of A-box repair capability or high-confidence performance.

### Deterministic evaluation views

The tracked development manifest `dev_prompt_v1_seed_13.json` selects 600 cases from 535,570 scanned records: 481
main-score and 119 diagnostic cases, with 360 A-box and 240 T-box cases. It was generated with seed 13 at
`2026-06-01T09:27:19.089812Z`.

The tracked core manifest `core_v1_seed_13.json` selects 4,800 cases after excluding the dev manifest: 3,818 main-score
and 982 diagnostic cases. It was generated with seed 13 at `2026-06-01T09:30:21.645758Z`.

| Core dimension | Count |
| --- | ---: |
| A-box / T-box | 4,204 / 596 |
| Type A / Type B / Type C / T-box | 2,228 / 924 / 1,052 / 596 |
| High / medium / low confidence | 1,589 / 2,229 / 982 |
| Popularity head / mid / tail | 1,485 / 2,092 / 1,223 |

Selection is deterministic and group-capped, but it is quota-driven rather than population-representative. Some strata
are severely supply-limited: the current core contains 33 T-box relaxation cases and no T-box restriction cases. The
core also contains 300 coincidental schema-change cases and 263 schema-update cases. Report per-stratum sample sizes and
do not interpret the core distribution as an estimate of Wikidata repair prevalence.

The selection funnel before Stage 4 is not release-grade. Local ignored fetch logs contain collection diagnostics, but
they are not checksum-bound to the current Stage 4 file and do not reconcile cleanly enough to serve as canonical counts.
The published release must include a single machine-readable funnel covering discovery, deduplication, successful
historical reconstruction, persistence filtering, classification, diagnostics exclusion, dev exclusion, and final
selection.

## Labels and Evaluation Units

- **A-box:** instance-level value repair.
- **T-box:** property-constraint or schema repair.
- **Type A:** rule or target-derived repair under the repository taxonomy.
- **Type B:** repair derivable from bounded local context.
- **Type C:** `EXTERNAL_BY_ELIMINATION` or a diagnostic unknown subtype. It is not proof that an external source was
  consulted by the historical editor.
- **Main-score cases:** cases admitted by the selection policy for headline scoring.
- **Diagnostic cases:** low-confidence, ambiguous, incomplete-context, or otherwise analysis-only cases.

Users must preserve case/group isolation and the manifest's main-versus-diagnostic partition. The core and development
manifests are evaluation views, not replacements for Stage 4.

## Language, Editors, Bots, and Privacy

The current artifacts privilege English labels and descriptions (`*_label_en`, `*_description_en`, and an English label
cache). This creates language and entity-coverage bias. Missing English text is not evidence that an entity lacks labels
in Wikidata, and multilingual performance is not measured by the current benchmark.

Historical repair records may retain the public Wikidata revision author field. Usernames can identify people and edits
by unregistered users may expose IP-address-like identifiers in upstream history. The current schema does not provide a
reliable bot flag, editor-demographic attributes, consent metadata, or a documented de-identification transform.
Consequently:

- do not infer editor identity, demographics, expertise, or intent
- do not use author strings as model features or publish author-level rankings
- aggregate or remove author identifiers from public derivatives unless they are scientifically necessary and ethically
  reviewed
- do not assume automated and human edits have been separated

The benchmark is derived from public collaborative data, but public availability does not eliminate privacy and research
ethics obligations.

## Missingness and Construction Bias

Missing labels, descriptions, neighbors, constraint metadata, historical snapshots, and persistence checks can reflect
API/network failures or construction limits rather than properties of the underlying graph. The collection pipeline has
bounded revision windows, retry behavior, snapshot-fetch failures, and persistence filtering. Cases that cannot be
reconstructed are not missing at random, so the released benchmark cannot directly estimate repair prevalence or editor
behavior across all Wikidata.

Additional known biases include:

- constraint-report coverage and report formatting determine candidate discovery
- historical survivorship and persistence checks favor repairs visible in the later snapshot
- property/revision repetition can make nominal case counts overstate independent evidence
- popularity stratification depends on pageviews, degree, and sitelinks and may favor well-connected entities
- current context is later than the historical target and may encode post-repair information
- selection quotas deliberately reshape the full population and several quotas are underfilled

Subgroup reporting should at minimum separate track, class/subtype, confidence, constraint family, popularity bucket, and
main versus diagnostic status. Where sample size permits, add property-level and revision-group sensitivity analyses.

## Contamination and Temporal Leakage

The frozen world state is later than the historical repair event. Target-property values, labels, neighboring claims, or
constraint state can therefore reveal the historical outcome. Use the official prompt/context construction and temporal
audit tooling, and reconstruct the target property's pre-repair state from the historical benchmark record. Do not pass
the raw later target-property value to a model.

Development prompt work has already used the tracked dev tier. Existing exploratory core outputs are not an untouched
confirmatory test. A paper claim requires a separately frozen, untouched test selection and protocol established before
test execution. Model pretraining contamination by public Wikidata content and revision history cannot be ruled out;
report model/version dates and treat results as task performance, not evidence of novel factual acquisition.

## Intended Uses

- compare KG repair systems on a fixed, manifest-bound release
- study A-box versus T-box behavior and information-necessity strata
- evaluate abstention, parsing, and repair validity alongside exact outcomes
- audit errors against historical public revisions and bounded graph context

## Out-of-Scope and Prohibited Interpretations

- estimating global Wikidata error or repair rates
- measuring individual editor quality or behavior
- treating Type C as verified external evidence
- treating diagnostic cases as ordinary main-score examples
- claiming multilingual generalization from English-centric fields
- rebuilding against live Wikidata and calling it byte-for-byte reproduction
- reporting results from unresolved or checksum-mismatched artifacts as official benchmark results

## Licensing

Repository source code is licensed under the root `LICENSE`. Wikidata content is made available upstream under CC0 1.0,
but that does not by itself settle the license for every project-authored annotation, selection, report, or packaged
derivative. `DATA_LICENSE.md` records the current boundary. Each published distribution entry must declare its own
license before redistribution; `null` means unresolved, not unrestricted.

## Maintenance, Versioning, and Corrections

Dataset versions must be immutable and identified by a source commit, release-manifest hash, distribution-manifest hash,
and per-file hashes. A live rebuild from a newer Wikidata snapshot is a new dataset version even when the code is
unchanged. Classifier, taxonomy, selection-policy, evaluator, or schema changes also require a new version and changelog
entry. Never replace files behind an existing version or URL.

Corrections should publish a new version, document affected case IDs and metrics, and retain the superseded manifest for
auditability. The project does not yet declare a long-term maintainer, support window, archival DOI, or vulnerability/
data-removal contact; these are release blockers for a durable public artifact.

## Reproducibility and Citation

Use `kg-artifacts` with a published distribution manifest to fetch and verify a clean clone. The checked-in template is
expected to fail closed because no release URLs or checksums have been declared. See
[Artifact Acquisition](../docs-technical/Artifact_Acquisition.md).

`CITATION.cff` provides repository-level citation metadata but intentionally contains no DOI or paper authorship claim.
For a paper, report the exact dataset version, source commit, release and distribution manifest hashes, selection
manifest hash, evaluator version, and model identifier.

## Further Reading

- [Benchmark Invariants](./Benchmark_Invariants.md)
- [Correct Usage and Pitfalls](./Correct_Usage_and_Pitfalls.md)
- [Release Structure](./Release_Structure.md)
- [Artifact Acquisition](../docs-technical/Artifact_Acquisition.md)
