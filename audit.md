# Restored-Baseline Audit Report

## Executive result

The restored baseline has complete, content-validated Stage 0--4 lineage and a complete final disposition for every one
of its 535,570 cases. The baseline is not the confirmatory paper population: it predates the methodology freeze and its
384 legacy rendered prompts contain deterministic temporal-leakage failures. It remains remediation evidence and an
exclusion source for the isolated post-freeze acquisition.

The final lineage manifest is `reports/lineage/restored_v3.json`. It binds the authoritative compiled Stage 2 JSON,
the recovered JSONL precursor, Stages 0/1/3/4, the declared historical transform, and the exact Git revision. The final
audit is `reports/automated_audit/full_v3/`.

## Lineage result

The two Stage 2 files are different lineage roles, not equivalent serializations:

| Artifact role | Records | Result |
| --- | ---: | --- |
| Recovered JSONL precursor | 544,178 | 544,177 physical lines; one historical multi-value line is declared |
| Authoritative compiled JSON | 535,570 | Used to construct Stages 3 and 4 |
| Shared ordered records | 535,570 | Canonically equal after removing only compiled `popularity` |
| Precursor-only records | 8,608 | Retained as historical precursor records; aggregate ID hash recorded |

Every authoritative ID is unique and occurs in precursor order. All 535,570 compiled rows contain `popularity`; none of
the precursor rows do. Stage 0 contains every authoritative QID and every attached popularity payload is canonically
equal. Every authoritative event has Stage 1 candidate provenance. Stages 2, 3, and 4 have exactly equal ID sets and
counts; Stage 2 and Stage 4 also have equal order and equal Stage 2 projection fields. All source artifacts were rehashed
when the strengthened popularity-payload check reused the already-passing relationship and identity subchecks.

Exact upstream snapshot URLs and retrieval timestamps for these recovered historical files remain unavailable and are
explicitly marked unresolved. No provenance value was inferred. This is why a separately acquired post-freeze snapshot
is still required for confirmatory evaluation.

## Exhaustive deterministic audit

The final deterministic run covered all 535,570 Stage 4 rows, all 535,570 world-state entries, authoritative Stage 2,
and 384 legacy prompt rows across 96 cases. Lineage, Stage 4 integrity/schema, joins, and render-count coverage passed.
There were no malformed rows, duplicate IDs, missing IDs, missing world-state entries, or Stage 2/3/4 projection errors.

| Deterministic status | Cases | Release meaning |
| --- | ---: | --- |
| `pass` | 472,790 | No implemented case-level disagreement or unsupported condition |
| `unsupported` | 56,761 | Diagnostic only; absence of support is not agreement |
| `disagreement` | 6,019 | Diagnostic only; stored classification and deterministic replay disagree |
| integrity `error` | 0 | No integrity exclusion |

The principal finding counts are 33,525 T-box causality-policy replays, 16,741 unsupported Type A subtypes, 5,402
unknown Type C diagnostics, 4,993 semantic-review local-evidence cases, 4,689 cardinality conflicts, 596 Type C cases
with visible local support, 350 Type A replay failures, 325 format contradictions, and 109 unreproduced Type B support
cases. Finding categories overlap and are not a population partition.

## Prompt and Codex error discovery

The hardened temporal scanner tested 3,824 hidden claims. It found 314 high-risk occurrences across 16 cases in the
legacy prompts, including current T-box target values, URL/query-embedded identifiers, replacement labels or descriptions
exposed under other QIDs, and version-suffix aliases. The prompt-level automated gate therefore correctly fails. These
cases require rerendering and are not selection eligible.

A fixed, label-hidden 450-case construct sample and 50-case temporal sample were reviewed with Codex CLI 0.144.3 and
`gpt-5.6-sol`. All 50 batches completed. Construct verdicts were 231 pass, 197 concern, and 22 uncertain. Temporal verdicts
were 43 pass and seven suspected-leakage reviews mapping to five cases. The review packets were byte-identical between
the pre-lineage and final audit runs; `review_reuse.json` binds both packet hashes, the clean review run, and zero repeated
provider queries.

The Codex mechanisms were converted into deterministic rules before the final rerun. Sixteen cases are now caught
deterministically. One additional format-normalization case remains conservatively pending rerender from the AI review;
its stripped identifier is mechanically supported by the classifier's visible format rule, but the review packet judged
that visibility insufficient. This disagreement is retained rather than adjudicated by changing a label.

## Final dispositions

The final disposition file has exactly one row for every case:

| Disposition | Cases | Selection eligibility |
| --- | ---: | --- |
| `include` | 472,605 | Eligible only as a baseline audit disposition; not a confirmatory selection |
| `diagnostic` | 62,948 | Ineligible |
| `exclude_pending_rerender` | 17 | Ineligible |
| integrity `exclude` | 0 | Ineligible |

Only `include` is admitted by the selector. Diagnostic, unsupported, disagreement, malformed, and pending-rerender
cases remain outside main-score evaluation. The disposition file SHA-256 is
`5f8d7ad44d952a4887cf311b709beec384756cd9faa54feca651e0e1b741df6c`.

## Historical `full_v1`

`reports/automated_audit/full_v1/` is retained as historical evidence. It covered the same 535,570 Stage 4/3 cases and
384 prompts before Stage 2 was available to the audit. It reported 473,412 pass, 60,537 unsupported, 1,621 disagreement,
and zero integrity errors, then finalized 473,222 include, 62,344 diagnostic, and four pending-rerender cases. Its Codex
sample produced 221 construct pass, 193 concern, 36 uncertain, 39 temporal pass, five suspected leakage, and six temporal
uncertain verdicts. It must not be substituted for the v3 lineage-bound result.

## Methodology and release boundary

`protocols/methodology_v1.json` is the clean, pre-acquisition freeze. It binds prompts, schemas, model identifiers and
inference settings, classifier/audit/selection/evaluation code, acquisition and selection policies, and the analysis
plan without depending on future selections or deployment digests. It was invalidated and regenerated when the temporal
audit defect was fixed.

After isolated acquisition, a dataset/allocation binding will authorize private selection. The later execution freeze
must bind the final 1,200/600 selection hashes and immutable model/deployment revisions. No model inference may begin
until that execution gate passes.

## Residual limitations

- Automated consistency checks and Codex error discovery do not establish human construct validity, semantic truth,
  causal necessity, or uniqueness of repair.
- The restored baseline has unresolved exact upstream retrieval provenance and is development-contaminated.
- The legacy prompt set fails the temporal gate and must never be used as the confirmatory selection's rendered prompts.
- Unsupported and disagreement dispositions are reported as audit findings, not relabeled or treated as ground truth.
- Confirmatory claims require a new post-freeze snapshot, exhaustive audit, reserve prompt scan, exactly 1,200 independent
  final cases, and the nested 600-case API calibration subset.
