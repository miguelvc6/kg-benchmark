# Dataset Card and Limitations

WikidataRepairEval contains historically reconstructed Wikidata claim and schema repairs. Historical acceptance is the
evaluation target; it is not a claim of universal semantic truth. The release includes acquisition provenance, complete
audit dispositions, independent event-group ordering, a reserved support bank, and selection views.

The main population contains 1,200 cases: 230 IC-L, 375 IC-G, 295 IC-E-elim, and at most 300 T-box cases. Azure receives
a nested 600-case view. Larger per-stratum prefixes may be registered later without changing the initial populations.

IC-E-elim means supported logical and local extraction did not identify the target. It does not prove external evidence
is necessary. IC-U and malformed, unsupported, disagreement, prompt-rejected, or diagnostic cases are excluded from
headline scores. The paper supplies no retrieval, human ground-truth certification, semantic retries, or verifier-guided
proposal correction.

The dataset derives from Wikidata and Wikimedia pageview sources subject to their respective licenses and terms. The
repository code and release metadata identify the source dates, checksums, and applicable licensing information.
