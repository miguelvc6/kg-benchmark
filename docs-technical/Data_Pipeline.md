# Data Pipeline

`kg-benchmark acquire` mines report candidates, reconstructs historical repairs with fail-closed API access, computes
popularity, and extracts world state from a checksum-bound dump. All outputs, caches, logs, and resume state are scoped
to `work/`. A partial JSONL is never treated as complete; completion state is explicit and resumable.

`kg-benchmark build` classifies the acquired records and emits canonical JSONL. The release roles are popularity,
candidates, repairs, world state, and cases. SQLite may be generated as a disposable read index but is never canonical
or published alongside an equivalent JSON artifact.

The audit and selection gates verify source provenance, record identity, Stage 2–4 projection, disposition coverage,
support exclusions, and selection manifests before promotion. `kg-benchmark promote` then verifies that every canonical
role is present and nonempty, copies only those roles to a temporary sibling, writes hashes/counts into the dataset
manifest, re-verifies the release, and atomically renames it to `dataset/`.
