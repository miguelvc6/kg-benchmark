# Post-freeze snapshots

This directory is the logical root for isolated confirmatory snapshots. Bulk
Stage 0--4 artifacts, snapshot-local caches, and dump files are intentionally
ignored by Git. Commit only small provenance manifests and summaries under the
repository's `release/` or `reports/` directories.

Each acquisition must use a new `data_post_freeze/<snapshot_id>/` namespace and
must not read from or write to the restored `data/` baseline. A dump may be a
symlink to capacity-managed local storage, but its immutable source identity,
byte size, upstream checksum, computed SHA-256, and resolved storage path must
be recorded in the snapshot acquisition summary.
