# Synthetic Release Fixture

This directory is a tracked one-case software fixture for the snapshot and release-manifest workflow. It contains no
paper data and must never be used for scientific results. `release_manifest.json` is a candidate manifest generated
from the tracked `data_sample/` inputs; it is intentionally not a confirmatory research release.

Verify it with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m artifact_release verify \
  --manifest release/sample-v0.1.0/release_manifest.json \
  --release-root release/sample-v0.1.0
```
