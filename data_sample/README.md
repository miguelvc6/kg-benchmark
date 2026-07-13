# Tracked Smoke Fixture

This directory contains one synthetic A-box repair and its frozen world state. It exists only to exercise repository commands with `--sample`; it is not research data and must not be included in paper results.

Generated outputs in this directory remain ignored. Rebuild them with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m classifier --sample --no-progress --no-full-output
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m splitter --sample
```
