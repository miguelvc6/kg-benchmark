# Reasoning-Floor Viewer

Launch the read-only Streamlit interface through the single CLI:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark viewer --dataset-dir dataset --runs-dir runs
```

The viewer reads the canonical dataset, population manifests, raw generations, run configuration, and rescored results.
It must not edit dataset or run artifacts. Views should expose task, model, population, prompt regime, context bundle,
information condition, repair locus, parse state, metrics, cost/latency, and provenance. It must display whether a result
belongs to the frozen 1,200/600 populations or a registered extension.
