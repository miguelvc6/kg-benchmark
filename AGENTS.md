# Documentation Guidance

This repository keeps documentation in two separate areas:

- `docs-conceptual/` for research framing, benchmark intent, hypotheses, evaluation goals, and other decision-level material.
- `docs-technical/` for repository structure, implementation details, scripts, artifacts, operational procedures, and developer-facing usage notes.

Use this split consistently:

- Edit `docs-conceptual/` only when the underlying research decisions have changed.
- Edit `docs-technical/` whenever the repository, code paths, CLI behavior, artifacts, or engineering choices change.
- Avoid mixed documents when possible. If a topic needs both perspectives, keep the research rationale in `docs-conceptual/` and the implementation detail in `docs-technical/`.
- Preserve navigability. When files move or split, update links and index pages in the same change.
- Do not duplicate the same content across both areas. Cross-link instead.

`docs/README.md` exists only as a pointer into the two primary documentation areas and should not become a third place for full documentation.

## Python Command Convention

This environment does not guarantee a `python` executable on `PATH`; `python3` is available, and project commands are already managed through `uv`.

- Prefer `uv run python ...` for repository scripts, tests, and other project-scoped commands.
- Use bare `python3` only for ad hoc one-off commands that are intentionally outside the project's `uv` environment.
- Do not assume `python` resolves successfully in this repository.
