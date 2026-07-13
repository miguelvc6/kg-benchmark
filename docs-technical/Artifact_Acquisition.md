# Artifact Acquisition

This document defines how a clean clone obtains large benchmark files. It complements `artifact_release.py`: the release
manifest validates scientific consistency and hashes files inside a completed release, while the distribution manifest
maps those immutable files to download locations and records upstream source provenance.

## Current Status

`release/artifact_distribution.template.json` is a reproducibility template, not a release. Its required artifacts are
marked `unresolved`, with empty URL lists and null checksums/byte counts. This is intentional. The repository currently
has no defensible way to reconstruct exact public URLs or source snapshot identifiers, and the CLI fails closed unless
the caller explicitly requests a status-only unresolved check.

## Clean-Clone Workflow

Install the project in the WSL environment and inspect the template:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv sync --extra dev
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-artifacts \
  --manifest release/artifact_distribution.template.json status --allow-unresolved
```

Once maintainers publish a versioned distribution manifest, place it outside the target data paths or obtain it from the
release page, then run:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-artifacts \
  --manifest release/artifact_distribution.v1.json --root . fetch
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-artifacts \
  --manifest release/artifact_distribution.v1.json --root . verify
```

Use `--artifact ARTIFACT_ID` repeatedly to select files. `fetch` accepts only HTTPS and local `file:` URLs, writes to a
temporary file in the destination directory, verifies byte count and SHA-256, and only then atomically installs it.
Existing valid files are reused; an invalid existing file requires an explicit `--overwrite`. Paths that are absolute or
escape the acquisition root are rejected.

`--allow-unresolved` changes unresolved required entries from errors to skips. It is useful for inspecting a draft; it
must not be used to claim that a release is complete.

## Manifest Contract

The JSON Schema is `schemas/artifact_distribution.schema.json`. The runtime validator also enforces uniqueness and
publication-state rules that are awkward to express portably in JSON Schema.

Top-level fields:

- `status`: `draft` or `published`
- `release`: benchmark version, repository, source commit, and URL/hash of the immutable scientific release manifest
- `source_snapshots`: exact upstream data provenance or an explicit unresolved declaration
- `artifacts`: destination path, role, status, mirrors, SHA-256, byte count, media type, license, and source links

Artifact states:

- `bundled`: shipped with the repository/distribution and still checksum-bound
- `published`: remotely retrievable with at least one URL, exact SHA-256, and byte count
- `unresolved`: no URL, checksum, or byte count is known; guessed values are forbidden

A top-level `published` manifest must bind a release-manifest URL and checksum and cannot contain unresolved required
artifacts. Every artifact license must be resolved as part of publication review even though the draft schema permits
`null` so omissions remain machine-visible.

## Producing a Published Manifest

1. Generate the complete Stage 2/3/4/5 release from a named source snapshot.
2. Run `kg-artifact-release build` and retain its validation output.
3. Upload immutable files without changing their names or contents after publication.
4. Copy exact byte counts and SHA-256 values from the release inventory; never hand-transcribe guessed values.
5. Record all mirrors, per-artifact licenses, upstream snapshot URLs/times, source commit, and release-manifest URL/hash.
6. Set the distribution status to `published` and validate it with the schema plus `kg-artifacts verify` in a clean clone.
7. Archive both manifests and cite their hashes in the paper and changelog.

The acquisition manifest is not a build recipe. Re-running live APIs creates a new dataset version and cannot substitute
for downloading the archived, checksum-bound paper artifacts.
