# Frozen prompt sources

This directory contains the model-visible prompt contracts used by the paper. Each file has `[system]` and `[user]`
sections. Few-shot examples are inserted immediately before `Input case`; their count and support-bank manifest are
part of the rendered prompt hash.

These files are the runtime source of truth. `guardian.prompts` loads them directly, and each run records their hashes.
The next methodology freeze will bind these exact bytes.
