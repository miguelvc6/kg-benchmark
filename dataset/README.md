# Final paper dataset

This directory represents the one immutable paper dataset. Bulk JSONL files are distributed as one external release and
are ignored by Git. Git tracks this README, the verified release manifest and download metadata after publication.

Construction happens only under `work/`. `kg-benchmark promote` creates this directory atomically after lineage, audit,
support-bank, prompt-quality, and population gates pass. It validates every published record, requires exactly 1,200
main cases and a nested 600-case Azure subset, and reproduces the version-2 manifest byte-for-byte before promotion. It
refuses to overwrite an existing final dataset.

Use `kg-benchmark fetch` in a clean clone and `kg-benchmark verify --dataset-dir dataset` before running experiments.
