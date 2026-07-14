# Prompt and Task Configuration

The model-visible sources are under `paper/prompts/`; their hashes are bound by the protocol. The A-box task returns a
claim patch, the T-box task returns a taxonomy patch, and diagnosis returns A-box, T-box, or ambiguous. Repair uses oracle
routing; diagnosis never routes proposals in the headline experiment.

Zero-shot and static few-shot are headline regimes. The default demonstration counts are four A-box, four T-box, and two
diagnosis examples. Counts are configuration fields and select deterministic support-bank prefixes. Rendering fails on
capacity, role-coverage, overlap, leakage, schema, or context-length violations.

IC-E-elim is evaluated with `logic_only` and `local_graph`, without external retrieval. It shares the A-box prompt and
response schema; its hidden condition label is used only for audit, selection, and stratified reporting.
