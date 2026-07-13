# Step 13A Render Gate Report

Status: `PASS`

The full A-box static few-shot render completed without model inference.

## Checks

- Rendered prompts: `8408`.
- Context split: `{'local_graph': 4204, 'logic_only': 4204}`.
- Leakage hard hits: `0`.
- Raw `repair_`/`reform_` ID hard hits: `0`.
- Structured hidden metadata hard hits: `0`.
- Soft visible-text hits: `79`.
- Soft hits by term: `{'classification': 75, 'popularity': 4}`.
- Overlap scan: `PASS` with core case overlap `0` and core T-box revision overlap `0`.

## Soft Hit Policy

Soft hits are non-blocking lexical occurrences inside visible natural-language values, labels, descriptions, report text, or snippets. They are documented in `few_shot_leakage_scan.json` under `soft_visible_text_matches`.

## Verdict

`PASS`
