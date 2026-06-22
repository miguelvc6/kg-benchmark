# Step 12 Combined Decision

Final verdict: **RUN_ABOX_ONLY_FEW_SHOT_ABLATION**

- Step 12A A-box: **PASS**
- Step 12B T-box taxonomy-patch: **PASS_WITH_REVIEW**

A-box is the clean follow-on candidate: hard gates pass, local_graph accepted/exact-value gains exceed 3 pp, exact-value does not regress, and TypeC behavior does not materially worsen.
T-box should not move straight to a full few-shot run: primary family/schema/taxonomy metrics improve, but value-delta false-positive rate increases by more than 5 pp in both contexts, so it remains review-gated.

Reports:
- Step 12A: `reports/prompt_dev/few_shot/evaluation_static_v1_core_abox_canary/step12a_gate_report.md`
- Step 12B: `reports/prompt_dev/few_shot/evaluation_static_v1_core_tbox_canary/step12b_gate_report.md`
