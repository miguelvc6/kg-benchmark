# Manual Audit Plan for Phase D

**Project:** WikidataRepairEval 1.0  
**Audit policy version:** `phase_d_v1`  
**Depends on:** Phase B classifier redesign and Phase C selection policy  
**Purpose:** Validate the highest-risk classifier decisions before using core labels as paper evidence.

## 1. Audit objective

The audit is not a generic data-cleaning pass. It answers whether the benchmark labels can support the paper's claims about information access and repair locus.

The most important questions are:

1. Are `TypeC / EXTERNAL_BY_ELIMINATION` cases really not locally supported, or did the extractor miss evidence?
2. Are `TypeA / DELETE_AMBIGUOUS` cases correctly separated from clean rule-implied repairs?
3. Are `TypeB / LOCAL_*` cases truly locally grounded rather than accidental literal or label matches?
4. Are `T_BOX / SCHEMA_UPDATE` and `T_BOX / COINCIDENTAL_SCHEMA_CHANGE` cases causally plausible enough for main or diagnostic reporting?
5. Which selected cases should be main score, diagnostic/challenge, or excluded?

## 2. Audit sample

Target size: **450 cases**.

| Audit stratum | Target count | Purpose |
|---|---:|---|
| `TypeC / EXTERNAL_BY_ELIMINATION`, QID truth | 30 | Test whether QID target is absent from local context and needs non-local evidence or memory. |
| `TypeC / EXTERNAL_BY_ELIMINATION`, literal/date/numeric truth | 30 | Test whether literal matching missed local evidence. |
| `TypeC / UNKNOWN_*` diagnostics, including bad-target/context, focus-QID domain reasoning, and multiplicity artifacts | 45 | Validate unknown/fallback behavior. |
| `TypeA / FORMAT_NORMALIZATION` and `FORMAT_VALUE_PRUNING` | 50 | Check deterministic format normalization and pruning. |
| `TypeA / REJECTION_FORMAT_INVALID`, `SELF_LINK_REJECTION`, `TARGET_REQUIRED_CLAIM`, and `MULTIPLICITY_NORMALIZATION` | 55 | Check rule/format claims are not overclaimed. |
| `TypeA / SET_MEMBERSHIP_REJECTION` | 25 | Check whether the removed value is directly ruled out by a set-membership constraint or whether selection needs evidence. |
| `TypeA / DELETE_AMBIGUOUS` | 25 | Check delete ambiguity and whether local/external evidence is needed to choose deletion. |
| `TypeB / LOCAL_TEXT_CONFIRMED`, `LOCAL_TEXT_DERIVED`, `LOCAL_SELECTION_CONFIRMED`, and rare `LOCAL_FOCUS_QID` | 85 | Check independent local grounding, deterministic local derivation, and leakage controls. |
| `T_BOX / SCHEMA_UPDATE` | 25 | Check schema-change causal plausibility. |
| `T_BOX / COINCIDENTAL_SCHEMA_CHANGE` | 25 | Estimate low-causality diagnostic precision. |
| `T_BOX / RELAXATION_SET_EXPANSION` or `RESTRICTION_SET_CONTRACTION` | 25 | Check directional T-box reform labels, selected violation mapping, type-compatible overlap, and polarity. |
| `T_BOX / UNKNOWN_TBOX_CAUSALITY` | 30 | Check cases where changed constraints do not establish a causal link. |
| **Total** | **450** |  |

The audit sample should be disjoint from dev if dev is used for prompt tuning. It may overlap core because its purpose is to validate core labels. If overlap is used, the manifest must record it.

T-box case cards show the selected violation candidate, candidate mapping preview, mapped-report constraint, changed target constraint, compatible and incompatible overlap fields, semantic vs ignored qualifier changes, precise directional subtype, and a compact diff summary. Metadata/status qualifiers such as `P2316` may show that a constraint changed but should not be treated as semantic polarity evidence. In lean Stage 4, full T-box signatures are intentionally pruned, so annotators should prefer the compact diff summary and causality block over the older full-signature diff section.

## 3. Annotation fields

The audit CSV/JSONL must contain the following columns.

### Case identity and selection metadata

| Field | Type | Required | Description |
|---|---|---|---|
| `case_id` | string | yes | Stage 4 id. |
| `qid` | string/null | yes | Focus entity id. |
| `property` | string/null | yes | Target property id. |
| `track` | string | yes | Historical repair locus. |
| `class` | string | yes | Classifier class. |
| `subtype` | string | yes | Classifier subtype. |
| `confidence` | string | yes | Classifier confidence. |
| `selection_stratum` | string | yes | Audit stratum. |
| `analysis_slice` | string | yes | Core analysis slice if applicable. |
| `main_score` | boolean | yes | Whether the current policy treats this as main-score eligible. |
| `diagnostic_only` | boolean | yes | Whether the current policy treats this as diagnostic-only. |
| `popularity_bucket` | string | yes | `head`, `mid`, `tail`, or `unknown`. |
| `constraint_family` | string/null | yes | Constraint-family id or label if available. |
| `decision_constraint_type_qid` | string/null | yes | Constraint type selected by the classifier rule, if any. |
| `decision_constraint_type_label` | string/null | yes | Human-readable label for the decision constraint. |
| `decision_constraint_source` | string/null | yes | Why/how the decision constraint was selected. |
| `classification_rule_family` | string/null | yes | Classifier rule family such as `format`, `local_evidence`, or `tbox_schema_causality`. |
| `classification_rule_subfamily` | string/null | yes | More specific rule subfamily. |

### Classifier diagnostics shown to annotator

| Field | Type | Required | Description |
|---|---|---|---|
| `truth_source` | string/null | yes | Source of target truth tokens. |
| `truth_token_kind` | string/null | yes | `qid`, `literal`, `date`, `numeric`, `mixed`, or `none_expected`. |
| `truth_tokens_preview` | string | yes | Short, non-lossy preview of tokens. |
| `decision_branch` | string/null | yes | Classifier branch. |
| `local_match_kind` | string/null | no | Match kind if TypeB. |
| `local_match_source` | string/null | no | Local source bucket if TypeB. |
| `tbox_revision_key` | string/null | no | Property-revision group for T-box. |
| `group_key` | string | yes | Split/leakage group key. |

### Human annotation fields

| Field | Allowed values |
|---|---|
| `repair_locus_correct` | `yes`, `no`, `unclear` |
| `historical_target_well_defined` | `yes`, `no`, `unclear` |
| `target_visible_locally` | `yes`, `no`, `partial`, `unclear` |
| `extractor_missed_local_evidence` | `yes`, `no`, `unclear`, `not_applicable` |
| `external_evidence_required` | `yes`, `no`, `maybe`, `unclear`, `not_applicable` |
| `typec_judgment` | `external_confirmed`, `external_by_elimination_ok`, `local_missed`, `unknown_or_incomplete`, `bad_target`, `not_typec` |
| `typea_judgment` | `clean_rule_or_format`, `delete_ambiguous_ok`, `needs_local_evidence`, `needs_external_evidence`, `overclaimed`, `not_typea` |
| `typeb_judgment` | `local_confirmed`, `local_derived_confirmed`, `local_false_positive`, `leakage_suspected`, `weak_literal_match`, `not_typeb` |
| `tbox_judgment` | `causal_schema_repair`, `plausible_schema_update`, `coincidental_or_weak`, `causal_confirmed`, `causal_plausible`, `coincidental_confirmed`, `unknown_causality`, `wrong_polarity`, `wrong_constraint_family`, `needs_discussion`, `not_tbox` |
| `core_recommendation` | `main`, `diagnostic`, `exclude`, `needs_discussion` |
| `notes` | free text |
| `annotator_id` | string |
| `annotation_timestamp_utc` | ISO timestamp |

## 4. Annotation rubric

### TypeC / EXTERNAL_BY_ELIMINATION

Mark `external_confirmed` only when the local graph and rule context do not justify the target and an outside source, domain fact, longer graph path, or retrieval step would plausibly be required.

Mark `local_missed` when the target truth is actually present in local context but the classifier did not match it.

Mark `unknown_or_incomplete` when the target is not well-defined or the artifact lacks enough context to decide.

### TypeA

Mark `clean_rule_or_format` only when the rule or violation shape itself justifies the action without needing local or external evidence.

Mark `needs_local_evidence` or `needs_external_evidence` when the delete/update action requires deciding which value is correct rather than simply rejecting a rule-invalid value.

### TypeB

Mark `local_confirmed` only when the target is clearly available in local graph context.

Mark `local_derived_confirmed` when the target literal is not directly present in local text but is deterministically derived from independent local text. Example: a P8726 Irish Statute Book ID derived from a local description such as `S.I. No. 483/2007` into `2007/si/483/made`.

Mark `weak_literal_match` when local evidence exists only as a short or ambiguous substring.

Mark `leakage_suspected` if evidence appears to expose the post-repair target value through the target property rather than through the synthetic pre-repair state or allowed non-target local context.

### T-box

Mark `causal_schema_repair` when the changed constraint family plausibly explains the violation disappearance.

Mark `coincidental_or_weak` when the property constraint changed around the same time but the relation to the violation is weak.

## 5. Phase D metrics

Compute at minimum:

| Metric | Definition |
|---|---|
| TypeC confirmed-external rate | Fraction of audited `EXTERNAL_BY_ELIMINATION` cases marked `external_confirmed`. |
| TypeC local-missed rate | Fraction marked `local_missed`. |
| TypeC unknown/incomplete rate | Fraction marked `unknown_or_incomplete` or `bad_target`. |
| TypeA overclaim rate | Fraction of clean TypeA sample marked `overclaimed`, `needs_local_evidence`, or `needs_external_evidence`. |
| Delete ambiguity confirmation rate | Fraction of `DELETE_AMBIGUOUS` sample judged correctly ambiguous. |
| TypeB local precision | Fraction of TypeB sample marked `local_confirmed`. |
| TypeB leakage suspicion rate | Fraction marked `leakage_suspected`. |
| T-box causal precision | Fraction of T-box main slices marked `causal_schema_repair` or `plausible_schema_update`. |
| T-box coincidental rate | Fraction of coincidental sample marked `coincidental_or_weak`. |
| Main-score keep rate | Fraction recommended `main` among main-score candidate strata. |
| Diagnostic/exclude rate | Fraction recommended `diagnostic` or `exclude`. |

## 6. Outputs

Phase D should write:

```text
reports/manual_audit/audit_phase_d_v1_seed_13.jsonl
reports/manual_audit/audit_phase_d_v1_seed_13.csv
reports/manual_audit/audit_annotation_schema.json
reports/manual_audit/audit_phase_d_v1_results.json
reports/manual_audit/audit_phase_d_v1_summary.md
```

## 7. Go/no-go thresholds before main LLM experiments

Proceed to main core LLM experiments only if:

1. TypeB local precision is high enough to defend local-context claims; target threshold: at least 0.85 on audited TypeB.
2. Clean TypeA overclaim rate is low; target threshold: at most 0.15.
3. `DELETE_AMBIGUOUS` remains diagnostic-only unless audit shows a clean split into local/external selection cases.
4. `COINCIDENTAL_SCHEMA_CHANGE` remains diagnostic-only unless audit shows strong causal precision.
5. TypeC is reported as external-by-elimination unless the confirmed-external rate is high enough to justify a stronger claim.

These thresholds are not statistical proof. They are quality gates for whether the paper can safely interpret each slice.
