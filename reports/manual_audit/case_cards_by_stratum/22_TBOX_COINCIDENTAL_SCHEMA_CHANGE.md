# TBOX_COINCIDENTAL_SCHEMA_CHANGE

Cases: 27

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `reform_Q10543346_P495_2438012093`

| Field | Value |
|---|---|
| qid | Q10543346 |
| property | P495 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P495::2438012093 |
| tbox_revision_key | TBOX::P495::2438012093 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2438012093,
  "property_revision_prev": 2433993519
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T10:10:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P495",
  "report_revision_new": 2440412851,
  "report_revision_old": 2439979116,
  "report_violation_type": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]",
  "report_violation_type_descriptions_en": [
    "anything that can be offered to a market",
    "any set of human beings",
    "any substance consumed to provide nutritional support for the body; form of energy stored in chemical",
    "sequence of at least two words",
    "characteristic style of cooking practices and traditions",
    "social culture within a larger culture",
    "group of domestic animals with a distinctive phenotype",
    "ethnic group, minority within a state, especially a single ethnic group within a polyethnic state",
    "supernatural animal, generally a hybrid, sometimes part human, whose existence cannot be proven, described in legends, myths, fables, folklore",
    "social and ideological movement in the religious sphere",
    "geographically- or socially-determined language variety",
    "musically-related сultural tradition of an area, people or social class that is passed down through subsequent generations",
    "category of creative works based on stylistic, thematic or technical criteria",
    "ethnic group whose members are also unified by a common language",
    "period and movement in cultural history",
    "group of methods and principles used to teach",
    "methodology for teaching a particular subject",
    "structured form of play",
    "forms of recreational activity, usually physical",
    "tangible or intangible thing, except a service, that satisfies human wants and provides utility",
    "Wikidata metaclass",
    "named person or animal that appears in legends that have some claim to be historical",
    "any conventional method of visually representing verbal or signed communication",
    "abstract systematic rules and conventions of a signifying system, independent of individuals, permitting the expression of parole as instances of this set",
    "... omitted 37 items"
  ],
  "report_violation_type_labels_en": [
    "product",
    "group of humans",
    "food",
    "phrase",
    "cuisine",
    "subculture",
    "breed",
    "ethnic minority group",
    "mythical creature",
    "religious movement",
    "dialect",
    "music tradition",
    "genre",
    "ethnolinguistic group",
    "cultural movement",
    "teaching method",
    "teaching methodology",
    "game",
    "sport",
    "goods",
    "type of food or dish",
    "legendary figure",
    "writing system",
    "language",
    "... omitted 37 items"
  ],
  "report_violation_type_normalized": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]",
  "report_violation_type_qids": [
    "Q2424752",
    "Q16334295",
    "Q2095",
    "Q187931",
    "Q1778821",
    "Q264965",
    "Q38829",
    "Q2531956",
    "Q2239243",
    "Q1826286",
    "Q33384",
    "Q60733114",
    "Q483394",
    "Q4533081",
    "Q2198855",
    "Q1813494",
    "Q350453",
    "Q11410",
    "Q349",
    "Q28877",
    "Q19861951",
    "Q13002315",
    "Q8192",
    "Q4113741",
    "... omitted 37 items"
  ],
  "report_violation_type_raw": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]",
  "value": null,
  "value_current_2026": [
    "Q34"
  ],
  "value_current_2026_descriptions_en": [
    "country in Northern Europe"
  ],
  "value_current_2026_labels_en": [
    "Sweden"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "country of origin of this item (creative work, food, phrase, product, etc.)",
    "label": "country of origin"
  },
  "qid": {
    "description": "poem",
    "label": "Kanske"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]"
    }
  ],
  "candidate_violation_names": [
    "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Trade",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "0823eee8022c1c3c6f79811c8a30c94e6d35336b",
  "hash_before": "6631a0745be7fff52e1004120b4f25dbd181224a",
  "property_revision_id": 2438012093,
  "property_revision_prev": 2433993519,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]"
      }
    ],
    "candidate_violation_names": [
      "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21502838"
    ],
    "changed_constraint_qids_from_entries": [
      "Q21502838"
    ],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502838"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|2424752, Q|16334295, Q|2095, Q|187931, Q|1778821, Q|264965, Q|38829, Q|2531956, Q|2239243, Q|1826286, Q|33384, Q|60733114, Q|483394, Q|4533081, Q|2198855, Q|1813494, Q|350453, Q|11410, Q|349, Q|28877, Q|19861951, Q|13002315, Q|8192, Q|4113741, Q|59544, Q|82821, Q|327496, Q|1485500, Q|17558136, Q|2235308, Q|1824109, Q|223557, Q|14897293, Q|130989, Q|384748, Q|1969448, Q|251777, Q|20203727, Q|17489659, Q|31629, Q|49371, Q|131569, Q|321839, Q|1572600, Q|386724... [truncated 178 chars]"
  }
]
```

---

## 002. `reform_Q108142720_P9818_2435817388`

| Field | Value |
|---|---|
| qid | Q108142720 |
| property | P9818 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21502838 conflicts-with constraint |
| group_key | TBOX::P9818::2435817388 |
| tbox_revision_key | TBOX::P9818::2435817388 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "conflicts-with constraint",
  "decision_constraint_type_qid": "Q21502838"
}
```

#### Repair Target

```json
{
  "author": "Jwillikers",
  "kind": "T_BOX",
  "property_revision_id": 2435817388,
  "property_revision_prev": 2435817341
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-11-30T04:46:58",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9818",
  "report_revision_new": 2435916802,
  "report_revision_old": 2435426806,
  "report_violation_type": "Conflicts with P|747",
  "report_violation_type_normalized": "Conflicts with P|747",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|747",
  "value": null,
  "value_current_2026": [
    "602718"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a book/publication on the Penguin Random House website",
    "label": "Penguin Random House work ID"
  },
  "qid": {
    "description": "2020 graphic novel by Trung Le Nguyen",
    "label": "The Magic Fish"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21502838",
      "mapped_violation_family": "conflicts_with",
      "violation_name": "Conflicts with P|747"
    }
  ],
  "candidate_violation_names": [
    "Conflicts with P|747"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "conflicts-with constraint",
  "mapped_report_constraint_qid": "Q21502838",
  "mapped_report_family": "conflicts_with",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "conflicts-with constraint",
  "mapped_violation_constraint_qid": "Q21502838",
  "mapped_violation_family": "conflicts_with",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Conflicts with P|747",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "conflicts-with constraint",
  "mapped_report_constraint_qid": "Q21502838",
  "mapped_report_family": "conflicts_with",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Conflicts with P|747",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Jwillikers",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "d431b608238c15297e2f4e4d6a4b629e4e24f667",
  "hash_before": "b4b27f076d5528ae0797f18fd64e5bf092758685",
  "property_revision_id": 2435817388,
  "property_revision_prev": 2435817341,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21502838",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Conflicts with P|747"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21502838",
        "mapped_violation_family": "conflicts_with",
        "violation_name": "Conflicts with P|747"
      }
    ],
    "candidate_violation_names": [
      "Conflicts with P|747"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "conflicts-with constraint",
    "mapped_report_constraint_qid": "Q21502838",
    "mapped_report_family": "conflicts_with",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "conflicts-with constraint",
    "mapped_violation_constraint_qid": "Q21502838",
    "mapped_violation_family": "conflicts_with",
    "mapped_violation_reason": "conflicts_with_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Conflicts with P|747",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Conflicts with P|747"
  }
]
```

---

## 003. `reform_Q115918230_P121_2442147265`

| Field | Value |
|---|---|
| qid | Q115918230 |
| property | P121 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P121::2442147265 |
| tbox_revision_key | TBOX::P121::2442147265 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "LBLaiSiNanHai",
  "kind": "T_BOX",
  "property_revision_id": 2442147265,
  "property_revision_prev": 2431827995
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T14:48:04",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P121",
  "report_revision_new": 2442311838,
  "report_revision_old": 2441808522,
  "report_violation_type": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682",
  "report_violation_type_descriptions_en": [
    "distinct and identifiable entity with agency, capable of performing actions",
    "organization which only appears in works of fiction",
    "any set of human beings",
    "occurrence of a fact or object in space-time; instantiation of a property in an object",
    "classification of military organizations by specific features",
    "economic product that directly satisfies wants without producing a lasting asset",
    "place such as an airport, bus station, and train station that is used for managing arriving and departing transport vehicles, and included facility for handling passengers",
    "place, equipment, or service to support a specific function",
    "fictional human or non-human character in a narrative work of art",
    "label applied to a person based on an activity they participate in",
    "storage and delivery agent of information or data",
    "route of a journey",
    "group of vehicles operated by an organization",
    "group of firms that produce a closely related set of raw materials, goods, or services",
    "non-tangible executable component of a computer",
    "series of events which occur over an extended period of time",
    "temporary and scheduled happening, like a conference, festival, competition or similar"
  ],
  "report_violation_type_labels_en": [
    "being",
    "fictional organization",
    "group of humans",
    "occurrence",
    "military unit class",
    "service",
    "transport facility",
    "facility",
    "character",
    "occupation",
    "communications media",
    "itinerary",
    "fleet",
    "industry",
    "software",
    "process",
    "event"
  ],
  "report_violation_type_normalized": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682",
  "report_violation_type_qids": [
    "Q24229398",
    "Q14623646",
    "Q16334295",
    "Q1190554",
    "Q62934160",
    "Q7406919",
    "Q55006986",
    "Q13226383",
    "Q95074",
    "Q12737077",
    "Q340169",
    "Q1322323",
    "Q47311934",
    "Q268592",
    "Q7397",
    "Q3249551",
    "Q1656682"
  ],
  "report_violation_type_raw": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682",
  "value": null,
  "value_current_2026": [
    "Q28843142"
  ],
  "value_current_2026_descriptions_en": [
    "balai kota"
  ],
  "value_current_2026_labels_en": [
    "Town hall of Fontaine-le-Port"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "equipment, installation or service operated by the subject",
    "label": "item operated"
  },
  "qid": {
    "description": "administration municipale of France",
    "label": "Ville de Fontaine-le-Port"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682"
    }
  ],
  "candidate_violation_names": [
    "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "LBLaiSiNanHai",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "91f4dae117a3fb5af4ade635095e90c4ffc4a7b8",
  "hash_before": "341a5a0a3b32c6863ad6fb34bc0f624b66c821a1",
  "property_revision_id": 2442147265,
  "property_revision_prev": 2431827995,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682"
      }
    ],
    "candidate_violation_names": [
      "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|24229398, Q|14623646, Q|16334295, Q|1190554, Q|62934160, Q|7406919, Q|55006986, Q|13226383, Q|95074, Q|12737077, Q|340169, Q|1322323, Q|47311934, Q|268592, Q|7397, Q|3249551, Q|1656682"
  }
]
```

---

## 004. `reform_Q119749395_P21_2442825468`

| Field | Value |
|---|---|
| qid | Q119749395 |
| property | P21 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P21::2442825468 |
| tbox_revision_key | TBOX::P21::2442825468 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Jerimee",
  "kind": "T_BOX",
  "property_revision_id": 2442825468,
  "property_revision_prev": 2440297725
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T13:25:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P21",
  "report_revision_new": 2444061546,
  "report_revision_old": 2443887960,
  "report_violation_type": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "kingdom of multicellular eukaryotic organisms",
    "character known only from narrations (fictional or in a factual manner) without a proof of existence; includes fictional, mythical, legendary or religious characters and similar",
    "entity that has no physical realisation",
    "Latin phrase; alternate self",
    "preserved remains or traces of organisms from a past geological age",
    "any individual living being or physical living system",
    "mechanical or virtual artificial agent carrying out physical activities, which can be guided by an external control device or the control may be embedded within",
    "anthropomorphic sexual device",
    "a human-sounding voice generated by a computer",
    "group of one or more organism(s), which a taxonomist adjudges to be a unit",
    "teknonym in an Arabic name, the name of an adult derived from their eldest son",
    "prenatal organism between the embryonic state and birth",
    "individual who died before or during birth",
    "type of a doll or (action) figure that may appear in different variants",
    "organism not more specified in a work of fiction",
    "model of a character or a human being, often used as a toy for children or an artistic hobby for adults",
    "group of one or more fictional organism(s), which a (fictional) taxonomist adjudges to be a unit",
    "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
    "fictitious name that a person or group assumes for a particular purpose, which differs from their original or true name (orthonym)"
  ],
  "report_violation_type_labels_en": [
    "human",
    "Animalia",
    "imaginary character",
    "abstract being",
    "alter ego",
    "fossil",
    "organism",
    "robot",
    "sex doll",
    "synthetic voice",
    "taxon",
    "kunya",
    "human fetus",
    "stillborn child",
    "doll or action figure model",
    "fictional creature",
    "doll",
    "fictional taxon",
    "individual animal",
    "pseudonym"
  ],
  "report_violation_type_normalized": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002",
  "report_violation_type_qids": [
    "Q5",
    "Q729",
    "Q115537581",
    "Q15619164",
    "Q201662",
    "Q40614",
    "Q7239",
    "Q11012",
    "Q10832",
    "Q79600797",
    "Q16521",
    "Q1285470",
    "Q26513",
    "Q2345820",
    "Q111282474",
    "Q2593744",
    "Q168658",
    "Q15707583",
    "Q26401003",
    "Q61002"
  ],
  "report_violation_type_raw": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002",
  "value": null,
  "value_current_2026": [
    "Q6581097"
  ],
  "value_current_2026_descriptions_en": [
    "to be used in \"sex or gender\" (P21) to indicate that the human subject is a male or \"semantic gender\" (P10339) to indicate that a word refers to a male person"
  ],
  "value_current_2026_labels_en": [
    "male"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
    "label": "sex or gender"
  },
  "qid": {
    "description": "Polish engineer, founder of the Tomaszów Mineral Resources Mine \"Biała Góra\"",
    "label": "Bohdan Łoziński"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002"
    }
  ],
  "candidate_violation_names": [
    "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Jerimee",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "9ffdd87dbef8eb12ce4d41577040011ac2ef7891",
  "hash_before": "4153e44669496bf040a48eb9d9fecb724aee43d6",
  "property_revision_id": 2442825468,
  "property_revision_prev": 2440297725,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002"
      }
    ],
    "candidate_violation_names": [
      "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21502838"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502838"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|5, Q|729, Q|115537581, Q|15619164, Q|201662, Q|40614, Q|7239, Q|11012, Q|10832, Q|79600797, Q|16521, Q|1285470, Q|26513, Q|2345820, Q|111282474, Q|2593744, Q|168658, Q|15707583, Q|26401003, Q|61002"
  }
]
```

---

## 005. `reform_Q131806934_P9818_2435817388`

| Field | Value |
|---|---|
| qid | Q131806934 |
| property | P9818 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21502838 conflicts-with constraint |
| group_key | TBOX::P9818::2435817388 |
| tbox_revision_key | TBOX::P9818::2435817388 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "conflicts-with constraint",
  "decision_constraint_type_qid": "Q21502838"
}
```

#### Repair Target

```json
{
  "author": "Jwillikers",
  "kind": "T_BOX",
  "property_revision_id": 2435817388,
  "property_revision_prev": 2435817341
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-11-30T04:46:58",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9818",
  "report_revision_new": 2435916802,
  "report_revision_old": 2435426806,
  "report_violation_type": "Conflicts with P|747",
  "report_violation_type_normalized": "Conflicts with P|747",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|747",
  "value": null,
  "value_current_2026": [
    "742811"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a book/publication on the Penguin Random House website",
    "label": "Penguin Random House work ID"
  },
  "qid": {
    "description": "2025 memoir by Bill Gates",
    "label": "Source Code"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21502838",
      "mapped_violation_family": "conflicts_with",
      "violation_name": "Conflicts with P|747"
    }
  ],
  "candidate_violation_names": [
    "Conflicts with P|747"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "conflicts-with constraint",
  "mapped_report_constraint_qid": "Q21502838",
  "mapped_report_family": "conflicts_with",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "conflicts-with constraint",
  "mapped_violation_constraint_qid": "Q21502838",
  "mapped_violation_family": "conflicts_with",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Conflicts with P|747",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "conflicts-with constraint",
  "mapped_report_constraint_qid": "Q21502838",
  "mapped_report_family": "conflicts_with",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Conflicts with P|747",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Jwillikers",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "d431b608238c15297e2f4e4d6a4b629e4e24f667",
  "hash_before": "b4b27f076d5528ae0797f18fd64e5bf092758685",
  "property_revision_id": 2435817388,
  "property_revision_prev": 2435817341,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21502838",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Conflicts with P|747"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21502838",
        "mapped_violation_family": "conflicts_with",
        "violation_name": "Conflicts with P|747"
      }
    ],
    "candidate_violation_names": [
      "Conflicts with P|747"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "conflicts-with constraint",
    "mapped_report_constraint_qid": "Q21502838",
    "mapped_report_family": "conflicts_with",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "conflicts-with constraint",
    "mapped_violation_constraint_qid": "Q21502838",
    "mapped_violation_family": "conflicts_with",
    "mapped_violation_reason": "conflicts_with_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Conflicts with P|747",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Conflicts with P|747"
  }
]
```

---

## 006. `reform_Q136796116_P856_2446852358`

| Field | Value |
|---|---|
| qid | Q136796116 |
| property | P856 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21502410 distinct-values constraint |
| group_key | TBOX::P856::2446852358 |
| tbox_revision_key | TBOX::P856::2446852358 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "distinct-values constraint",
  "decision_constraint_type_qid": "Q21502410"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2446852358,
  "property_revision_prev": 2443941926
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-25T18:00:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P856",
  "report_revision_new": 2447046423,
  "report_revision_old": 2446480257,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "http://dx.doi.org/10.1101/2022.03.25.485874"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "URL of the official page of an item (current or former). Usage: If a listed URL no longer points to the official website, do not remove it, but see the \"Hijacked or dead websites\" section of the Talk page",
    "label": "official website"
  },
  "qid": {
    "description": null,
    "label": "RNA-targeting CRISPR-Cas13 Provides Broad-spectrum Phage Immunity"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "value_or_property_overlap_only",
      "candidate_score": 20,
      "mapped_violation_constraint_qid": "Q21502410",
      "mapped_violation_family": "distinct_values",
      "violation_name": "Unique value"
    }
  ],
  "candidate_violation_names": [
    "Unique value"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "distinct-values constraint",
  "mapped_violation_constraint_qid": "Q21502410",
  "mapped_violation_family": "distinct_values",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Unique value",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Unique value",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Clemens Dulcis",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "a36a61f0e86f73f6deb18c55577835a1b9bbf650",
  "hash_before": "a24e4427528acfa48d593fe8b3d3db4c276c2009",
  "property_revision_id": 2446852358,
  "property_revision_prev": 2443941926,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "value_or_property_overlap_only",
        "candidate_score": 20,
        "mapped_violation_constraint_qid": "Q21502410",
        "mapped_violation_family": "distinct_values",
        "violation_name": "Unique value"
      }
    ],
    "candidate_violation_names": [
      "Unique value"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "distinct-values constraint",
    "mapped_report_constraint_qid": "Q21502410",
    "mapped_report_family": "distinct_values",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "distinct-values constraint",
    "mapped_violation_constraint_qid": "Q21502410",
    "mapped_violation_family": "distinct_values",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Unique value",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Unique value"
  }
]
```

---

## 007. `reform_Q136925578_P856_2447213274`

| Field | Value |
|---|---|
| qid | Q136925578 |
| property | P856 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21502410 distinct-values constraint |
| group_key | TBOX::P856::2447213274 |
| tbox_revision_key | TBOX::P856::2447213274 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "distinct-values constraint",
  "decision_constraint_type_qid": "Q21502410"
}
```

#### Repair Target

```json
{
  "author": "Trivialist",
  "kind": "T_BOX",
  "property_revision_id": 2447213274,
  "property_revision_prev": 2446852358
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T11:39:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P856",
  "report_revision_new": 2447351370,
  "report_revision_old": 2447046423,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "http://dx.doi.org/10.59350/1r9b8-v4y82"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "URL of the official page of an item (current or former). Usage: If a listed URL no longer points to the official website, do not remove it, but see the \"Hijacked or dead websites\" section of the Talk page",
    "label": "official website"
  },
  "qid": {
    "description": null,
    "label": "A databank of molecular dynamics reaction trajectories (DDT) focused on undergraduate teaching."
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "value_or_property_overlap_only",
      "candidate_score": 20,
      "mapped_violation_constraint_qid": "Q21502410",
      "mapped_violation_family": "distinct_values",
      "violation_name": "Unique value"
    }
  ],
  "candidate_violation_names": [
    "Unique value"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "distinct-values constraint",
  "mapped_violation_constraint_qid": "Q21502410",
  "mapped_violation_family": "distinct_values",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Unique value",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Unique value",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Trivialist",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "a3c6b9ae9ecf7dae4721f27e9093f68129421a13",
  "hash_before": "a36a61f0e86f73f6deb18c55577835a1b9bbf650",
  "property_revision_id": 2447213274,
  "property_revision_prev": 2446852358,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "value_or_property_overlap_only",
        "candidate_score": 20,
        "mapped_violation_constraint_qid": "Q21502410",
        "mapped_violation_family": "distinct_values",
        "violation_name": "Unique value"
      }
    ],
    "candidate_violation_names": [
      "Unique value"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "distinct-values constraint",
    "mapped_report_constraint_qid": "Q21502410",
    "mapped_report_family": "distinct_values",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "distinct-values constraint",
    "mapped_violation_constraint_qid": "Q21502410",
    "mapped_violation_family": "distinct_values",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Unique value",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Unique value"
  }
]
```

---

## 008. `reform_Q139829_P2670_2370327892`

| Field | Value |
|---|---|
| qid | Q139829 |
| property | P2670 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q25796498 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P2670::2370327892 |
| tbox_revision_key | TBOX::P2670::2370327892 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "BartoszKonkol",
  "kind": "T_BOX",
  "property_revision_id": 2370327892,
  "property_revision_prev": 2345278158
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-06-30T11:17:51",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2670",
  "report_revision_new": 2370604028,
  "report_revision_old": 2369890566,
  "report_violation_type": "Item P|31",
  "report_violation_type_normalized": "Item P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|31",
  "value": null,
  "value_current_2026": [
    "Q26401",
    "Q26382",
    "Q3064117"
  ],
  "value_current_2026_descriptions_en": [
    "special kind of point that describes the corners or intersections of geometric shapes",
    "line segment joining two adjacent vertices in a polygon or polytope",
    "in geometry, a planar surface that forms part of the boundary of a solid object"
  ],
  "value_current_2026_labels_en": [
    "vertex",
    "edge",
    "face"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the subject has one or more parts of the object class",
    "label": "has part(s) of the class"
  },
  "qid": {
    "description": "Johnson solid",
    "label": "gyroelongated pentagonal pyramid"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|31"
    }
  ],
  "candidate_violation_names": [
    "Item P|31"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q52004125"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52004125"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|31",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q52004125"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52004125"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|31",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "BartoszKonkol",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "ca2cf7025a412e51b7c45debae8a24f6b26c788d",
  "hash_before": "f625c9d92869ac9fde4d6504167caeaf4393dcad",
  "property_revision_id": 2370327892,
  "property_revision_prev": 2345278158,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503247",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Item P|31"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|31"
      }
    ],
    "candidate_violation_names": [
      "Item P|31"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q52004125"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q52004125"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|31",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Item P|31"
  }
]
```

---

## 009. `reform_Q16472035_P131_2437507566`

| Field | Value |
|---|---|
| qid | Q16472035 |
| property | P131 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | TBOX::P131::2437507566 |
| tbox_revision_key | TBOX::P131::2437507566 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "author": "Necessarycoot72",
  "kind": "T_BOX",
  "property_revision_id": 2437507566,
  "property_revision_prev": 2437506232
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T14:12:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
  "report_revision_new": 2440051055,
  "report_revision_old": 2439581737,
  "report_violation_type": "Target required claim P|17",
  "report_violation_type_normalized": "Target required claim P|17",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|17",
  "value": null,
  "value_current_2026": [
    "Q16459845",
    "Q12672407"
  ],
  "value_current_2026_descriptions_en": [
    "former district municipality of Lithuania",
    null
  ],
  "value_current_2026_labels_en": [
    "Marijampolė District Municipality",
    "Sasnavos valsčius"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
    "label": "located in the administrative territorial entity"
  },
  "qid": {
    "description": null,
    "label": "Sasnavos apylinkė"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510864",
      "mapped_violation_family": "value_required_statement",
      "violation_name": "Target required claim P|17"
    }
  ],
  "candidate_violation_names": [
    "Target required claim P|17"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value requires statement constraint",
  "mapped_violation_constraint_qid": "Q21510864",
  "mapped_violation_family": "value_required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Target required claim P|17",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Target required claim P|17",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Necessarycoot72",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "5760e2f84413098a914c8a5b9269acb4fce68305",
  "hash_before": "69774ae093d067e99709c2489d12aec3785522ce",
  "property_revision_id": 2437507566,
  "property_revision_prev": 2437506232,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510864",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Target required claim P|17"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510864",
        "mapped_violation_family": "value_required_statement",
        "violation_name": "Target required claim P|17"
      }
    ],
    "candidate_violation_names": [
      "Target required claim P|17"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value requires statement constraint",
    "mapped_report_constraint_qid": "Q21510864",
    "mapped_report_family": "value_required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value requires statement constraint",
    "mapped_violation_constraint_qid": "Q21510864",
    "mapped_violation_family": "value_required_statement",
    "mapped_violation_reason": "value_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Target required claim P|17",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Target required claim P|17"
  }
]
```

---

## 010. `reform_Q16472548_P131_2437507566`

| Field | Value |
|---|---|
| qid | Q16472548 |
| property | P131 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | TBOX::P131::2437507566 |
| tbox_revision_key | TBOX::P131::2437507566 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "author": "Necessarycoot72",
  "kind": "T_BOX",
  "property_revision_id": 2437507566,
  "property_revision_prev": 2437506232
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T14:12:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
  "report_revision_new": 2440051055,
  "report_revision_old": 2439581737,
  "report_violation_type": "Target required claim P|17",
  "report_violation_type_normalized": "Target required claim P|17",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|17",
  "value": null,
  "value_current_2026": [
    "Q2089772",
    "Q16472550"
  ],
  "value_current_2026_descriptions_en": [
    "municipality in Lithuania",
    null
  ],
  "value_current_2026_labels_en": [
    "Anykščiai District Municipality",
    "Skiemonių valsčius"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
    "label": "located in the administrative territorial entity"
  },
  "qid": {
    "description": null,
    "label": "Skiemonių apylinkė"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510864",
      "mapped_violation_family": "value_required_statement",
      "violation_name": "Target required claim P|17"
    }
  ],
  "candidate_violation_names": [
    "Target required claim P|17"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value requires statement constraint",
  "mapped_violation_constraint_qid": "Q21510864",
  "mapped_violation_family": "value_required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Target required claim P|17",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Target required claim P|17",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Necessarycoot72",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "5760e2f84413098a914c8a5b9269acb4fce68305",
  "hash_before": "69774ae093d067e99709c2489d12aec3785522ce",
  "property_revision_id": 2437507566,
  "property_revision_prev": 2437506232,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510864",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Target required claim P|17"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510864",
        "mapped_violation_family": "value_required_statement",
        "violation_name": "Target required claim P|17"
      }
    ],
    "candidate_violation_names": [
      "Target required claim P|17"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value requires statement constraint",
    "mapped_report_constraint_qid": "Q21510864",
    "mapped_report_family": "value_required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value requires statement constraint",
    "mapped_violation_constraint_qid": "Q21510864",
    "mapped_violation_family": "value_required_statement",
    "mapped_violation_reason": "value_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Target required claim P|17",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Target required claim P|17"
  }
]
```

---

## 011. `reform_Q16473381_P131_2437507566`

| Field | Value |
|---|---|
| qid | Q16473381 |
| property | P131 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | TBOX::P131::2437507566 |
| tbox_revision_key | TBOX::P131::2437507566 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "author": "Necessarycoot72",
  "kind": "T_BOX",
  "property_revision_id": 2437507566,
  "property_revision_prev": 2437506232
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T14:12:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
  "report_revision_new": 2440051055,
  "report_revision_old": 2439581737,
  "report_violation_type": "Target required claim P|17",
  "report_violation_type_normalized": "Target required claim P|17",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|17",
  "value": null,
  "value_current_2026": [
    "Q12673389",
    "Q12667415"
  ],
  "value_current_2026_descriptions_en": [
    null,
    null
  ],
  "value_current_2026_labels_en": [
    "Smėlių rajonas",
    "Pabaisko valsčius"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
    "label": "located in the administrative territorial entity"
  },
  "qid": {
    "description": null,
    "label": "Steponavos apylinkė"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510864",
      "mapped_violation_family": "value_required_statement",
      "violation_name": "Target required claim P|17"
    }
  ],
  "candidate_violation_names": [
    "Target required claim P|17"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value requires statement constraint",
  "mapped_violation_constraint_qid": "Q21510864",
  "mapped_violation_family": "value_required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Target required claim P|17",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Target required claim P|17",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Necessarycoot72",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "5760e2f84413098a914c8a5b9269acb4fce68305",
  "hash_before": "69774ae093d067e99709c2489d12aec3785522ce",
  "property_revision_id": 2437507566,
  "property_revision_prev": 2437506232,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510864",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Target required claim P|17"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510864",
        "mapped_violation_family": "value_required_statement",
        "violation_name": "Target required claim P|17"
      }
    ],
    "candidate_violation_names": [
      "Target required claim P|17"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value requires statement constraint",
    "mapped_report_constraint_qid": "Q21510864",
    "mapped_report_family": "value_required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value requires statement constraint",
    "mapped_violation_constraint_qid": "Q21510864",
    "mapped_violation_family": "value_required_statement",
    "mapped_violation_reason": "value_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Target required claim P|17",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Target required claim P|17"
  }
]
```

---

## 012. `reform_Q214048_P17_2442267688`

| Field | Value |
|---|---|
| qid | Q214048 |
| property | P17 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | TBOX::P17::2442267688 |
| tbox_revision_key | TBOX::P17::2442267688 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2442267688,
  "property_revision_prev": 2442209108
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T13:51:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2444068162,
  "report_revision_old": 2443895183,
  "report_violation_type": "empty",
  "report_violation_type_normalized": "empty",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "empty",
  "report_violation_types": [
    "empty",
    "Target required claim P|30"
  ],
  "value": null,
  "value_current_2026": [
    "Q34"
  ],
  "value_current_2026_descriptions_en": [
    "country in Northern Europe"
  ],
  "value_current_2026_labels_en": [
    "Sweden"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "municipality in Stockholm County, Sweden",
    "label": "Norrtälje Municipality"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510864",
      "mapped_violation_family": "value_required_statement",
      "violation_name": "Target required claim P|30"
    },
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "empty"
    }
  ],
  "candidate_violation_names": [
    "empty",
    "Target required claim P|30"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value requires statement constraint",
  "mapped_violation_constraint_qid": "Q21510864",
  "mapped_violation_family": "value_required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Target required claim P|30",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Target required claim P|30",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Clemens Dulcis",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "5ef7d89649c9ff90fe1a52368496e645eacec34b",
  "hash_before": "7165d90b925db25df453378686fd6771831f1cc6",
  "property_revision_id": 2442267688,
  "property_revision_prev": 2442209108,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510864",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Target required claim P|30"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510864",
        "mapped_violation_family": "value_required_statement",
        "violation_name": "Target required claim P|30"
      },
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "empty"
      }
    ],
    "candidate_violation_names": [
      "empty",
      "Target required claim P|30"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value requires statement constraint",
    "mapped_report_constraint_qid": "Q21510864",
    "mapped_report_family": "value_required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value requires statement constraint",
    "mapped_violation_constraint_qid": "Q21510864",
    "mapped_violation_family": "value_required_statement",
    "mapped_violation_reason": "value_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Target required claim P|30",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Target required claim P|30"
  }
]
```

---

## 013. `reform_Q24090499_P2440_1695570929`

| Field | Value |
|---|---|
| qid | Q24090499 |
| property | P2440 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21510856 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P2440::1695570929 |
| tbox_revision_key | TBOX::P2440::1695570929 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1695570929,
  "property_revision_prev": 1695570487
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-08-04T22:52:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2440",
  "report_revision_new": 1696514016,
  "report_revision_old": 1695233363,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "Танабэ"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "conversion of text to alternate script (use as a qualifier for monolingual text statements; please use specific property if possible)",
    "label": "transliteration or transcription"
  },
  "qid": {
    "description": "Japanese family name (田辺)",
    "label": "Tanabe"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "fb47ab16c6cc776f3f9035247d6ca1a05b92022c",
  "hash_before": "c070f880066096d1b80b62193b70ee20257c2e93",
  "property_revision_id": 1695570929,
  "property_revision_prev": 1695570487,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510851",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q53869507"
    ],
    "changed_constraint_qids_from_entries": [
      "Q53869507"
    ],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q53869507"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 014. `reform_Q2574551_P5800_1839493915`

| Field | Value |
|---|---|
| qid | Q2574551 |
| property | P5800 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510856 required qualifier constraint |
| group_key | TBOX::P5800::1839493915 |
| tbox_revision_key | TBOX::P5800::1839493915 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "required qualifier constraint",
  "decision_constraint_type_qid": "Q21510856"
}
```

#### Repair Target

```json
{
  "author": "OmegaFallon",
  "kind": "T_BOX",
  "property_revision_id": 1839493915,
  "property_revision_prev": 1839493614
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-02-23T20:14:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5800",
  "report_revision_new": 1840033229,
  "report_revision_old": 1839112494,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "report_violation_types": [
    "Mandatory Qualifiers",
    "One of"
  ],
  "value": null,
  "value_current_2026": [
    "Q247297"
  ],
  "value_current_2026_descriptions_en": [
    "stock character, expendable adherents of villain"
  ],
  "value_current_2026_labels_en": [
    "henchperson"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "narrative role of this character (should be used as a qualifier with P674 or restricted to a certain work using P10663)",
    "label": "narrative role"
  },
  "qid": {
    "description": "トランスフォーマーシリーズの登場キャラクター",
    "label": "Skywarp"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510856",
      "mapped_violation_family": "mandatory_qualifier",
      "violation_name": "Mandatory Qualifiers"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510859",
      "mapped_violation_family": "one_of",
      "violation_name": "One of"
    }
  ],
  "candidate_violation_names": [
    "Mandatory Qualifiers",
    "One of"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "mandatory qualifier constraint",
  "mapped_report_constraint_qid": "Q21510856",
  "mapped_report_family": "mandatory_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "mandatory qualifier constraint",
  "mapped_violation_constraint_qid": "Q21510856",
  "mapped_violation_family": "mandatory_qualifier",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Mandatory Qualifiers",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "mandatory qualifier constraint",
  "mapped_report_constraint_qid": "Q21510856",
  "mapped_report_family": "mandatory_qualifier",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Mandatory Qualifiers",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "OmegaFallon",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "427147685b3b5f5354652f600db4a0a14896f66b",
  "hash_before": "f4c0d35976e743caeb893ae9e4ab41709d66736c",
  "property_revision_id": 1839493915,
  "property_revision_prev": 1839493614,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510856",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Mandatory Qualifiers"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510856",
        "mapped_violation_family": "mandatory_qualifier",
        "violation_name": "Mandatory Qualifiers"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510859",
        "mapped_violation_family": "one_of",
        "violation_name": "One of"
      }
    ],
    "candidate_violation_names": [
      "Mandatory Qualifiers",
      "One of"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21503250"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503250"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "mandatory qualifier constraint",
    "mapped_report_constraint_qid": "Q21510856",
    "mapped_report_family": "mandatory_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "mandatory qualifier constraint",
    "mapped_violation_constraint_qid": "Q21510856",
    "mapped_violation_family": "mandatory_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Mandatory Qualifiers",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Mandatory Qualifiers"
  }
]
```

---

## 015. `reform_Q26387_P1296_1648181364`

| Field | Value |
|---|---|
| qid | Q26387 |
| property | P1296 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | TBOX::P1296::1648181364 |
| tbox_revision_key | TBOX::P1296::1648181364 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "author": "Nikki",
  "kind": "T_BOX",
  "property_revision_id": 1648181364,
  "property_revision_prev": 1638585077
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-06-01T10:55:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1296",
  "report_revision_new": 1651853720,
  "report_revision_old": 1647499267,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "0078496",
    "0078498"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for an item in the Gran Enciclopèdia Catalana. Replaced with \"Gran Enciclopèdia Catalana ID (P12385)\"",
    "label": "Gran Enciclopèdia Catalana ID (former scheme)"
  },
  "qid": {
    "description": "species of eel",
    "label": "European eel"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "lexeme requires language constraint",
    "qid": "Q55819106"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q19474404",
      "mapped_violation_family": "single_value",
      "violation_name": "Single value"
    }
  ],
  "candidate_violation_names": [
    "Single value"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "single-value constraint",
  "mapped_report_constraint_qid": "Q19474404",
  "mapped_report_family": "single_value",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "single-value constraint",
  "mapped_violation_constraint_qid": "Q19474404",
  "mapped_violation_family": "single_value",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Single value",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "single-value constraint",
  "mapped_report_constraint_qid": "Q19474404",
  "mapped_report_family": "single_value",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Single value",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Nikki",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "48ae3a4b24989e2f2e751e655c874dceb0e12c5f",
  "hash_before": "02755eb4bf2c3dd6a77ae6165cb9c4894a779a45",
  "property_revision_id": 1648181364,
  "property_revision_prev": 1638585077,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q19474404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Single value"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q19474404",
        "mapped_violation_family": "single_value",
        "violation_name": "Single value"
      }
    ],
    "candidate_violation_names": [
      "Single value"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21503250"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503250"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "single-value constraint",
    "mapped_report_constraint_qid": "Q19474404",
    "mapped_report_family": "single_value",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "single-value constraint",
    "mapped_violation_constraint_qid": "Q19474404",
    "mapped_violation_family": "single_value",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Single value",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Single value"
  }
]
```

---

## 016. `reform_Q28680219_P159_2438613236`

| Field | Value |
|---|---|
| qid | Q28680219 |
| property | P159 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P159::2438613236 |
| tbox_revision_key | TBOX::P159::2438613236 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Arlo Barnes",
  "kind": "T_BOX",
  "property_revision_id": 2438613236,
  "property_revision_prev": 2436458585
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T13:19:18",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P159",
  "report_revision_new": 2440033122,
  "report_revision_old": 2439570804,
  "report_violation_type": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
  "report_violation_type_descriptions_en": [
    "social entity established to meet needs or pursue goals",
    "organization which only appears in works of fiction",
    "scheduled publication containing news of events, articles, features, editorials, and advertising; online, in print, or (usually) both",
    "territorial entity for administration purposes, with or without its own local government",
    "publication type, serial publication that appears in a new edition on a regular schedule",
    "business organization which only exists in fiction",
    "set of related web pages served from a single web domain",
    "legal entity representing an association of people, whether natural, legal or a mixture of both, with a specific objective",
    "organisational part of a government responsible for specific public services, such as health, judiciary, education, transportation, foreign affairs, etc",
    "a designated body with authority",
    "identification for a good or service",
    "transition of power from one president to another",
    "collaborative enterprise, frequently involving research or design, that is carefully planned to achieve a particular aim",
    "organized set of events or activities focused on a theme (cultural, religious or other) that recurs regularly (e.g. once a year) and lasts anywhere from several hours to weeks",
    "type of administrative division, in some countries",
    "series of operations undertaken to achieve a defined goal",
    "temporary and scheduled happening, like a conference, festival, competition or similar",
    "contest for a prize or award",
    "large area with high population density and infrastructure of built environment",
    "organization undertaking commercial, industrial, or professional activity",
    "entity that no longer operates or is terminated",
    "defunct, destroyed, demolished, or discontinued organization, establishment, group, etc.",
    "theatre organization that produces puppetry performances"
  ],
  "report_violation_type_labels_en": [
    "organization",
    "fictional organization",
    "newspaper",
    "administrative territorial entity",
    "periodical",
    "fictional company",
    "website",
    "company",
    "government agency",
    "governing body",
    "brand",
    "presidential transition",
    "project",
    "festival",
    "district",
    "campaign",
    "event",
    "competition",
    "urban area",
    "business",
    "former entity",
    "defunct organization",
    "puppetry company"
  ],
  "report_violation_type_normalized": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
  "report_violation_type_qids": [
    "Q43229",
    "Q14623646",
    "Q11032",
    "Q56061",
    "Q1002697",
    "Q5446565",
    "Q35127",
    "Q783794",
    "Q327333",
    "Q895526",
    "Q431289",
    "Q104921473",
    "Q170584",
    "Q132241",
    "Q149621",
    "Q6056746",
    "Q1656682",
    "Q841654",
    "Q702492",
    "Q4830453",
    "Q15893266",
    "Q55097243",
    "Q3982337"
  ],
  "report_violation_type_raw": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
  "value": null,
  "value_current_2026": [
    "Q597"
  ],
  "value_current_2026_descriptions_en": [
    "municipality and capital city of Portugal"
  ],
  "value_current_2026_labels_en": [
    "Lisbon"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "city or town where an organization's headquarters is or has been situated. Use P276 qualifier for specific building",
    "label": "headquarters location"
  },
  "qid": {
    "description": "Интегралистская антикоммунистическая политическая партия Португалии (1974)",
    "label": "Partido do Progresso / Movimento Federalista Português"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337"
    }
  ],
  "candidate_violation_names": [
    "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Arlo Barnes",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "c9ee80954d6bab31b8d9166eba86ae29f8ec4a36",
  "hash_before": "299c5fe622eb5253084572fbe0e5a2bd6c3b94af",
  "property_revision_id": 2438613236,
  "property_revision_prev": 2436458585,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337"
      }
    ],
    "candidate_violation_names": [
      "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337"
  }
]
```

---

## 017. `reform_Q2900306_P242_1682463811`

| Field | Value |
|---|---|
| qid | Q2900306 |
| property | P242 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P242::1682463811 |
| tbox_revision_key | TBOX::P242::1682463811 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1682463811,
  "property_revision_prev": 1682281639
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-07-20T11:14:04",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P242",
  "report_revision_new": 1683039747,
  "report_revision_old": 1682123108,
  "report_violation_type": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885",
  "report_violation_type_descriptions_en": [
    "point, line or area on or near Earth",
    "relations between two subjects of public international law",
    "territorial entity for administration purposes, with or without its own local government",
    "organization established by treaty between governments",
    "alliance between different states with the purpose to cooperate militarily",
    "2D or 3D defined space on something, mainly in terrestrial and astrophysics sciences",
    "occurrence of a fact or object in space-time; instantiation of a property in an object",
    "image on the celestial sphere consisting of stars according to any current of historical system or description",
    "group of independent or autonomous territories sharing a given set of traits",
    "structured system of communication",
    "place that exists only in fiction and not in reality",
    "taxonomic rank between family and genus",
    "socially defined category of people who identify with each other",
    "human-designed and -made structure",
    "group of languages related through descent from a common ancestor",
    "semantic unit understood in different ways, e.g. as mental representation, ability or abstract object (philosophy)"
  ],
  "report_violation_type_labels_en": [
    "geographic location",
    "bilateral relation",
    "administrative territorial entity",
    "international organization",
    "military alliance",
    "region",
    "occurrence",
    "constellation",
    "geopolitical group",
    "language",
    "fictional location",
    "tribe",
    "ethnic group",
    "architectural structure",
    "language family",
    "concept"
  ],
  "report_violation_type_normalized": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885",
  "report_violation_type_qids": [
    "Q2221906",
    "Q15221623",
    "Q56061",
    "Q484652",
    "Q1127126",
    "Q82794",
    "Q1190554",
    "Q8928",
    "Q52110228",
    "Q315",
    "Q3895768",
    "Q227936",
    "Q41710",
    "Q811979",
    "Q25295",
    "Q151885"
  ],
  "report_violation_type_raw": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885",
  "value": null,
  "value_current_2026": [
    "LesBeunes.jpg"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "geographic map image which highlights the location of the subject within some larger entity",
    "label": "locator map image"
  },
  "qid": {
    "description": "name of several rivers in Dordogne, France",
    "label": "Beune"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885"
    }
  ],
  "candidate_violation_names": [
    "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21502410"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502410"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21502410"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502410"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "3145a1b36a4b0d500e98964902b60ebb87d52762",
  "hash_before": "d0acb3ef1ca8f81d862a9a5785d7c98c7782b750",
  "property_revision_id": 1682463811,
  "property_revision_prev": 1682281639,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885"
      }
    ],
    "candidate_violation_names": [
      "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21502410"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502410"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885"
  }
]
```

---

## 018. `reform_Q3540477_P31_2442828050`

| Field | Value |
|---|---|
| qid | Q3540477 |
| property | P31 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | TBOX::P31::2442828050 |
| tbox_revision_key | TBOX::P31::2442828050 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "author": "Jerimee",
  "kind": "T_BOX",
  "property_revision_id": 2442828050,
  "property_revision_prev": 2442704771
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T00:13:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
  "report_revision_new": 2443908957,
  "report_revision_old": 2443512419,
  "report_violation_type": "Target required claim P|279",
  "report_violation_type_normalized": "Target required claim P|279",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|279",
  "value": null,
  "value_current_2026": [
    "Q3540478"
  ],
  "value_current_2026_descriptions_en": [
    "former football competition in France"
  ],
  "value_current_2026_labels_en": [
    "Trophée de France"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
    "label": "instance of"
  },
  "qid": {
    "description": null,
    "label": "Trophée de France 1908"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510864",
      "mapped_violation_family": "value_required_statement",
      "violation_name": "Target required claim P|279"
    }
  ],
  "candidate_violation_names": [
    "Target required claim P|279"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52558054"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value requires statement constraint",
  "mapped_violation_constraint_qid": "Q21510864",
  "mapped_violation_family": "value_required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Target required claim P|279",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52558054"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value requires statement constraint",
  "mapped_report_constraint_qid": "Q21510864",
  "mapped_report_family": "value_required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Target required claim P|279",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Jerimee",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f70bbdec141115c1b3be9558021013a0a7950f89",
  "hash_before": "d848f4a3f57e469600cc4f5f1793452d7f9c6ef1",
  "property_revision_id": 2442828050,
  "property_revision_prev": 2442704771,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510864",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Target required claim P|279"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510864",
        "mapped_violation_family": "value_required_statement",
        "violation_name": "Target required claim P|279"
      }
    ],
    "candidate_violation_names": [
      "Target required claim P|279"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q52558054"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q52558054"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value requires statement constraint",
    "mapped_report_constraint_qid": "Q21510864",
    "mapped_report_family": "value_required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value requires statement constraint",
    "mapped_violation_constraint_qid": "Q21510864",
    "mapped_violation_family": "value_required_statement",
    "mapped_violation_reason": "value_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Target required claim P|279",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Target required claim P|279"
  }
]
```

---

## 019. `reform_Q3608979_P2521_1725922197`

| Field | Value |
|---|---|
| qid | Q3608979 |
| property | P2521 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q53869507 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503250 type constraint |
| group_key | TBOX::P2521::1725922197 |
| tbox_revision_key | TBOX::P2521::1725922197 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-09-11T11:10:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2521",
  "report_revision_new": 1726203339,
  "report_revision_old": 1725211201,
  "report_violation_type": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
  "report_violation_type_descriptions_en": [
    "occupation requiring specialized training",
    "social role with a set of powers and responsibilities within an organization",
    "label applied to a person based on an activity they participate in",
    "field of work that requires particular skills and knowledge of skilled work",
    "subclass of noble titles",
    "legal privilege given to some members in monarchial and princely societies",
    "titles given in an organization to show what duties and responsibilities a person has",
    "human relationship term; web of social relationships that form an important part of the lives of most humans in most societies; form of social connection",
    "level in a hierarchy",
    "profession in fictional stories",
    "part of a naming scheme for individuals, used in many cultures worldwide",
    "métaclasse d'ambassadeurs",
    "part of statements according to the Wikidata data model, appearing as the 2nd item in the statement triple",
    "title bestowed upon individuals or organizations as an award in recognition of their merits",
    "title to indicate the completion of a course of study or the extent of academic achievement",
    "enumeration value for a Wikidata property",
    "name of a employee's role assigned by their employer",
    "type of identity create by a type of religious belief",
    "joint arrangement of a team on its field of play and/or the standardized place of any individual player",
    "name for a resident of a locality",
    "character or part played by a performer",
    "Item with label/aliases for inverse relation of property. This helps the related items gadget - which you can enable in your wikidata preferences - to function.",
    "person who is paid to undertake a specialized set of tasks and to complete them for a fee",
    "words or grammatical forms that denote a positive affect",
    "... omitted 8 items"
  ],
  "report_violation_type_labels_en": [
    "profession",
    "position",
    "occupation",
    "craft",
    "hereditary title",
    "noble title",
    "corporate title",
    "kinship",
    "rank",
    "fictional profession",
    "family name",
    "class of ambassadors",
    "Wikidata property",
    "title of honor",
    "academic title",
    "Wikidata enumeration value",
    "job title",
    "religious identity",
    "position",
    "demonym",
    "role",
    "inverse property label item",
    "professional",
    "approbative",
    "... omitted 8 items"
  ],
  "report_violation_type_normalized": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
  "report_violation_type_qids": [
    "Q28640",
    "Q4164871",
    "Q12737077",
    "Q2207288",
    "Q5737899",
    "Q355567",
    "Q11488158",
    "Q171318",
    "Q4120621",
    "Q17305127",
    "Q101352",
    "Q29918287",
    "Q18616576",
    "Q3320743",
    "Q3529618",
    "Q21874278",
    "Q828803",
    "Q4392985",
    "Q1781513",
    "Q217438",
    "Q1707847",
    "Q65932995",
    "Q702269",
    "Q4781727",
    "... omitted 8 items"
  ],
  "report_violation_type_raw": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
  "value": null,
  "value_current_2026": [
    "Staatsalchimistin@lb"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "female form of name or title (for male use P3321)",
    "label": "female form of label"
  },
  "qid": {
    "description": "military rank in Fullmetal Alchemist",
    "label": "State Alchemist"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359"
    }
  ],
  "candidate_violation_names": [
    "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f0cc45546da8768eeffa00936c0e3a553c2aca4b",
  "hash_before": "42e4e2c51d366fbf19084c4ab0a0759e9306eafa",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359"
      }
    ],
    "candidate_violation_names": [
      "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359"
  }
]
```

---

## 020. `reform_Q4189957_P1588_2438259013`

| Field | Value |
|---|---|
| qid | Q4189957 |
| property | P1588 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P1588::2438259013 |
| tbox_revision_key | TBOX::P1588::2438259013 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Danil Satria",
  "kind": "T_BOX",
  "property_revision_id": 2438259013,
  "property_revision_prev": 2318558985
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T09:09:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1588",
  "report_revision_new": 2439930839,
  "report_revision_old": 2439522304,
  "report_violation_type": "Item P|625",
  "report_violation_type_normalized": "Item P|625",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|625",
  "value": null,
  "value_current_2026": [
    "1206071"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "unique code for a place, last issued in 2021 by Statistics Indonesia (Badan Pusat Statistik)",
    "label": "Statistics Indonesia area code"
  },
  "qid": {
    "description": "district in Toba Regency, North Sumatra Province, Indonesia",
    "label": "Pintu Pohan Meranti"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|625"
    }
  ],
  "candidate_violation_names": [
    "Item P|625"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q19474404"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|625",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q19474404"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|625",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Danil Satria",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "4d5f40959f2e6f6eddc3e6ab3532c653518b1f92",
  "hash_before": "51cc9c5f0280c618a493380dd2a11cbcde31a446",
  "property_revision_id": 2438259013,
  "property_revision_prev": 2318558985,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503247",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Item P|625"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|625"
      }
    ],
    "candidate_violation_names": [
      "Item P|625"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q19474404"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q19474404"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|625",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Item P|625"
  }
]
```

---

## 021. `reform_Q44734546_P8988_2445674626`

| Field | Value |
|---|---|
| qid | Q44734546 |
| property | P8988 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P8988::2445674626 |
| tbox_revision_key | TBOX::P8988::2445674626 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "JAn Dudík",
  "kind": "T_BOX",
  "property_revision_id": 2445674626,
  "property_revision_prev": 2437926724
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T06:56:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8988",
  "report_revision_new": 2445904045,
  "report_revision_old": 2445349914,
  "report_violation_type": "Item P|625",
  "report_violation_type_normalized": "Item P|625",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|625",
  "value": null,
  "value_current_2026": [
    "stre&id=82812"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a point of interest in Mapy.com",
    "label": "Mapy.com ID"
  },
  "qid": {
    "description": "street in Moravská Nová Ves, Czech Republic",
    "label": "Krátká"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|625"
    }
  ],
  "candidate_violation_names": [
    "Item P|625"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|625",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|625",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "JAn Dudík",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "698c87a03258d2c8497f4f2fe89ac50608e68d09",
  "hash_before": "24d513db5ec3bd62001f537388c44a29b482fdd7",
  "property_revision_id": 2445674626,
  "property_revision_prev": 2437926724,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503247",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Item P|625"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|625"
      }
    ],
    "candidate_violation_names": [
      "Item P|625"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21503250"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503250"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|625",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Item P|625"
  }
]
```

---

## 022. `reform_Q537575_P2521_1719625275`

| Field | Value |
|---|---|
| qid | Q537575 |
| property | P2521 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q53869507 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P2521::1719625275 |
| tbox_revision_key | TBOX::P2521::1719625275 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "HarryNº2",
  "kind": "T_BOX",
  "property_revision_id": 1719625275,
  "property_revision_prev": 1719579235
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-09-05T08:49:34",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2521",
  "report_revision_new": 1721422247,
  "report_revision_old": 1720685015,
  "report_violation_type": "Item P|3321",
  "report_violation_type_normalized": "Item P|3321",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|3321",
  "value": null,
  "value_current_2026": [
    "fossoyeuse@fr",
    "sepulturera@es",
    "Totengräberin@de",
    "coveira@pt",
    "могильщица@ru",
    "gropară@ro"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "female form of name or title (for male use P3321)",
    "label": "female form of label"
  },
  "qid": {
    "description": "cemetery worker responsible for digging a grave",
    "label": "gravedigger"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|3321"
    }
  ],
  "candidate_violation_names": [
    "Item P|3321"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|3321",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|3321",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "HarryNº2",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "42e4e2c51d366fbf19084c4ab0a0759e9306eafa",
  "hash_before": "cc14febcd9fad0d2148f06cbf373d4850baf5d97",
  "property_revision_id": 1719625275,
  "property_revision_prev": 1719579235,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503247",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Item P|3321"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|3321"
      }
    ],
    "candidate_violation_names": [
      "Item P|3321"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21502838"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502838"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|3321",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Item P|3321"
  }
]
```

---

## 023. `reform_Q6566190_P910_2444366220`

| Field | Value |
|---|---|
| qid | Q6566190 |
| property | P910 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q52060874 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | TBOX::P910::2444366220 |
| tbox_revision_key | TBOX::P910::2444366220 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2444366220,
  "property_revision_prev": 2430910127
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T09:21:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P910",
  "report_revision_new": 2444868799,
  "report_revision_old": 2444440134,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "Q8797333",
    "Q6948964"
  ],
  "value_current_2026_descriptions_en": [
    "Wikimedia category",
    "Wikimedia category"
  ],
  "value_current_2026_labels_en": [
    "Category:Sports portal",
    "Category:Sports and games portals"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "main Wikimedia category",
    "label": "topic's main category"
  },
  "qid": {
    "description": "Wikipedia portal for content related to Sports",
    "label": "Portal:Sports"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "inverse constraint",
    "qid": "Q21510855"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "value_or_property_overlap_only",
      "candidate_score": 20,
      "mapped_violation_constraint_qid": "Q19474404",
      "mapped_violation_family": "single_value",
      "violation_name": "Single value"
    }
  ],
  "candidate_violation_names": [
    "Single value"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q52060874"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52060874"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "single-value constraint",
  "mapped_report_constraint_qid": "Q19474404",
  "mapped_report_family": "single_value",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "single-value constraint",
  "mapped_violation_constraint_qid": "Q19474404",
  "mapped_violation_family": "single_value",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Single value",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q52060874"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52060874"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "single-value constraint",
  "mapped_report_constraint_qid": "Q19474404",
  "mapped_report_family": "single_value",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Single value",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Clemens Dulcis",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "c04c8b7bba53756644b83d744bd63d65b3af1b1a",
  "hash_before": "243303b03a05ec9a4d9d29506693772706b73942",
  "property_revision_id": 2444366220,
  "property_revision_prev": 2430910127,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q19474404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Single value"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "value_or_property_overlap_only",
        "candidate_score": 20,
        "mapped_violation_constraint_qid": "Q19474404",
        "mapped_violation_family": "single_value",
        "violation_name": "Single value"
      }
    ],
    "candidate_violation_names": [
      "Single value"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q52060874"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q52060874"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "single-value constraint",
    "mapped_report_constraint_qid": "Q19474404",
    "mapped_report_family": "single_value",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "single-value constraint",
    "mapped_violation_constraint_qid": "Q19474404",
    "mapped_violation_family": "single_value",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Single value",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Single value"
  }
]
```

---

## 024. `reform_Q68833_P828_1456089230`

| Field | Value |
|---|---|
| qid | Q68833 |
| property | P828 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510855 inverse constraint |
| group_key | TBOX::P828::1456089230 |
| tbox_revision_key | TBOX::P828::1456089230 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "inverse constraint",
  "decision_constraint_type_qid": "Q21510855"
}
```

#### Repair Target

```json
{
  "author": "Azertus",
  "kind": "T_BOX",
  "property_revision_id": 1456089230,
  "property_revision_prev": 1456089162
}
```

### Violation Context

```json
{
  "report_fix_date": "2021-07-12T22:10:18",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P828",
  "report_revision_new": 1458933058,
  "report_revision_old": 851643686,
  "report_violation_type": "Inverse",
  "report_violation_type_normalized": "Inverse",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Inverse",
  "value": null,
  "value_current_2026": [
    "Q41392",
    "Q193078"
  ],
  "value_current_2026_descriptions_en": [
    "fatigue-induced fracture of the bone caused by repeated stress over time",
    "physiological wound caused by an external source"
  ],
  "value_current_2026_labels_en": [
    "stress fracture",
    "injury"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "underlying cause, entity that ultimately resulted in this effect",
    "label": "has cause"
  },
  "qid": {
    "description": "medical condition in which there is physical damage to the continuity of the bones",
    "label": "bone fracture"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510855",
      "mapped_violation_family": "inverse",
      "violation_name": "Inverse"
    }
  ],
  "candidate_violation_names": [
    "Inverse"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "inverse constraint",
  "mapped_report_constraint_qid": "Q21510855",
  "mapped_report_family": "inverse",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "inverse constraint",
  "mapped_violation_constraint_qid": "Q21510855",
  "mapped_violation_family": "inverse",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Inverse",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "inverse constraint",
  "mapped_report_constraint_qid": "Q21510855",
  "mapped_report_family": "inverse",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Inverse",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Azertus",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "decc1b77a57a7346eede7b4580873f5838aad75e",
  "hash_before": "9085b5e9fe35369430c9508d4f5414cc7ca466e9",
  "property_revision_id": 1456089230,
  "property_revision_prev": 1456089162,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510855",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Inverse"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510855",
        "mapped_violation_family": "inverse",
        "violation_name": "Inverse"
      }
    ],
    "candidate_violation_names": [
      "Inverse"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q53869507"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q53869507"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "inverse constraint",
    "mapped_report_constraint_qid": "Q21510855",
    "mapped_report_family": "inverse",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "inverse constraint",
    "mapped_violation_constraint_qid": "Q21510855",
    "mapped_violation_family": "inverse",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Inverse",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Inverse"
  }
]
```

---

## 025. `reform_Q7211483_P1753_2444818722`

| Field | Value |
|---|---|
| qid | Q7211483 |
| property | P1753 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21510855 inverse constraint |
| group_key | TBOX::P1753::2444818722 |
| tbox_revision_key | TBOX::P1753::2444818722 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "inverse constraint",
  "decision_constraint_type_qid": "Q21510855"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2444818722,
  "property_revision_prev": 2422397565
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T10:54:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1753",
  "report_revision_new": 2445982077,
  "report_revision_old": 2445412111,
  "report_violation_type": "Inverse",
  "report_violation_type_normalized": "Inverse",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Inverse",
  "value": null,
  "value_current_2026": [
    "Q108679834"
  ],
  "value_current_2026_descriptions_en": [
    "List of Mexican secretaries of Finance and Public Credit"
  ],
  "value_current_2026_labels_en": [
    "list of Secretaries of Finance and Public Credit"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "Wikimedia list equivalent to a Wikimedia category",
    "label": "list related to category"
  },
  "qid": {
    "description": "Wikimedia category",
    "label": "Category:Secretaries of finance of Mexico"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "inverse constraint",
    "qid": "Q21510855"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510855",
      "mapped_violation_family": "inverse",
      "violation_name": "Inverse"
    }
  ],
  "candidate_violation_names": [
    "Inverse"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q19474404"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "inverse constraint",
  "mapped_report_constraint_qid": "Q21510855",
  "mapped_report_family": "inverse",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "inverse constraint",
  "mapped_violation_constraint_qid": "Q21510855",
  "mapped_violation_family": "inverse",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Inverse",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q19474404"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "inverse constraint",
  "mapped_report_constraint_qid": "Q21510855",
  "mapped_report_family": "inverse",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Inverse",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Clemens Dulcis",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "c3d24c35bf0405e95e1e61dc12f57b8146a87504",
  "hash_before": "eca87051cb54f66af4b9f86e887a2ea2711125e5",
  "property_revision_id": 2444818722,
  "property_revision_prev": 2422397565,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21510855",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Inverse"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510855",
        "mapped_violation_family": "inverse",
        "violation_name": "Inverse"
      }
    ],
    "candidate_violation_names": [
      "Inverse"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q19474404"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q19474404"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "inverse constraint",
    "mapped_report_constraint_qid": "Q21510855",
    "mapped_report_family": "inverse",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "inverse constraint",
    "mapped_violation_constraint_qid": "Q21510855",
    "mapped_violation_family": "inverse",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Inverse",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Inverse"
  }
]
```

---

## 026. `reform_Q76386047_P957_1774652896`

| Field | Value |
|---|---|
| qid | Q76386047 |
| property | P957 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21502410 distinct-values constraint |
| group_key | TBOX::P957::1774652896 |
| tbox_revision_key | TBOX::P957::1774652896 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "distinct-values constraint",
  "decision_constraint_type_qid": "Q21502410"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1774652896,
  "property_revision_prev": 1774652608
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-11-19T09:46:39",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P957",
  "report_revision_new": 1774857602,
  "report_revision_old": 1774139072,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "984-8682-62-7"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "former identifier for a book (edition), ten digits. Used for all publications up to 2006 (convertible to ISBN-13 for some online catalogs; useful for old books or fac-similes not reedited since 2007)",
    "label": "ISBN-10"
  },
  "qid": {
    "description": "ফেব্রুয়ারি ২০০৪ সালে মুদ্রিত হুমায়ূন আহমেদের উপন্যাসের সংস্করণ",
    "label": "যদিও সন্ধ্যা"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21502410",
      "mapped_violation_family": "distinct_values",
      "violation_name": "Unique value"
    }
  ],
  "candidate_violation_names": [
    "Unique value"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "distinct-values constraint",
  "mapped_violation_constraint_qid": "Q21502410",
  "mapped_violation_family": "distinct_values",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Unique value",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Unique value",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "9aba6940dbcd0fb102409e100e7e40ecbfc6b531",
  "hash_before": "91ecbfa2476203c7a7244a62197c0de673ffc775",
  "property_revision_id": 1774652896,
  "property_revision_prev": 1774652608,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21502410",
        "mapped_violation_family": "distinct_values",
        "violation_name": "Unique value"
      }
    ],
    "candidate_violation_names": [
      "Unique value"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q53869507"
    ],
    "changed_constraint_qids_from_entries": [
      "Q53869507"
    ],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q53869507"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "distinct-values constraint",
    "mapped_report_constraint_qid": "Q21502410",
    "mapped_report_family": "distinct_values",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "distinct-values constraint",
    "mapped_violation_constraint_qid": "Q21502410",
    "mapped_violation_family": "distinct_values",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Unique value",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Unique value"
  }
]
```

---

## 027. `reform_Q871819_P2659_1488142524`

| Field | Value |
|---|---|
| qid | Q871819 |
| property | P2659 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | coincidental_schema_change |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P2659::1488142524 |
| tbox_revision_key | TBOX::P2659::1488142524 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "coincidental_schema_change",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "SilentSpike",
  "kind": "T_BOX",
  "property_revision_id": 1488142524,
  "property_revision_prev": 1488139479
}
```

### Violation Context

```json
{
  "report_fix_date": "2021-08-29T07:26:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2659",
  "report_revision_new": 1489359691,
  "report_revision_old": 1484026669,
  "report_violation_type": "Item P|3137",
  "report_violation_type_normalized": "Item P|3137",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|3137",
  "value": null,
  "value_current_2026": [
    "+18.9 http://www.wikidata.org/entity/Q828224"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "minimum distance to a point of higher elevation",
    "label": "topographic isolation"
  },
  "qid": {
    "description": "mountain in the Seckauer Tauern in Styria",
    "label": "Geierhaupt"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "allowed units constraint",
    "qid": "Q21514353"
  },
  {
    "label_en": "range constraint",
    "qid": "Q21510860"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|3137"
    }
  ],
  "candidate_violation_names": [
    "Item P|3137"
  ],
  "causality_match_level": "constraint_family_mismatch",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21510860"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510860"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|3137",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510860"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510860"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|3137",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "SilentSpike",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f4954230753ca7cbcd77c53e0d3e38a9d5b11da9",
  "hash_before": "3f34c4dd83b0c3f364388cf5252efa6a3f03ffb0",
  "property_revision_id": 1488142524,
  "property_revision_prev": 1488139479,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "constraint_family_mismatch",
    "mapped_violation_constraint_qid": "Q21503247",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Item P|3137"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|3137"
      }
    ],
    "candidate_violation_names": [
      "Item P|3137"
    ],
    "causality_match_level": "constraint_family_mismatch",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21510860"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510860"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "result": "COINCIDENTAL_SCHEMA_CHANGE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|3137",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Item P|3137"
  }
]
```

---
