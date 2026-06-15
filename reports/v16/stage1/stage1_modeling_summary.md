# Stage 1 modeling summary

- **stage**: `Stage 1 — Neo4j GDS/tabular baseline`
- **model**: `stage1_tabular_extra_trees`
- **status**: `skipped`
- **skip_reason**: `no_leakage_safe_features_found`
- **stage_dir**: `A:\Repositories\PRING\runs\cyp450_5enzymes_uncapped_raw_rematerialized\graph\ml\modeling\stage1_neo4j_gds_baselines`
- **training_file**: `A:\Repositories\PRING\runs\cyp450_5enzymes_uncapped_raw_rematerialized\graph\ml\modeling\stage1_neo4j_gds_baselines\compound_target_training_pairs_for_gds.csv`
- **candidate_file**: `A:\Repositories\PRING\runs\cyp450_5enzymes_uncapped_raw_rematerialized\graph\ml\modeling\stage1_neo4j_gds_baselines\candidate_pairs_for_gds_scoring.csv`
- **feature_policy**: `leakage_safe`
- **leakage_warning**: `The current Stage 1 export appears to contain only evidence/outcome-derived columns. Re-export FastRP/GraphSAGE/topological features from Neo4j GDS for a valid structural baseline.`
- **excluded_feature_columns**: `['label', 'active_endpoint_count', 'ambiguous_endpoint_count', 'assay_count', 'best_negative_log10_molar', 'best_value_molar', 'best_value_um', 'bindingdb_best_affinity_type', 'bindingdb_best_affinity_value', 'bindingdb_has_record', 'bindingdb_min_ic50_nm', 'bindingdb_min_kd_nm', 'bindingdb_min_ki_nm', 'bindingdb_record_count', 'compound_node_id', 'compound_node_ref', 'endpoint_type_counts', 'evidence_assays', 'evidence_count', 'evidence_endpoints', 'evidence_measuregroups', 'evidence_references', 'ic50_endpoint_count', 'inactive_endpoint_count', 'kd_endpoint_count', 'ki_endpoint_count', 'label_rule', 'min_ic50_molar', 'min_kd_molar', 'min_ki_molar', 'negative_endpoint_count', 'negative_source', 'positive_endpoint_count', 'protein_node_id', 'protein_node_ref', 'reference_count', 'split', 'split_group', 'split_strategy', 'stage_use', 'target_relation', 'textmine_confidence', 'textmine_confidence_score', 'textmine_cooc_count', 'textmine_reference_count', 'textmine_score_max', 'textmine_score_mean', 'weak_endpoint_count', '_label']`

## Metrics
