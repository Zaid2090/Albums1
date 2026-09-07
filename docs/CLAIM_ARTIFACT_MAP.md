# Claim-to-Artifact Map

This map identifies the primary artifact supporting each manuscript claim.

| Claim | Primary artifact(s) |
|---|---|
| Strict Test contains 73,791 rows after overlap purge | `manifests/data_audit_summary_v1.json`, `manifests/strict_split_manifest_v1.json` |
| Frozen detector uses 39 numeric features and seed 20260907 | `manifests/frozen_detector_v1.json` |
| Validation-only feature contract selection | `manifests/feature_contract_selection_v1.json` |
| Five-seed stability without best-seed selection | `manifests/five_seed_stability_v1.json`, `results/five_seed_validation_metrics_v1.csv` |
| Strict Test Accuracy 89.6112%, F1 90.6669%, Recall 97.3159% | `manifests/final_frozen_test_v1.json`, `results/final_frozen_test_metrics_v1.csv` |
| 95% bootstrap uncertainty for strict Test metrics | `results/strict_test_bootstrap_ci_v1.json` |
| Exact TreeSHAP top-5 drop 0.24784 vs random 0.05991 | `manifests/shap_faithfulness_v1.json`, `results/shap_faithfulness_summary_v1.json` |
| Exact full-forest grounded path P95 258.925 ms | `manifests/grounded_fast_path_v1.json`, `results/grounded_fast_path_latency_summary_v1.csv` |
| Strict validator accepts valid outputs and rejects 732 deliberate corruptions | `manifests/grounded_fast_path_v1.json`, `results/strict_validator_mutation_stress_v1.csv` |
| ONNX decision agreement 100% and detector P95 0.0572 ms | `manifests/onnx_validation_v1.json`, `results/onnx_detector_latency_summary_v1.csv` |
| Approximate TreeSHAP rejected because Top-2 overlap was 21% | `manifests/shap_acceleration_validation_v1.json`, `manifests/explanation_acceleration_selection_v1.json` |
| TSES-20 selected on Validation | `manifests/tree_subset_surrogate_validation_v1.json`, `manifests/explanation_acceleration_selection_v1.json` |
| TSES-20 confirmatory Test Top-2 95.06%, Top-5 87.40%, E2E P95 36.457 ms | `manifests/tree_subset_surrogate_confirmatory_test_v1.json`, `results/surrogate_fast_path_latency_summary_test_v1.csv` |
| Fuzzers account for the majority of false negatives | `manifests/test_error_analysis_v1.json`, `results/test_attack_category_recall_v1.csv`, `results/false_negative_attack_cat_top20_v1.csv` |

## Important claim boundary

TSES-20 is an **explanation-only surrogate**. It does not replace the 150-tree detector for class or probability. Its confirmatory Test evaluation occurred after the Test partition had already been accessed, and therefore must not be described as a fresh blind Test.
