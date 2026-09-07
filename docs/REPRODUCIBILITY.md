# Reproducibility Guide

## Scientific boundary

This package reproduces the UNSW-NB15 external replication/portability study used in the journal manuscript. It is independent from the core thesis experiments and must not be described as a direct no-retraining transfer of a CSE-CIC-IDS2018 model.

## Execution order

Run stages in numerical order:

1. `stage01_data_audit.py` — audit raw files, construct strict partitions, and freeze candidate feature contracts.
2. `stage02_feature_contract_ablation.py` — Validation-only feature-contract comparison.
3. `stage03_stability_and_frozen_test.py` — five-seed stability, freeze canonical model, one strict Test evaluation.
4. `stage04_shap_faithfulness.py` — exact TreeSHAP faithfulness on 800 correctly detected Attack events.
5. `stage05_onnx_validation.py` — export the frozen detector to ONNX and verify numerical/decision equivalence.
6. `stage06_grounded_fast_path.py` — exact full-forest grounded explanation path and strict validator.
7. `stage07_validation_shap_acceleration.py` — Validation-only approximate TreeSHAP acceleration candidate.
8. `stage07b_validation_tree_subset_surrogate.py` — Validation-only TSES candidate development and selection.
9. `stage08_surrogate_confirmatory_test.py` — confirmatory Test evaluation of the frozen TSES-20 explanation surrogate.
10. `stage09_test_error_analysis.py` — descriptive post-hoc Test error analysis only.
11. `stage10_paper_ready_summary.py` — consolidate manuscript-ready results.

## Environment

Recommended Python: 3.12.x. Install the pinned packages with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Frozen selection rules

- Test labels were not used for feature-contract selection, seed selection, detector training decisions, threshold selection, or TSES-20 selection.
- Canonical detector seed: `20260907`.
- Detector threshold: `0.5`.
- Frozen detector feature contract: `NUMERIC_ONLY_39`.
- TSES-20 was selected on Validation and is explanation-only.
- The Test evaluation of TSES-20 is confirmatory because the Test partition had already been accessed in the preceding frozen detector and explanation analyses.

## Runtime interpretation

Latency values are post-flow and hardware/runtime specific. They exclude packet capture, flow construction, network transport, queueing, external logging, and any generative fallback.

The exact full-forest grounded path is reported even though it fails the 50-ms target. The TSES-20 result is reported as a distinct validation-selected acceleration path; it is not exact full-forest TreeSHAP.

## Integrity

Key SHA-256 identifiers are recorded in the manifests. The frozen Random Forest model hash is:

`d0122bb1501c17d55934b9ed5484b03f03879af8fead19007e6fb428c70d2f05`

The ONNX detector hash is:

`ef468176f0ed38c970f58a593826c6d2cc8fd038d51125cedc56044d433327ad`

The TSES-20 explanation surrogate hash is:

`0d4c2bb1d9007dbdf155b15ecd0726c04af0164446493336ac058155b1f6e6f4`

Use `manifests/` as the authoritative source if a compact summary conflicts with a secondary table or narrative.
