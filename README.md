# EGCP External Validation on UNSW-NB15

Reproducibility package for the independent UNSW-NB15 replication and explanation-latency study accompanying the journal manuscript on the **Evidence-Governed Cascade Protocol (EGCP)**.

## Scope

This repository documents an independent benchmark replication on UNSW-NB15. It does **not** claim direct cross-dataset transfer of the CSE-CIC-IDS2018-trained model. The UNSW-NB15 detector was trained and frozen using UNSW-NB15 development data under a leakage-controlled protocol, then evaluated on a strict duplicate-purged Test partition.

The package focuses on four questions:

1. How well does the frozen detector generalize on a strict UNSW-NB15 Test partition?
2. Are TreeSHAP explanations locally faithful to the frozen detector?
3. Does exact full-forest TreeSHAP satisfy a sub-50-ms post-flow explanation budget?
4. Can an explanation-only tree-subset surrogate recover the latency target while preserving high evidence fidelity?

## Key confirmatory results

- Strict Test rows: **73,791**
- Accuracy: **89.6112%**
- F1: **90.6669%**
- Recall: **97.3159%**
- ROC-AUC: **98.1313%**
- Exact TreeSHAP top-5 probability drop: **0.24784** vs random **0.05991**
- ONNX detector decision agreement: **100%**
- ONNX detector-only P95: **0.0572 ms**
- Exact full-forest grounded path P95: **258.925 ms**
- TSES-20 confirmatory grounded path P95: **36.457 ms**
- TSES-20 Top-2 / Top-5 overlap with exact full-forest evidence: **95.06% / 87.40%**
- Strict explanation validity: **100%**

## TSES-20

**TSES-20 (Tree-Subset Explanation Surrogate)** is an explanation-only surrogate formed from 20 deterministically spaced trees taken from the frozen 150-tree Random Forest. The full 150-tree detector remains authoritative for probability, class, threshold, and detection metrics. TSES-20 is used only to compute local explanation evidence.

The surrogate was selected on Validation before its confirmatory Test evaluation. Its Test evaluation is explicitly marked **confirmatory after prior Test access**, not a new blind Test.

## Repository structure

- `src/` — executable analysis stages and grounded renderer/validator.
- `manifests/` — frozen study decisions, hashes, selection boundaries, and final evaluation manifests.
- `results/` — compact numerical outputs used to support manuscript claims.
- `figures/` — publication-supporting figures generated from the recorded results.
- `docs/` — reproducibility, data-boundary, and claim-to-artifact documentation.
- `requirements.txt` — recorded Python package versions.

## Data availability

Raw UNSW-NB15 data are **not redistributed** in this repository. Obtain the dataset from the official UNSW-NB15 source. The retrieval audit records the exact public files, row counts, SHA-256 hashes, and semantic partition mapping used in this study.

## Reproducibility boundary

Large trained model binaries and raw datasets are intentionally excluded from the public repository. Their SHA-256 identifiers and full model configuration are preserved in the manifests. The scripts document how the detector, ONNX export, explanation analyses, TSES-20 surrogate, latency measurements, and error analysis were produced.

See `docs/REPRODUCIBILITY.md` and `docs/CLAIM_ARTIFACT_MAP.md` before reproducing or citing individual claims.

## Citation

See `CITATION.cff`. A Zenodo DOI will be added to the citation metadata after the archival release is minted.

## License

Code and documentation are released under the MIT License. Dataset licensing and terms remain governed by the original UNSW-NB15 provider.