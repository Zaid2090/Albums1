# Data Boundary and Partition Audit

## Dataset

The external replication uses **UNSW-NB15**. Raw data are not redistributed here.

Official dataset page recorded by the study:

`https://research.unsw.edu.au/projects/unsw-nb15-dataset`

A public mirror was used for retrieval. The mirror stored the two predefined partition contents under reversed filenames, so semantic assignment was determined from the official published row counts and verified class distributions.

## Retrieved files

- Mirror `UNSW_NB15_testing-set.csv`: 175,341 rows, used as the semantic development/training partition.
- Mirror `UNSW_NB15_training-set.csv`: 82,332 rows, used as the semantic official Test partition.

The original downloads were preserved unchanged and their SHA-256 hashes are recorded in `manifests/dataset_retrieval_and_partition_audit_v1.json`.

## Leakage-control audit

Predictor-identical groups were identified without using the row identifier. The audit found:

- 1,302 predictor-overlap groups across the original development/Test boundary.
- 8,541 official Test rows involved in cross-boundary predictor overlap.
- These 8,541 rows were removed from the strict Test evaluation.
- Strict Test size: **73,791 rows**.
- Predictor-group overlap after the strict construction: **0** across development/Validation/Test.

The unpurged official Test result is also retained descriptively so the strict purge cannot be interpreted as a performance-enhancing hidden selection step.
