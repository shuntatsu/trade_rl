# Serving Critical Coverage Recovery

The first complete exact-head run after adding `PolicyObservationSnapshot` passed all 1,026 tests and total branch coverage (83.28%), but the existing Serving critical branch-coverage group fell to 88.60% against its unchanged 90.00% threshold.

The threshold was not lowered. Five focused fail-closed cases were added for the new snapshot-serving boundary:

- inactive runtime without an active policy;
- normalized-observation mismatch;
- a bundle path without a normalizer;
- a dataset-native policy implementing `predict_from_dataset`;
- an invalid dataset-native action outside the action schema.

The focused formatting and test workflow `29817272033` passed and removed itself before publishing product commit `e637948bbc8940c7442ff6e9b530e999dfd4a490`. The final exact-head CI run must confirm that the Serving group is again at or above 90.00% together with the complete repository suite.
