# Confidence models `v1`

Feature schema v1 (QualitySignals v1); logistic regressions, JSON weights.
Fitted 2026-09-18 on val scenes tless [1, 6, 11, 16], xyzibd [0, 20, 40, 60].

- **Model H** (per refined PoseHypothesis): 1901 fit rows, C = 0.1; eval ROC-AUC 91.1, Brier 0.127, ECE 8.1 %.
- **Model F** (per FusedPose): 5638 fit rows, C = 0.03; eval ROC-AUC 92.4, Brier 0.112, ECE 4.5 %.
- **Verdict**: accept ≥ 0.823, reject ≤ 0.675 (targets 0.95 / 0.90).

Success = MSSD < 0.1 · diameter against the nearest same-object ground truth. Signals come from CNOS + FoundPose + point-to-plane ICP on T-LESS test_primesense and XYZ-IBD val; a different segmenter, estimator or refiner changes the signal distributions and needs a refit.

Files: model_h.json, model_f.json, thresholds.json; analysis in outputs/confidence/<tag>/report.md.
