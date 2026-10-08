# Detection sweep (TEST split only)

Baseline = (0.5, none), matching `features_test.npz` (488/653 = 74.7%).

`new` = images detected under this setting but NOT in the baseline; `model acc on new` = existing `mp_model_mlp.joblib` accuracy on those images (checks the newly detected landmarks aren't garbage).

| min_det_conf | preprocessing | detected | rate | new vs baseline | model acc on new (ok/wrong) | unreadable |
|---|---|---|---|---|---|---|
| 0.5 | none | 488/653 | 74.7% | 0 | n/a (no new detections) | 0 |
| 0.5 | pad25 | 462/653 | 70.8% | 33 | 0.9394 (31/33) | 0 |
| 0.5 | up2x | 467/653 | 71.5% | 3 | 0.3333 (1/3) | 0 |
| 0.5 | pad25_up2x | 452/653 | 69.2% | 28 | 0.8929 (25/28) | 0 |
| 0.3 | none | 505/653 | 77.3% | 17 | 0.7059 (12/17) | 0 |
| 0.3 | pad25 | 488/653 | 74.7% | 39 | 0.9487 (37/39) | 0 |
| 0.3 | up2x | 485/653 | 74.3% | 8 | 0.3750 (3/8) | 0 |
| 0.3 | pad25_up2x | 473/653 | 72.4% | 35 | 0.8857 (31/35) | 0 |
| 0.1 | none | 518/653 | 79.3% | 30 | 0.8667 (26/30) | 0 |
| 0.1 | pad25 | 515/653 | 78.9% | 51 | 0.8627 (44/51) | 0 |
| 0.1 | up2x | 499/653 | 76.4% | 21 | 0.6667 (14/21) | 0 |
| 0.1 | pad25_up2x | 496/653 | 76.0% | 42 | 0.8810 (37/42) | 0 |

Runtime: 138.2s. Model: `mp_model_mlp.joblib` (val acc 0.9700214132762313) — retrained nothing; only inference.
