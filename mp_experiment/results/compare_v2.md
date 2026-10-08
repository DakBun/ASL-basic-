# MediaPipe v2 (cascade) — TEST split report

Cascade: stage 1 = HandLandmarker conf 0.5, no preprocessing; stage 2 = conf 0.1 + 25% gray pad (only on stage-1 misses). Model `mp_model_v2.joblib` trained on detected train samples only; nothing retrained beyond that.

- Test size: 653; detected: 539 (82.5%); unreadable counted as undetected.
- Coverage overall: **82.5%**; excluding `nothing`: **85.4%** (539/631).
- Detected by stage: stage 1 = 488, stage 2 = 51.

## Per-class coverage (detected / total)

| class | detected / total | rate |
|---|---|---|
| A | 21/22 | 95.5% |
| B | 16/22 | 72.7% |
| C | 18/23 | 78.3% |
| D | 20/23 | 87.0% |
| E | 20/23 | 87.0% |
| F | 22/23 | 95.7% |
| G | 19/23 | 82.6% |
| H | 19/23 | 82.6% |
| I | 18/22 | 81.8% |
| J | 22/22 | 100.0% |
| K | 21/23 | 91.3% |
| L | 22/22 | 100.0% |
| M | 18/22 | 81.8% |
| N | 13/22 | 59.1% |
| O | 20/23 | 87.0% |
| P | 15/22 | 68.2% |
| Q | 17/22 | 77.3% |
| R | 20/22 | 90.9% |
| S | 19/23 | 82.6% |
| T | 21/23 | 91.3% |
| U | 21/23 | 91.3% |
| V | 21/22 | 95.5% |
| W | 19/22 | 86.4% |
| X | 20/23 | 87.0% |
| Y | 22/23 | 95.7% |
| Z | 22/23 | 95.7% |
| del | 16/22 | 72.7% |
| nothing | 0/22 | 0.0% |
| space | 17/23 | 73.9% |

## Accuracy on detected-only samples

- Overall: **0.9722** (539 samples, 0.14 ms/sample classifier-only)
- Stage 1 only: **0.9795** (488 samples)
- Stage 2 only: **0.9020** (51 samples)

## Per-class F1 (detected-only)

| class | F1 |
|---|---|
| A | 0.9756 |
| B | 0.9697 |
| C | 0.9714 |
| D | 0.9744 |
| E | 0.9756 |
| F | 0.9778 |
| G | 1.0000 |
| H | 0.9744 |
| I | 1.0000 |
| J | 1.0000 |
| K | 0.9767 |
| L | 1.0000 |
| M | 0.9412 |
| N | 0.8462 |
| O | 0.9500 |
| P | 0.9655 |
| Q | 0.9444 |
| R | 0.9744 |
| S | 0.9730 |
| T | 0.9767 |
| U | 0.9767 |
| V | 1.0000 |
| W | 1.0000 |
| X | 0.9744 |
| Y | 0.9333 |
| Z | 1.0000 |
| del | 0.9677 |
| nothing | 0.0000 |
| space | 0.9412 |

## Top 10 confused LETTER pairs (detected-only; `nothing`, `del`, `space` excluded)

- M <-> N: 2
- X <-> Y: 1
- T <-> Y: 1
- R <-> U: 1
- N <-> Q: 1
- H <-> P: 1
- F <-> O: 1
- E <-> S: 1
- D <-> Y: 1
- C <-> O: 1

### The three non-letter classes, separately

- **nothing** (n=0 detected): top confused with -> none
- **del** (n=16 detected): top confused with -> Q: 1
- **space** (n=17 detected): top confused with -> K: 1, A: 1

## Policies for undetected images

**(A) undetected -> "nothing":** overall accuracy = **0.8361**; `nothing` recall = **1.0000**, `nothing` precision = **0.1930**.

**(B) undetected = abstain:** abstain rate = **17.5%** (114/653); accuracy on answered samples = **0.9722**.

## J and Z only (detected-only accuracy)

- **J**: 22/22 (1.0000)
- **Z**: 22/22 (1.0000)

**Note:** J and Z are dynamic letters (motion), but this dataset provides single static frames only, so these numbers do not measure motion recognition.

## Leakage check (nearest-train distance)

- Reference set: 2444 detected TRAIN samples; query: 539 detected TEST samples; exact brute-force Euclidean NN in the normalized 63-d feature space.
- Median nearest-train distance: **0.3517** (0 = identical landmark geometry).
- Accuracy on nearest 25% (134 samples): **1.0000**
- Accuracy on farthest 25% (134 samples): **0.8955**
- Gap (near - far): **+0.1045**

**How to read this:** Kaggle ASL Alphabet is burst-shot video, so consecutive frames per subject are near-duplicates; the stratified split puts such siblings into both train and test. A large positive gap (near >> far) means the score is inflated by near-duplicate frames leaking across the split, and true generalization to new hands/positions is closer to the far-quartile number. A small gap suggests the model generalizes rather than memorizes neighbors.
