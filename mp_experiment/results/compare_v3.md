# MediaPipe v3 (mirror-augmented) vs v2 -- TEST split comparison

The mirrored test is synthetic: it simulates the opposite hand by negating x. It is not a substitute for real left-hand data.

- Detected test samples: 539; the mirrored test is the same set with x negated (counts identical by construction).
- Train 2444 -> 4888 after mirror augmentation, val 518 -> 1036; test never used for training or selection.
- Same recipe as train_mp_v2.py: RandomForest(n_estimators=300, random_state=42, n_jobs=-1) and MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800, random_state=42); winner = higher val accuracy, MLP wins ties.
- v2: kind=mlp, val_accuracy=0.9633; v3: kind=mlp, val_accuracy=0.9653 (v3 val = original + mirrored).
- Sanity check: 5 training samples mirrored twice == original: **OK**.

## Four accuracy combinations (detected test)

| model | test | accuracy | correct/n |
|---|---|---|---|
| v2 | original test | 0.9722 | 524/539 |
| v2 | mirrored test | 0.1169 | 63/539 |
| v3 | original test | 0.9666 | 521/539 |
| v3 | mirrored test | 0.9722 | 524/539 |

- v3 - v2 on the ORIGINAL test: **-0.0056**

## Per-class accuracy on the MIRRORED test

| class | n | v2 | v3 |
|---|---|---|---|
| A | 21 | 0.0000 | 0.9524 |
| B | 16 | 0.0000 | 1.0000 |
| C | 18 | 0.0000 | 0.9444 |
| D | 20 | 0.0000 | 1.0000 |
| E | 20 | 0.0000 | 1.0000 |
| F | 22 | 0.0000 | 1.0000 |
| G | 19 | 0.0000 | 1.0000 |
| H | 19 | 0.0000 | 0.9474 |
| I | 18 | 1.0000 | 1.0000 |
| J | 22 | 0.0000 | 1.0000 |
| K | 21 | 0.0000 | 1.0000 |
| L | 22 | 0.0000 | 1.0000 |
| M | 18 | 0.7778 | 0.9444 |
| N | 13 | 0.4615 | 0.7692 |
| O | 20 | 0.0000 | 1.0000 |
| P | 15 | 0.0667 | 0.8667 |
| Q | 17 | 0.0000 | 1.0000 |
| R | 20 | 0.0000 | 0.9500 |
| S | 19 | 0.0000 | 0.9474 |
| T | 21 | 0.0000 | 1.0000 |
| U | 21 | 0.9048 | 0.9524 |
| V | 21 | 0.0000 | 1.0000 |
| W | 19 | 0.0000 | 1.0000 |
| X | 20 | 0.0000 | 0.9500 |
| Y | 22 | 0.2273 | 0.9545 |
| Z | 22 | 0.0000 | 1.0000 |
| del | 16 | 0.0000 | 1.0000 |
| space | 17 | 0.0000 | 0.9412 |

## Top 10 confused letter pairs (v3, original test; A-Z, unordered)

- M <-> N: 2
- P <-> Q: 2
- A <-> O: 1
- B <-> N: 1
- C <-> D: 1
- D <-> Y: 1
- H <-> J: 1
- H <-> P: 1
- J <-> X: 1
- J <-> Y: 1

## ponytail

- ceiling: the mirror is a synthetic proxy for the other hand (x negated only). Upgrade path: real left-hand data via collect_landmarks.py.

## Reproduce

```
python -m py_compile train_mp_v3.py
python train_mp_v3.py --dir E:/ASL/mp_experiment
```
