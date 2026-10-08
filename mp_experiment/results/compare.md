# ResNet50 vs MediaPipe comparison (FULL test split)

**Caution / optimism note:** ResNet50 is evaluated through the hosted API on the same Kaggle images the project was trained from on Roboflow, so the hosted model may have seen these test images during training. **ResNet50 numbers may be optimistic** and are not a clean held-out estimate; MediaPipe features are extracted locally and never uploaded, so its numbers are unaffected by that risk.

- Test size: 653 images, 29 classes (stratified split, seed 42).
- MP model: `mp_model_mlp.joblib`; no-hand -> "nothing".
- ResNet50: `sign-language-2c6kk/2` via `classify.roboflow.com` (server-side preprocessing).
- MediaPipe detection rate on test: 74.7% (488/653).
- ResNet50 API rows with error/missing: 0 (counted as misses).

| model | overall accuracy | macro F1 | inference ms/image |
|---|---|---|---|
| ResNet50 (hosted) | 1.0000 | 1.0000 | 305.1 (incl. network) |
| MediaPipe + mp_model_mlp.joblib | 0.7550 | 0.8089 | 0.10 (classifier only, excl. landmark extraction) |

## Per-class F1

| class | ResNet50 F1 | MediaPipe F1 |
|---|---|---|
| A | 1.0000 | 0.9000 |
| B | 1.0000 | 0.8421 |
| C | 1.0000 | 0.8500 |
| D | 1.0000 | 0.8205 |
| E | 1.0000 | 0.8571 |
| F | 1.0000 | 0.9565 |
| G | 1.0000 | 0.8000 |
| H | 1.0000 | 0.8500 |
| I | 1.0000 | 0.9000 |
| J | 1.0000 | 0.8780 |
| K | 1.0000 | 0.9333 |
| L | 1.0000 | 0.9767 |
| M | 1.0000 | 0.6061 |
| N | 1.0000 | 0.4138 |
| O | 1.0000 | 0.8444 |
| P | 1.0000 | 0.7429 |
| Q | 1.0000 | 0.7895 |
| R | 1.0000 | 0.8293 |
| S | 1.0000 | 0.8780 |
| T | 1.0000 | 0.9048 |
| U | 1.0000 | 0.8372 |
| V | 1.0000 | 0.9767 |
| W | 1.0000 | 0.9000 |
| X | 1.0000 | 0.8500 |
| Y | 1.0000 | 0.9333 |
| Z | 1.0000 | 0.9091 |
| del | 1.0000 | 0.7778 |
| nothing | 1.0000 | 0.2353 |
| space | 1.0000 | 0.4667 |

## Top 10 most confused class pairs (both directions summed)

**ResNet50:**

**MediaPipe:**
- N <-> nothing: 15
- nothing <-> space: 14
- M <-> nothing: 11
- P <-> nothing: 9
- del <-> nothing: 7
- Q <-> nothing: 7
- G <-> nothing: 7
- B <-> nothing: 6
- X <-> nothing: 5
- H <-> nothing: 5

## J and Z only (static frames)

- **J**: ResNet50 22/22 (1.0000); MediaPipe 18/22 (0.8182)
- **Z**: ResNet50 23/23 (1.0000); MediaPipe 20/23 (0.8696)

**Note:** J and Z are dynamic letters (motion), but this dataset provides single static frames only, so these numbers do not measure motion recognition. They only reflect how distinct the static J/Z handshape looks against other classes.

## Out-of-dataset (eval_external)

Skipped: folder `eval_external/<class_name>/*.jpg` does not exist.
