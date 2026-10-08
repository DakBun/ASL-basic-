# MediaPipe vs hosted ResNet50 — ASL Alphabet (29 classes)

This folder holds a local experiment comparing a **MediaPipe hand-landmark
cascade + small classifier** against the **hosted ResNet50**
(`sign-language-2c6kk/2` on `classify.roboflow.com`) on the same 29 ASL
classes (A–Z, del, nothing, space).

## Run order (v2 pipeline)

1. `python make_split.py` — stratified split (seed 42): `split_train.csv`
   (3045), `split_val.csv` (652), `split_test.csv` (653).
2. `python extract_landmarks_v2.py` — cascade: stage 1 = HandLandmarker
   conf 0.5, no preprocessing; stage 2 = conf 0.1 + 25% gray pad on misses.
   Writes `features_v2_{train,val,test}.npz` (63 features, stage 0/1/2).
3. `python train_mp_v2.py` — trains MLP + RF on detected train samples only,
   saves the winner as `mp_model_v2.joblib`.
4. `python eval_resnet.py` — hosted ResNet50 on the test split
   (`resnet_preds_test.csv`, resumable, 5 test calls first).
5. `python compare.py` — writes `results/compare.md` + confusion PNGs.
6. `python compare_v2.py` — MediaPipe v2 only: `results/compare_v2.md` +
   `results/cm_v2_detected.png`.

## Where results are

- `results/compare.md` — ResNet50 vs MediaPipe v1 (full test split).
- `results/compare_v2.md` — MediaPipe v2 coverage, accuracy, per-stage,
  two policies for undetected images, J/Z, and a nearest-train leakage check.
- `results/detection_by_class.md` and `results/detection_sweep.md` —
  detection-rate diagnostics.
- `results/cm_mp.png`, `results/cm_resnet.png`, `results/cm_v2_detected.png`
  — confusion matrices.

## Caveat — hosted ResNet50 numbers may be contaminated

`eval_resnet.py` queries the hosted model on the same Kaggle images the
project was trained from on Roboflow, so the hosted model may have seen
these test images during training. **ResNet50 numbers in `compare.md` may
be optimistic** and are not a clean held-out estimate. MediaPipe features
are extracted locally and never uploaded, so its numbers are unaffected.

## Dependency

`extract_landmarks_v2.py` needs the MediaPipe hand landmarker model at
`E:/ASL/_archive_rf/models/hand_landmarker.task` (7.8 MB, bundled). Do not
delete it or move `_archive_rf/`.