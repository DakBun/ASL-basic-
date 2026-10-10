# ASL Alphabet Recognition — MediaPipe landmarks vs hosted ResNet50

This project classifies static American Sign Language (ASL) alphabet hand
signs — 29 classes: A–Z plus `del`, `nothing`, `space` — from small RGB
images. The main experiment (in `mp_experiment/`) compares a **fully local
pipeline** (MediaPipe `HandLandmarker` cascade + a small scikit-learn
classifier on 63 normalized landmark features) against a **hosted Roboflow
ResNet50** on the same stratified train/val/test split. MediaPipe features
are always extracted locally and never uploaded; only the optional ResNet50
baseline sends images to a remote API.

## Repository layout

```
.
├── README.md, requirements.txt
├── sample_kaggle_asl.py   # sample N images/class from the Kaggle dataset -> kaggle_sample/
├── webcam_demo.py         # live demo via the hosted Roboflow model (needs API key)
├── kaggle_sample/         # 29 classes x 150 images (git-ignored; build it yourself)
├── data/vsl_alphabet/     # placeholder for future self-collected data
├── mp_experiment/         # main pipeline; see mp_experiment/README.md for v1/v2 details
│   ├── make_split.py, extract_landmarks.py, extract_landmarks_v2.py
│   ├── train_mp.py, train_mp_v2.py, train_mp_v3.py
│   ├── compare.py, compare_v2.py, eval_resnet.py, eval_external.py
│   ├── detect_report.py, detect_sweep.py
│   ├── webcam_mp_v2.py, collect_landmarks.py, capture_external.py
│   ├── split_{train,val,test}.csv, resnet_preds_test.csv   # committed
│   └── results/           # committed reports (.md) + confusion-matrix PNGs
└── _archive_rf/           # archived earlier pipeline; expected location of
                           # models/hand_landmarker.task (not shipped; *.task is git-ignored)
```

## Setup

1. `pip install -r requirements.txt`
2. **Dataset** — download the Kaggle dataset **"ASL Alphabet"**
   (`https://www.kaggle.com/datasets/grassknoted/asl-alphabet`; the archive
   contains an `asl_alphabet_train/` folder with one subfolder per class),
   then build the working sample (defaults: 150 images/class, seed 42):

   ```
   python sample_kaggle_asl.py <kaggle_root>/asl_alphabet_train E:/ASL/kaggle_sample --per-class 150
   ```

3. **Hand landmarker model** — download
   `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
   and save it as `E:/ASL/_archive_rf/models/hand_landmarker.task`.
   The path is hardcoded in `extract_landmarks_v2.py`; the file is
   git-ignored and is **not** included in a fresh clone.

Note: the scripts default to absolute Windows paths (`E:/ASL/...`); override
them with `--dataset` / `--dir` / `--out` where the script provides those
flags (verified for make_split, extract_landmarks*, train_mp*, compare*,
eval_* and webcam_mp_v2).

## Reproduce (run inside `mp_experiment/`)

```
python make_split.py                   # split_{train,val,test}.csv (seed 42, 70/15/15)
python extract_landmarks_v2.py         # features_v2_{train,val,test}.npz (63 features)
python train_mp_v2.py                  # mp_model_v2.joblib (required input for v3)
python train_mp_v3.py                  # mp_model_v3.joblib + results/compare_v3.md
python compare_v2.py                   # results/compare_v2.md + confusion PNG
python webcam_mp_v2.py --no-canon      # live demo (no camera: --selftest --no-canon)
```

All script names and flags above were verified against the code. There is
**no `compare_v3.py`**: the v3 report `results/compare_v3.md` is written by
`train_mp_v3.py` itself, and that script refuses to run until
`mp_model_v2.joblib` exists (hence the `train_mp_v2.py` step).

Optional hosted ResNet50 baseline (requires the "optional: ResNet50 baseline"
packages and `ROBOFLOW_API_KEY` in a local `E:/ASL/.env`):

```
python eval_resnet.py                  # resnet_preds_test.csv (resumable cache)
python compare.py                      # results/compare.md (needs resnet_preds_test.csv)
```

## Results (numbers copied from the committed files in `mp_experiment/results/`)

Test split: 653 images, 29 classes, stratified (seed 42).

From `results/compare.md` (full test split):

| model | overall accuracy | macro F1 |
|---|---|---|
| ResNet50 (hosted) | 1.0000 | 1.0000 |
| MediaPipe + `mp_model_mlp.joblib` | 0.7550 | 0.8089 |

MediaPipe v1 detection rate on test: 74.7% (488/653). J: 18/22, Z: 20/23
(MediaPipe).

From `results/compare_v2.md` (2-stage cascade):

- Detected: 539/653 (82.5%); stage 1 = 488, stage 2 = 51.
- Accuracy on detected samples: 0.9722; J 22/22, Z 22/22.
- Undetected -> "nothing" policy: 0.8361 overall; abstain policy: 17.5%
  abstain (114/653), 0.9722 accuracy on answered samples.
- Leakage check: accuracy on the nearest-train-distance quartile 1.0000 vs
  farthest quartile 0.8955 (gap +0.1045).

From `results/compare_v3.md` (mirror augmentation, detected test n=539):

| model | original test | mirrored test |
|---|---|---|
| v2 | 0.9722 | 0.1169 |
| v3 | 0.9666 | 0.9722 |

From `results/detection_sweep.md` (v1 settings on test): best swept setting
detected 518/653 (79.3%) vs the 74.7% baseline.

## Limitations

- The hosted ResNet50 result (1.0000) is **not a clean baseline**: the hosted
  model was likely trained on these same Kaggle images (via Roboflow), so its
  test score may be optimistic.
- All numbers are measured on **one Kaggle sample** (29 classes x 150 images,
  200x200) whose burst-shot frames are near-duplicates across the train/test
  split, so scores are inflated relative to real generalization.
- The **mirrored test is synthetic** (x negated only) and does not replace
  real left-hand data.
- **J and Z are classified from a static frame only**; motion is not measured.
- **MediaPipe misses the hand on about 15-17 percent of the images**
  (17.5% overall; 14.6% excluding `nothing` for the v2 cascade).

## Security notes

- `.joblib` models are Python pickles: they are not shipped with this repo,
  and you should never load a pickle from an untrusted source.
- API keys belong in a local `.env` (`ROBOFLOW_API_KEY`), which is
  git-ignored. Never commit keys.

## Data and licenses

The images come from the Kaggle dataset **"ASL Alphabet"**
(`https://www.kaggle.com/datasets/grassknoted/asl-alphabet`); the hand
landmark model comes from Google **MediaPipe**
(`https://mediapipe.dev`). Neither the dataset images nor the
`hand_landmarker.task` file are shipped in this repository. Check the
dataset's and MediaPipe's own terms before using, retraining on, or
redistributing them; this README deliberately states no license for them.