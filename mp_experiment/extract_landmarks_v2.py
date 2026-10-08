"""Cascade hand-landmark extraction (v2).

Stage 1 per image: HandLandmarker (Tasks API, IMAGE mode, num_hands=1,
min_hand_detection_confidence=0.5, no preprocessing).
Stage 2 (only when stage 1 found nothing): same detector at
min_hand_detection_confidence=0.1 on the image padded 25% per side with
constant gray (same pad as detect_sweep.py).

Normalization identical to extract_landmarks.py: wrist subtraction, scale
by wrist->landmark-9 distance, 63 features.

Writes features_v2_{train,val,test}.npz with keys: features, label,
detected, stage (0=none, 1, 2), path. Never touches features_*.npz.

Usage:
    python extract_landmarks_v2.py [--dataset E:/ASL/kaggle_sample] [--dir E:/ASL/mp_experiment]
"""
import argparse
import csv
import os
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

SPLITS = ("train", "val", "test")
MODEL_PATH = "E:/ASL/_archive_rf/models/hand_landmarker.task"
PAD_FRAC = 0.25
GRAY = (128, 128, 128)  # constant gray, same as detect_sweep.py
CONF_STAGE1 = 0.5
CONF_STAGE2 = 0.1
CLASSES = sorted(
    [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["del", "nothing", "space"]
)


def build_detector(min_det_conf):
    from mediapipe.tasks.python import vision

    # ponytail: ceiling = 2-stage cascade, one fixed pad transform, no
    # border-value/CLAHE tuning. Upgrade path = sweep pad fraction +
    # border color + min_hand_presence_confidence per stage.
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_det_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def pad25(img):
    """Pad 25% each side with constant gray (same geometry as detect_sweep)."""
    h, w = img.shape[:2]
    t = b = int(round(h * PAD_FRAC))
    l = r = int(round(w * PAD_FRAC))
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=GRAY)


def normalize(lm):
    """lm: (21,3). Wrist-center, MCP9-scale -> (63,), identical to v1."""
    wrist = lm[0].copy()
    centered = lm - wrist
    scale = float(np.linalg.norm(centered[9]))
    if scale < 1e-6:
        return None  # degenerate geometry -> counts as undetected
    return (centered / scale).reshape(-1)


def detect(detector, img):
    """feat63 or None; never raises (bad image / detector crash -> None)."""
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:
        return None
    if not result.hand_landmarks:
        return None
    lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]],
                  dtype=np.float64)
    return normalize(lm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="E:/ASL/kaggle_sample")
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    args = parser.parse_args()

    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"hand landmarker model not found: {MODEL_PATH}")
    det1 = build_detector(CONF_STAGE1)
    det2 = build_detector(CONF_STAGE2)

    for split in SPLITS:
        rows = load_split(args.dir, split)
        feats, labels, stages, paths = [], [], [], []
        n_unread = 0
        stage_counts = Counter()
        det_by_class, s1_by_class, s2_by_class, tot_by_class = (
            Counter(), Counter(), Counter(), Counter())

        for relpath, label in rows:
            tot_by_class[label] += 1
            abs_path = os.path.join(args.dataset, relpath)
            img = cv2.imread(abs_path) if os.path.isfile(abs_path) else None
            if img is None:
                n_unread += 1
            # stage 1: baseline settings, no preprocessing
            feat = detect(det1, img)
            stage = 1 if feat is not None else 0
            # stage 2: only on stage-1 misses (unreadable stays stage 0)
            if stage == 0 and img is not None:
                feat = detect(det2, pad25(img))
                stage = 2 if feat is not None else 0
            if feat is None:
                feats.append(np.zeros(63, dtype=np.float64))
            else:
                feats.append(feat)
                det_by_class[label] += 1
                (s1_by_class if stage == 1 else s2_by_class)[label] += 1
            stages.append(stage)
            labels.append(label)
            paths.append(relpath)
            stage_counts[stage] += 1

        out = os.path.join(args.dir, f"features_v2_{split}.npz")
        detected = np.array(stages, dtype=np.int8) > 0
        np.savez_compressed(
            out,
            features=np.stack(feats).astype(np.float32),
            label=np.array(labels),
            detected=detected,
            stage=np.array(stages, dtype=np.int8),
            path=np.array(paths),
        )
        n = len(rows)
        n_det = int(sum(det_by_class.values()))
        s1, s2, s0 = stage_counts[1], stage_counts[2], stage_counts[0]
        print(f"[{split}] {n_det}/{n} detected ({n_det / n:.1%}) | "
              f"stage1={s1} ({s1 / n:.1%}), stage2={s2} ({s2 / n:.1%}), "
              f"none={s0} | unreadable={n_unread} -> {out}")
        n_not = tot_by_class.get("nothing", 0)
        n_not_det = det_by_class.get("nothing", 0)
        n_cov = n - n_not
        print(f"[{split}] coverage excl. nothing: {n_det - n_not_det}/{n_cov} "
              f"({(n_det - n_not_det) / n_cov:.1%})")
        print(f"[{split}] per class [s1, s2]:")
        for cls in sorted(tot_by_class):
            d = det_by_class[cls]
            t = tot_by_class[cls]
            print(f"  {cls}: {d}/{t} ({d / t:.1%}) "
                  f"[s1={s1_by_class[cls]}, s2={s2_by_class[cls]}]")


def load_split(directory, split):
    path = os.path.join(directory, f"split_{split}.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"split file not found: {path} (run make_split.py)")
    with open(path, newline="") as f:
        rows = [(r["relpath"], r["label"]) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"split file is empty: {path}")
    return rows


if __name__ == "__main__":
    main()
