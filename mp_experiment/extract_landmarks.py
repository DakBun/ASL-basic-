"""Extract MediaPipe hand landmarks for each image in the saved splits.

Substitution note: mediapipe 1.0.1 removed mp.solutions, so this uses the
Tasks API HandLandmarker in IMAGE running mode with num_hands=1 and
min_hand_detection_confidence=0.5 (equivalent of the spec's Hands(
static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)).

Normalization: subtract wrist (landmark 0), divide by wrist->MCP9 distance.
Flatten to 63 features. Undetected images are kept with detected=False and
zero features so the test split stays FULL for fair comparison.

Usage:
    python extract_landmarks.py [--dataset E:/ASL/kaggle_sample] [--dir E:/ASL/mp_experiment]
"""
import argparse
import csv
import os
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")


def build_detector(model_path):
    # ponytail: ceiling = hand_landmarker.task (~8MB, bundled in _archive_rf/models).
    # Upgrade path = hand_landmarker_full task or a newer model version.
    from mediapipe.tasks.python import vision

    if not os.path.isfile(model_path):
        raise SystemExit(
            f"hand landmarker model not found: {model_path}\n"
            "expected _archive_rf/models/hand_landmarker.task "
            "(run from E:/ASL so the relative path resolves)"
        )
    base = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def normalize(lm):
    """lm: (21,3) float array. Wrist-center, MCP9-scale. Returns (63,) array."""
    wrist = lm[0].copy()
    centered = lm - wrist
    scale = float(np.linalg.norm(centered[9]))  # wrist -> middle-finger MCP
    if scale < 1e-6:
        return None  # degenerate geometry, treat as undetected
    return (centered / scale).reshape(-1)


def extract_one(detector, abs_path):
    """Return (feat63 or None, handedness str or None). Never raises on bad input."""
    img = cv2.imread(abs_path)
    if img is None:
        return None, None  # unreadable image -> logged by caller
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:
        return None, None  # detection crash -> count, don't crash
    if not result.hand_landmarks:
        return None, None
    lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]],
                  dtype=np.float64)
    feat = normalize(lm)
    if feat is None:
        return None, None
    hand = "Unknown"
    try:
        hand = result.handedness[0][0].category_name or "Unknown"
    except Exception:
        pass
    return feat, hand


def load_split(split_csv):
    if not os.path.isfile(split_csv):
        raise SystemExit(f"split file not found: {split_csv} "
                         "(run make_split.py first)")
    rows = []
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["relpath"], row["label"]))
    if not rows:
        raise SystemExit(f"split file is empty: {split_csv}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="E:/ASL/kaggle_sample")
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    parser.add_argument("--model",
                        default="E:/ASL/_archive_rf/models/hand_landmarker.task")
    args = parser.parse_args()

    detector = build_detector(args.model)

    for split in SPLITS:
        rows = load_split(os.path.join(args.dir, f"split_{split}.csv"))
        feats, detected, hands, labels, paths = [], [], [], [], []
        n_unreadable = 0
        det_by_class = Counter()
        tot_by_class = Counter()
        hand_counts = Counter()

        for relpath, label in rows:
            tot_by_class[label] += 1
            abs_path = os.path.join(args.dataset, relpath)
            if not os.path.isfile(abs_path):
                n_unreadable += 1
                feats.append(np.zeros(63))
                detected.append(False)
                hands.append("Unknown")
                labels.append(label)
                paths.append(relpath)
                continue
            feat, hand = extract_one(detector, abs_path)
            if feat is None:
                if hand is None and cv2.imread(abs_path) is None:
                    n_unreadable += 1
                feats.append(np.zeros(63))
                detected.append(False)
                hands.append("Unknown")
            else:
                feats.append(feat)
                detected.append(True)
                hands.append(hand)
                det_by_class[label] += 1
                hand_counts[hand] += 1
            labels.append(label)
            paths.append(relpath)

        out = os.path.join(args.dir, f"features_{split}.npz")
        np.savez_compressed(
            out,
            features=np.stack(feats).astype(np.float32),
            detected=np.array(detected, dtype=bool),
            handedness=np.array(hands),
            label=np.array(labels),
            path=np.array(paths),
        )
        n_det = int(np.sum(detected))
        print(f"[{split}] {n_det}/{len(rows)} detected "
              f"({n_det / len(rows):.1%}), unreadable={n_unreadable} -> {out}")
        print(f"[{split}] handedness counts: {dict(hand_counts)}")
        print(f"[{split}] detection rate per class:")
        for cls in sorted(tot_by_class):
            r = det_by_class[cls] / tot_by_class[cls]
            print(f"  {cls}: {det_by_class[cls]}/{tot_by_class[cls]} ({r:.1%})")


if __name__ == "__main__":
    main()
