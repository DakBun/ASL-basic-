"""Detection sweep on the TEST split: min_hand_detection_confidence x preprocessing.

Combos: min_hand_detection_confidence in {0.5, 0.3, 0.1} x
preprocessing in {none, pad25, up2x, pad25_up2x}. Baseline = (0.5, none),
matching extract_landmarks.py / features_test.npz.

For images newly detected under a setting (detected now, NOT in baseline),
runs mp_model_mlp.joblib and reports its accuracy on those images separately
(landmark-garbage check).

Writes results/detection_sweep.md. Never touches features_*.npz.

Usage:
    python detect_sweep.py [--dataset E:/ASL/kaggle_sample] [--dir E:/ASL/mp_experiment]
"""
import argparse
import csv
import os
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_landmarks import normalize  # reuse the exact normalization

CONFS = (0.5, 0.3, 0.1)
PREPS = ("none", "pad25", "up2x", "pad25_up2x")
PAD_FRAC = 0.25
GRAY = (128, 128, 128)  # constant gray fill for padding


def preprocess(img, prep):
    """Return preprocessed image for one sweep setting. ponytail: ceiling =
    these 4 fixed geometric transforms only (no histogram/CLAHE/background
    masking). Upgrade path = grid over CLAHE/border-value/background removal."""
    if img is None:
        return None
    if prep == "none":
        return img
    h, w = img.shape[:2]
    if prep == "pad25":
        top = bottom = int(round(h * PAD_FRAC))
        left = right = int(round(w * PAD_FRAC))
        return cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=GRAY)
    if prep == "up2x":
        return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if prep == "pad25_up2x":
        return cv2.resize(preprocess(img, "pad25"), None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)
    raise ValueError(f"unknown prep: {prep}")


def build_detector(model_path, min_det_conf):
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_det_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_one(detector, img):
    """Return (feat63, handedness) or (None, None). Never raises."""
    if img is None:
        return None, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:
        return None, None
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


def load_test(directory):
    csv_path = os.path.join(directory, "split_test.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"split file not found: {csv_path} (run make_split.py)")
    with open(csv_path, newline="") as f:
        rows = [(r["relpath"], r["label"]) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"split file is empty: {csv_path}")
    return rows


def load_baseline(directory):
    """Detected flags + labels from features_test.npz (baseline reference)."""
    path = os.path.join(directory, "features_test.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"features not found: {path} (run extract_landmarks.py)")
    d = np.load(path, allow_pickle=True)
    return (np.asarray(d["detected"], dtype=bool),
            np.asarray(d["label"]).astype(str))


def load_model(directory):
    path = os.path.join(directory, "mp_model_mlp.joblib")
    if not os.path.isfile(path):
        raise SystemExit(f"model not found: {path} (run train_mp.py)")
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="E:/ASL/kaggle_sample")
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    args = parser.parse_args()

    rows = load_test(args.dir)
    base_det, base_labels = load_baseline(args.dir)
    if len(base_det) != len(rows):
        raise SystemExit(f"features_test.npz ({len(base_det)}) and "
                         f"split_test.csv ({len(rows)}) disagree on length")
    if not np.array_equal(base_labels, np.array([l for _, l in rows])):
        print("WARNING: label order mismatch between npz and csv; "
              "using npz labels as ground truth")
    clf_blob = load_model(args.dir)
    clf = clf_blob["model"] if isinstance(clf_blob, dict) else clf_blob

    model_path = "E:/ASL/_archive_rf/models/hand_landmarker.task"
    if not os.path.isfile(model_path):
        raise SystemExit(f"model not found: {model_path}")

    results = []
    t_start = time.perf_counter()
    for conf in CONFS:
        detector = build_detector(model_path, conf)
        for prep in PREPS:
            n_det = n_unread = 0
            new_ok = new_wrong = 0
            for i, (relpath, _label) in enumerate(rows):
                abs_path = os.path.join(args.dataset, relpath)
                raw = cv2.imread(abs_path) if os.path.isfile(abs_path) else None
                if raw is None:
                    n_unread += 1
                    continue
                feat, _hand = detect_one(detector, preprocess(raw, prep))
                if feat is None:
                    continue
                n_det += 1
                if not base_det[i]:
                    # garbage check: does the existing model get it right?
                    pred = str(clf.predict(feat.reshape(1, -1))[0])
                    if pred == base_labels[i]:
                        new_ok += 1
                    else:
                        new_wrong += 1
            n = len(rows) - n_unread
            row = {"conf": conf, "prep": prep, "det": n_det,
                   "rate": n_det / n if n else 0.0,
                   "new": new_ok + new_wrong, "new_ok": new_ok,
                   "new_wrong": new_wrong, "unread": n_unread}
            results.append(row)
            if new_ok + new_wrong:
                acc_s = f"{new_ok / (new_ok + new_wrong):.4f}"
            else:
                acc_s = "n/a"
            print(f"conf={conf} prep={prep}: detected {n_det}/{n} "
                  f"({row['rate']:.1%}), new={row['new']}, "
                  f"model_acc_on_new={acc_s} (ok={new_ok}, "
                  f"wrong={new_wrong}), unreadable={n_unread}")
    elapsed = time.perf_counter() - t_start
    print(f"sweep done in {elapsed:.1f}s")
    # --- markdown table ---
    L = ["# Detection sweep (TEST split only)", ""]
    L.append(f"Baseline = (0.5, none), matching `features_test.npz` "
             f"({int(base_det.sum())}/{len(base_det)} = "
             f"{base_det.mean():.1%}).")
    L.append("")
    L.append("`new` = images detected under this setting but NOT in the "
             "baseline; `model acc on new` = existing "
             "`mp_model_mlp.joblib` accuracy on those images (checks the "
             "newly detected landmarks aren't garbage).")
    L.append("")
    L.append("| min_det_conf | preprocessing | detected | rate | new vs "
             "baseline | model acc on new (ok/wrong) | unreadable |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        if r["new"] > 0:
            acc = r["new_ok"] / r["new"]
            acc_s = f"{acc:.4f} ({r['new_ok']}/{r['new']})"
        else:
            acc_s = "n/a (no new detections)"
        L.append(f"| {r['conf']} | {r['prep']} | {r['det']}/{len(rows)} | "
                 f"{r['rate']:.1%} | {r['new']} | {acc_s} | {r['unread']} |")
    L.append("")
    val_acc = (clf_blob.get("val_accuracy", "?")
               if isinstance(clf_blob, dict) else "?")
    L.append(f"Runtime: {elapsed:.1f}s. Model: `mp_model_mlp.joblib` "
             f"(val acc {val_acc}) — retrained nothing; only inference.")
    L.append("")

    md = os.path.join(args.dir, "results", "detection_sweep.md")
    os.makedirs(os.path.dirname(md), exist_ok=True)
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"saved -> {md}")


if __name__ == "__main__":
    main()
