"""Live webcam demo for the MediaPipe v2 ASL classifier (stage 1 of 2).

Stage 1 = full pipeline + offline selftest, no camera loop (stage 2 adds it).
Pipeline (identical to extract_landmarks_v2.py, imported through sys.path and
never modified):
  stage 1 = HandLandmarker (Tasks API, IMAGE mode, num_hands=1,
            min_hand_detection_confidence=0.5) on the raw image;
  stage 2 = only on stage-1 misses: conf 0.1 on the image padded 25% per
            side with constant gray.
Features: subtract wrist, divide by wrist->landmark-9 distance -> 63 dims.
Classifier: mp_model_v2.joblib dict ['model','kind','classes','val_accuracy'],
predicted with obj["model"].predict_proba + obj["classes"]. It never predicts
"nothing" (trained on detected samples only), so a missing hand is reported
as "(no hand)" instead.

Handedness: extract_landmarks_v2.detect() does NOT expose it (returns only
the 63-vector and drops result.handedness), so _detect() below repeats the
same detector.detect() call and reads result.handedness[0][0].category_name
from the raw HandLandmarker result -- extract_landmarks_v2.py stays untouched.

Canonicalization: 93% of training samples have handedness "Right"; the
webcam hand is mirrored relative to that and the model is confidently wrong,
so when the label != CANON_LABEL the landmark x axis is negated (after wrist
subtraction, before scaling/flattening to 63 features). --no-canon disables
it; stage 2 adds the H key toggle.

Usage:
    python webcam_mp_v2.py --selftest [--n 100] [--seed 42] [--no-canon]
"""
import argparse
import os
import sys

import cv2
import joblib
import mediapipe as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import extract_landmarks_v2 as ext  # noqa: E402  (has main guard, import-safe)

# --- constants (no config system) ---
DATASET = "E:/ASL/kaggle_sample"
MP_DIR = "E:/ASL/mp_experiment"
MODEL_NAME = "mp_model_v2.joblib"
CANON_LABEL = "Right"
TOP_K = 3
SEED = 42
N_SELFTEST = 100


def load_model(path):
    """(model, classes) from the v2 joblib blob, validated."""
    if not os.path.isfile(path):
        raise SystemExit(f"model not found: {path} (run train_mp_v2.py)")
    blob = joblib.load(path)
    required = ("model", "kind", "classes", "val_accuracy")
    if not isinstance(blob, dict) or any(k not in blob for k in required):
        raise SystemExit(
            f"unexpected model format, want dict with keys {required}: {path}")
    model = blob["model"]
    if not hasattr(model, "predict_proba"):
        raise SystemExit(f"classifier has no predict_proba: {path}")
    classes = [str(c) for c in blob["classes"]]
    print(f"model: {blob['kind']} classes={len(classes)} "
          f"val_accuracy={blob['val_accuracy']}")
    return model, classes


def _detect(detector, img):
    """(landmarks (21,3) or None, handedness label or None). Never raises.

    extract_landmarks_v2.detect() cannot be reused here because it does not
    expose handedness; this repeats its detector.detect() call and takes the
    label from the raw result (same access pattern as detect_sweep.detect_one).
    """
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
    hand = "Unknown"
    try:
        hand = result.handedness[0][0].category_name or "Unknown"
    except Exception:
        pass
    return lm, hand


# ponytail: ceiling = handedness is itself a noisy prediction, so a wrong
# label flips a well-detected hand into an unrecognizable one. Upgrade path =
# apply the same canonicalization in training and retrain, or add left-hand
# data.
def features(lm, handedness, canon):
    """(feat63 or None, flipped): wrist-subtract, canon-flip x, scale+flatten.

    The x negation happens after wrist subtraction and before scaling and
    flattening to 63 features; y and z are kept.
    """
    centered = lm - lm[0]
    flipped = bool(canon and handedness != CANON_LABEL)
    if flipped:
        centered[:, 0] *= -1.0
    return ext.normalize(centered), flipped


def cascade(img, det1, det2, canon=True):
    """(feat63 or None, stage 0/1/2, handedness or None, flipped)."""
    if img is None:
        return None, 0, None, False
    lm, hand = _detect(det1, img)  # stage 1: conf 0.5, raw image
    if lm is not None:
        feat, flipped = features(lm, hand, canon)
        if feat is not None:
            return feat, 1, hand, flipped
    lm, hand = _detect(det2, ext.pad25(img))  # stage 2: conf 0.1, 25% gray pad
    if lm is None:
        return None, 0, None, False
    feat, flipped = features(lm, hand, canon)
    if feat is None:
        return None, 0, None, False
    return feat, 2, hand, flipped


def predict(feature_vec, model, classes):
    """top-3 [(letter, prob), ...], highest probability first."""
    proba = model.predict_proba(np.asarray(feature_vec).reshape(1, -1))[0]
    if len(proba) != len(classes):
        raise SystemExit(
            f"predict_proba size {len(proba)} != len(classes) {len(classes)}")
    order = np.argsort(proba)[::-1][:TOP_K]
    return [(str(classes[i]), float(proba[i])) for i in order]


def selftest(n=N_SELFTEST, seed=SEED, canon=True, dataset=DATASET,
             directory=MP_DIR):
    """Run the SAME cascade() + predict() (no smoothing) on n random test rows."""
    if not os.path.isdir(dataset):
        raise SystemExit(f"dataset root not found: {dataset}")
    if not os.path.isfile(ext.MODEL_PATH):
        raise SystemExit(f"hand landmarker task not found: {ext.MODEL_PATH}")
    rows = ext.load_split(directory, "test")  # split_test.csv (relpath,label)
    if n < 1 or n > len(rows):
        raise SystemExit(f"--n must be in [1, {len(rows)}] (test split size)")
    model, classes = load_model(os.path.join(directory, MODEL_NAME))
    idx = np.random.default_rng(seed).choice(len(rows), n, replace=False)
    sample = [rows[i] for i in idx]
    det1 = ext.build_detector(ext.CONF_STAGE1)
    det2 = ext.build_detector(ext.CONF_STAGE2)
    detected = correct = no_hand = flipped = not_right = unreadable = 0
    stage_counts = {}
    try:
        for relpath, label in sample:
            path = os.path.join(dataset, relpath)
            img = cv2.imread(path) if os.path.isfile(path) else None
            if img is None:
                unreadable += 1  # counts as no hand (same rule as extract)
            feat, stage, hand, was_flipped = cascade(img, det1, det2, canon)
            if feat is None:
                no_hand += 1
                continue
            detected += 1
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            flipped += int(was_flipped)
            not_right += int(hand != CANON_LABEL)
            top = predict(feat, model, classes)
            correct += int(top[0][0] == label)
    finally:
        det1.close()
        det2.close()
    acc = correct / detected if detected else 0.0
    print(f"selftest n={n} seed={seed} canon={'on' if canon else 'off'}")
    print(f"detected: {detected}/{n}")
    if detected:
        print(f"accuracy on detected: {acc:.4f} ({correct}/{detected})")
    else:
        print("accuracy on detected: n/a (nothing detected)")
    print(f"no hand: {no_hand}")
    print(f"stage counts: {dict(sorted(stage_counts.items()))}")
    print(f"flipped: {flipped}")
    print(f'handedness != "{CANON_LABEL}": {not_right}')
    if unreadable:
        print(f"unreadable images: {unreadable} (counted as no hand)")
    return {"detected": detected, "acc": acc, "no_hand": no_hand,
            "stages": stage_counts, "flipped": flipped,
            "not_right": not_right}


def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe v2 ASL webcam demo (stage 1: selftest only)")
    parser.add_argument("--selftest", action="store_true",
                        help="run the offline pipeline selftest")
    parser.add_argument("--n", type=int, default=N_SELFTEST,
                        help="number of random test images")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-canon", action="store_true",
                        help="disable handedness canonicalization")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dir", default=MP_DIR)
    args = parser.parse_args()
    if not args.selftest:
        parser.print_help()
        print("\nstage 1 supports --selftest only; the live loop arrives in stage 2")
        return
    selftest(n=args.n, seed=args.seed, canon=not args.no_canon,
             dataset=args.dataset, directory=args.dir)


if __name__ == "__main__":
    main()