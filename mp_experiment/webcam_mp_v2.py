"""Live webcam demo for the MediaPipe v2 ASL classifier (stage 2: live loop).

Pipeline = extract_landmarks_v2.py (imported via sys.path, never edited):
stage 1 = HandLandmarker (Tasks API, IMAGE mode, num_hands=1) conf 0.5 on the
raw image; stage 2 = conf 0.1 on the image padded 25% per side with constant
gray (stage-1 misses only). Landmarks are wrist-centered, wrist->9 scaled
(63 features). Classifier: mp_model_v2.joblib (predict_proba + classes);
it never predicts "nothing", so a missing hand is shown as "(no hand)".

Handedness: ext.detect() does not expose it (it drops result.handedness), so
_detect() repeats its detector.detect() call and reads
result.handedness[0][0].category_name from the raw result -- ext is untouched.
When that label != CANON_LABEL ("Right", 93% of training samples) the
landmark x axis is negated after wrist subtraction, before scaling/flattening.
--no-canon disables it; key H toggles it live.

Usage:
    python webcam_mp_v2.py [--camera 0]
    python webcam_mp_v2.py --selftest [--n 100] [--seed 42] [--no-canon]
"""
import argparse
import os
import sys
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import extract_landmarks_v2 as ext  # noqa: E402  (main guard, import-safe)

# --- constants (no config system) ---
DATASET = "E:/ASL/kaggle_sample"
MP_DIR = "E:/ASL/mp_experiment"
MODEL_NAME = "mp_model_v2.joblib"
CANON_LABEL = "Right"
TOP_K = 3
SEED = 42
N_SELFTEST = 100
PAD_FRAC = 0.25         # matches ext.pad25; used to remap stage-2 landmarks
SMOOTH_N = 7            # keep the last N predictions with conf >= SMOOTH_CONF
SMOOTH_CONF = 0.6       # lower-confidence predictions never enter the window
SMOOTH_VOTES = 4        # votes needed to display a letter (else "...")
NOHAND_CLEAR = 10       # consecutive no-hand frames that clear the window


def load_model(path):
    """(model, classes) from the v2 joblib blob, validated."""
    if not os.path.isfile(path):
        raise SystemExit(f"model not found: {path} (run train_mp_v2.py)")
    blob = joblib.load(path)
    required = ("model", "kind", "classes", "val_accuracy")
    if not isinstance(blob, dict) or any(k not in blob for k in required):
        raise SystemExit(f"bad model format, want dict {required}: {path}")
    if not hasattr(blob["model"], "predict_proba"):
        raise SystemExit(f"classifier has no predict_proba: {path}")
    classes = [str(c) for c in blob["classes"]]
    print(f"model: {blob['kind']} classes={len(classes)} "
          f"val_accuracy={blob['val_accuracy']}")
    return blob["model"], classes


def _detect(detector, img):
    """(landmarks (21,3) or None, handedness or None). Never raises.

    ext.detect() cannot be reused here because it does not expose handedness;
    this repeats its detector.detect() call and reads the raw result
    (same access pattern as detect_sweep.detect_one).
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
    """(feat63 or None, flipped): wrist-subtract, canon-flip x, scale+flatten."""
    centered = lm - lm[0]
    flipped = bool(canon and handedness != CANON_LABEL)
    if flipped:
        centered[:, 0] *= -1.0
    return ext.normalize(centered), flipped


def cascade(img, det1, det2, canon=True, return_lm=False):
    """(feat63 or None, stage 0/1/2, handedness, flipped) [+ lm if return_lm]."""
    lm, stage, feat, hand, flipped = None, 0, None, None, False
    if img is not None:
        lm, hand = _detect(det1, img)  # stage 1: conf 0.5, raw image
        if lm is not None:
            feat, flipped = features(lm, hand, canon)
            stage = 1 if feat is not None else 0
        if feat is None:
            lm, hand = _detect(det2, ext.pad25(img))  # stage 2: conf 0.1 + pad
            if lm is not None:
                feat, flipped = features(lm, hand, canon)
                stage = 2 if feat is not None else 0
        if feat is None:
            stage, hand, flipped = 0, None, False
    out = (feat, stage, hand, flipped)
    return out + (lm,) if return_lm else out


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


def _to_feed_lm(lm, stage):
    """Landmarks from the cascade's input frame -> unpadded frame coords."""
    if stage == 2:  # stage 2 runs on the 25%-padded image
        lm = lm.copy()
        k = 1.0 + 2.0 * PAD_FRAC
        lm[:, 0] = lm[:, 0] * k - PAD_FRAC
        lm[:, 1] = lm[:, 1] * k - PAD_FRAC
    return lm


def draw_hand(display, lm, stage, feed_flip):
    """21 landmarks + bbox; feed-frame lm -> displayed (mirrored) frame."""
    h, w = display.shape[:2]
    lm = _to_feed_lm(lm, stage)
    pts = []
    for p in lm:
        x = p[0] if feed_flip else 1.0 - p[0]  # display = mirror(raw)
        pts.append((int(round(x * w)), int(round(p[1] * h))))
    for pt in pts:
        cv2.circle(display, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    cv2.rectangle(display, (min(xs) - 8, min(ys) - 8),
                  (max(xs) + 8, max(ys) + 8), (0, 255, 0), 2)


def draw_hud(display, lines, shown, has_hand):
    h = display.shape[0]
    panel = display.copy()
    cv2.rectangle(panel, (0, 0), (360, 18 + 22 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, display, 0.45, 0, display)
    for i, text in enumerate(lines):
        cv2.putText(display, text, (10, 32 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    color = (0, 255, 0) if has_hand else (0, 0, 255)
    cv2.putText(display, shown, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, color, 3, cv2.LINE_AA)


def run_live(args):
    """Interactive camera loop (untested here: no camera in this environment)."""
    if not os.path.isfile(ext.MODEL_PATH):
        raise SystemExit(f"hand landmarker task not found: {ext.MODEL_PATH}")
    model, classes = load_model(os.path.join(args.dir, MODEL_NAME))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera} "
                         "(plugged in? in use by another app?)")
    det1 = ext.build_detector(ext.CONF_STAGE1)
    det2 = ext.build_detector(ext.CONF_STAGE2)
    canon, feed_flip, smooth = not args.no_canon, False, True
    hist = deque(maxlen=SMOOTH_N)
    nohand, prev, fps = 0, time.perf_counter(), 0.0
    # J and Z are dynamic letters: a single frame cannot capture motion.
    # ponytail: ceiling = static frame; upgrade = fingertip trajectory tracking.
    print("keys: Q quit | F feed | S smooth | H canon")
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("camera read failed, quitting")
                break
            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 1.0 / dt if fps <= 0 else 0.9 * fps + 0.1 / dt
            display = cv2.flip(raw, 1)              # always show mirrored
            feed = display if feed_flip else raw    # default: feed un-mirrored
            # full frame, no center crop
            feat, stage, hand, flipped, lm = cascade(
                feed, det1, det2, canon, return_lm=True)
            if feat is None:
                nohand += 1
                if nohand >= NOHAND_CLEAR:
                    hist.clear()
                top, shown, has_hand = None, "(no hand)", False
            else:
                nohand, has_hand = 0, True
                top = predict(feat, model, classes)
                if smooth:
                    if top[0][1] >= SMOOTH_CONF:
                        hist.append(top[0][0])
                    if hist:
                        common, votes = Counter(hist).most_common(1)[0]
                        shown = common if votes >= SMOOTH_VOTES else "..."
                    else:
                        shown = "..."
                else:
                    shown = top[0][0]
                draw_hand(display, lm, stage, feed_flip)
            lines = [
                "  ".join(f"{c} {p:.2f}" for c, p in top) if top
                else "(no hand)",
                f"stage: {stage}",
                f"hand: {hand or 'none'}  flipped: {'yes' if flipped else 'no'}",
                f"canon: {'on' if canon else 'off'}",
                f"feed: {'mirrored' if feed_flip else 'unmirrored'}",
                f"smooth: {'on' if smooth else 'off'}  fps: {fps:.1f}",
                "Q quit | F feed | S smooth | H canon",
            ]
            draw_hud(display, lines, shown, has_hand)
            cv2.imshow("webcam_mp_v2", display)
            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += 32  # accept Caps-Lock as well
            if key == ord("q"):
                break
            if key == ord("f"):
                feed_flip = not feed_flip
            if key == ord("s"):
                smooth = not smooth
            if key == ord("h"):
                canon = not canon
    finally:
        cap.release()
        cv2.destroyAllWindows()
        det1.close()
        det2.close()


def main():
    parser = argparse.ArgumentParser(description="MediaPipe v2 ASL webcam demo")
    parser.add_argument("--selftest", action="store_true",
                        help="offline pipeline selftest (no camera)")
    parser.add_argument("--n", type=int, default=N_SELFTEST,
                        help="selftest sample size")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-canon", action="store_true",
                        help="disable handedness canonicalization")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dir", default=MP_DIR)
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index for the live demo")
    args = parser.parse_args()
    if args.selftest:
        selftest(n=args.n, seed=args.seed, canon=not args.no_canon,
                 dataset=args.dataset, directory=args.dir)
        return
    run_live(args)


if __name__ == "__main__":
    main()