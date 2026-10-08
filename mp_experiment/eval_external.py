"""Compare hosted ResNet50 vs MediaPipe v2 on out-of-dataset images.

Walks eval_external/<class>/*.jpg (never modifies them). MediaPipe v2 uses the
same cascade + normalization as extract_landmarks_v2.py (functions copied
below, identical code, so the behaviour matches the trained model).
Hosted ResNet50 uses the same API call + caching + backoff as eval_resnet.py.
Caches predictions to resnet_preds_external.csv (resumable).

Writes results/compare_external.md. No API calls are made unless the user
explicitly runs eval_external.py.

Usage:
    python eval_external.py [--dataset E:/ASL/eval_external] [--dir E:/ASL/mp_experiment]
"""
import argparse
import csv
import glob
import os
import time
from collections import Counter, defaultdict

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# --- identical to extract_landmarks_v2.py (copied, not imported, so this
# script can run standalone without touching that file's internals) ---
MODEL_PATH = "E:/ASL/_archive_rf/models/hand_landmarker.task"
PAD_FRAC = 0.25
GRAY = (128, 128, 128)
CONF_STAGE1 = 0.5
CONF_STAGE2 = 0.1
CLASSES = sorted(
    [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["del", "nothing", "space"]
)
NOTHING = "nothing"
LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
TOP_N = 10
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_detector(min_det_conf):
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_det_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def pad25(img):
    h, w = img.shape[:2]
    t = b = int(round(h * PAD_FRAC))
    l = r = int(round(w * PAD_FRAC))
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT,
                              value=GRAY)


def normalize(lm):
    wrist = lm[0].copy()
    centered = lm - wrist
    scale = float(np.linalg.norm(centered[9]))
    if scale < 1e-6:
        return None
    return (centered / scale).reshape(-1)


def detect(detector, img):
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


def mp_predict_all(paths, model_path, clf):
    """Cascade per image: stage1 conf0.5 no-prep; stage2 conf0.1 + pad25."""
    det1 = build_detector(CONF_STAGE1)
    det2 = build_detector(CONF_STAGE2)
    out = []
    for p in paths:
        img = cv2.imread(p) if os.path.isfile(p) else None
        if img is None:
            out.append((None, 0, None))  # feat, stage, pred
            continue
        feat = detect(det1, img)
        stage = 1 if feat is not None else 0
        if stage == 0:
            feat = detect(det2, pad25(img))
            stage = 2 if feat is not None else 0
        pred = str(clf.predict(feat.reshape(1, -1))[0]) if feat is not None else None
        out.append((feat, stage, pred))
    return out


def load_done(pred_csv):
    done = {}
    if os.path.isfile(pred_csv):
        with open(pred_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("pred") and not r.get("error"):
                    done[r["path"]] = r
    return done


def top_pairs(cm, labels, k, exclude=()):
    idx = [i for i, l in enumerate(labels) if l not in exclude]
    pairs = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="E:/ASL/eval_external")
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    parser.add_argument("--resnet-only", action="store_true",
                        help="only run the hosted ResNet50 pass")
    parser.add_argument("--mp-only", action="store_true",
                        help="only run the MediaPipe pass")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        raise SystemExit(
            f"no images found: {args.dataset}\n"
            "Run capture_external.py first to populate it, e.g.:\n"
            "    python mp_experiment/capture_external.py --session bg1_light")
    # ponytail: ceiling = one pass per model, no per-session sampling
    # strategy. Upgrade path = session-aware stratified mini-split.
    rows = []
    for cls in CLASSES:
        d = os.path.join(args.dataset, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMAGE_EXTS):
                rows.append((os.path.join(d, fn), cls, fn))
    if not rows:
        raise SystemExit(f"no images found under {args.dataset}")
    print(f"external images: {len(rows)} "
          f"({len({c for _, c, _ in rows})} classes)")

    y_true = [c for _, c, _ in rows]
    paths = [p for p, _, _ in rows]
    sessions = [fn.split("_")[1] if "_" in fn else "unknown"
                for _, _, fn in rows]

    blob = joblib.load(os.path.join(args.dir, "mp_model_v2.joblib"))
    clf = blob["model"]
    print(f"model: {blob.get('kind', '?')} "
          f"(val acc {blob.get('val_accuracy', '?')})")

    # --- MediaPipe v2 ---
    if not args.resnet_only:
        if not os.path.isfile(MODEL_PATH):
            raise SystemExit(f"hand landmarker model not found: {MODEL_PATH}")
        t0 = time.perf_counter()
        mp_out = mp_predict_all(paths, MODEL_PATH, clf)
        mp_ms = (time.perf_counter() - t0) * 1000.0 / max(len(rows), 1)
        mp_pred = [p if p is not None else NOTHING for _, _, p in mp_out]
        mp_stage = [s for _, s, _ in mp_out]
        mp_detected = [s > 0 for _, s, _ in mp_out]
        mp_pred = np.array(mp_pred)
        mp_stage = np.array(mp_stage, dtype=np.int8)
        mp_detected = np.array(mp_detected, dtype=bool)
        print(f"MediaPipe: {int(mp_detected.sum())}/{len(rows)} detected "
              f"({mp_detected.mean():.1%}), {mp_ms:.2f} ms/image "
              f"(classifier-only)")
    else:
        mp_pred = np.array([NOTHING] * len(rows))
        mp_stage = np.zeros(len(rows), dtype=np.int8)
        mp_detected = np.zeros(len(rows), dtype=bool)

    # --- Hosted ResNet50 (same pattern as eval_resnet.py) ---
    rn_pred = np.array([""] * len(rows), dtype=object)
    if not args.mp_only:
        from dotenv import load_dotenv
        from inference_sdk import InferenceHTTPClient

        load_dotenv("E:/ASL/.env")
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise SystemExit("ROBOFLOW_API_KEY not found (check E:/ASL/.env)")
        pred_csv = os.path.join(args.dir, "resnet_preds_external.csv")
        done = load_done(pred_csv)
        print(f"resnet: {len(rows)} images, already cached: {len(done)}")
        client = InferenceHTTPClient(api_url="https://classify.roboflow.com",
                                     api_key=api_key)
        new_file = not os.path.isfile(pred_csv)
        f = open(pred_csv, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=["path", "label", "pred",
                                               "confidence", "ms", "error"])
        if new_file:
            writer.writeheader()
        n_new = 0
        try:
            for i, (p, label, _fn) in enumerate(rows):
                if p in done:
                    rn_pred[i] = done[p]["pred"]
                    continue
                last_err = ""
                for attempt in range(4):
                    t0 = time.perf_counter()
                    try:
                        # ponytail: ceiling = one blocking infer() per image
                        # over the hosted API. Upgrade path = batch/parallel
                        # requests + local checkpoint eval.
                        result = client.infer(p, model_id="sign-language-2c6kk/2")
                        ms = (time.perf_counter() - t0) * 1000.0
                        writer.writerow({"path": p, "label": label,
                                         "pred": result.get("top", ""),
                                         "confidence": result.get("confidence", ""),
                                         "ms": f"{ms:.1f}", "error": ""})
                        last_err = ""
                        rn_pred[i] = result.get("top", "")
                        break
                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e}"
                        if attempt < 3:
                            time.sleep(2.0 * (2 ** attempt))
                if last_err:
                    writer.writerow({"path": p, "label": label,
                                     "pred": "", "confidence": "",
                                     "ms": "", "error": last_err})
                    print(f"[{i + 1}/{len(rows)}] ERROR {p}: {last_err}")
                    f.flush()
                n_new += 1
                if (i + 1) % 10 == 0:
                    print(f"[{i + 1}/{len(rows)}] ok (new={n_new})")
                    f.flush()
                time.sleep(0.1)
        finally:
            f.close()
        print(f"resnet done: new={n_new} -> {pred_csv}")
    else:
        rn_pred = np.array([NOTHING] * len(rows), dtype=object)
            i, j = idx[a], idx[b]
            n = int(cm[i, j] + cm[j, i])
            if n:
                pairs.append((n, labels[i], labels[j]))
    return sorted(pairs, reverse=True)[:k]


def plot_cm(cm, labels, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved -> {out}")