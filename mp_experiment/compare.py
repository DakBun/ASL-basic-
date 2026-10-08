"""Compare the MediaPipe model vs the hosted ResNet50 on the FULL test split.

Reads mp_model_*.joblib, features_test.npz and resnet_preds_test.csv.
Undetected non-"nothing" images count as errors for MediaPipe (predicted as
"nothing"). Writes results/compare.md, two confusion PNGs, top-10 confused
pairs, and a J/Z section.

Usage:
    python compare.py [--dir E:/ASL/mp_experiment]
"""
import argparse
import csv
import glob
import os
import time

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)

matplotlib.use("Agg")

CLASSES = sorted(
    [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["del", "nothing", "space"]
)
EXTERNAL_DIR = "E:/ASL/eval_external"


def load_test_features(directory):
    path = os.path.join(directory, "features_test.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"features not found: {path} (run extract_landmarks.py)")
    d = np.load(path, allow_pickle=True)
    return (np.asarray(d["features"]), np.asarray(d["detected"], dtype=bool),
            np.asarray(d["label"]).astype(str), np.asarray(d["path"]).astype(str))


def load_mp_model(directory):
    cands = sorted(glob.glob(os.path.join(directory, "mp_model_*.joblib")))
    if not cands:
        raise SystemExit("no mp_model_*.joblib found (run train_mp.py)")
    blob = joblib.load(cands[0])
    print(f"loaded MP model: {os.path.basename(cands[0])} "
          f"(val acc={blob.get('val_accuracy', '?')})")
    return blob["model"], os.path.basename(cands[0])


def load_resnet_preds(directory, test_paths):
    path = os.path.join(directory, "resnet_preds_test.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"predictions not found: {path} (run eval_resnet.py)")
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            # later rows override earlier; only trust successful ones
            if r.get("pred") and not r.get("error"):
                rows[r["relpath"]] = r
    missing = [p for p in test_paths if p not in rows]
    if missing:
        raise SystemExit(f"resnet preds incomplete: {len(missing)} test rows "
                         f"missing (re-run eval_resnet.py to resume)")
    return rows


def top_confused(cm, labels, k=10):
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            n = int(cm[i, j] + cm[j, i])
            if n:
                pairs.append((n, labels[i], labels[j]))
    return sorted(pairs, reverse=True)[:k]


def plot_cm(cm, labels, title, out):
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


def evaluate(X, det, y_true, mp_clf, resnet_rows):
    """Return metrics dict for both models on one dataset."""
    # MediaPipe: no hand -> "nothing"
    t0 = time.perf_counter()
    mp_pred = []
    for xi, d in zip(X, det):
        if not d:
            mp_pred.append("nothing")
        else:
            mp_pred.append(str(mp_clf.predict(xi.reshape(1, -1))[0]))
    mp_ms = (time.perf_counter() - t0) * 1000.0 / max(len(X), 1)
    mp_pred = np.array(mp_pred)

    rn_pred = np.array([p if p else "<<error>>" for p, _ in resnet_rows])

    def stats(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        macro = f1_score(y_true, y_pred, labels=CLASSES, average="macro",
                         zero_division=0)
        per = f1_score(y_true, y_pred, labels=CLASSES, average=None,
                       zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        return acc, macro, dict(zip(CLASSES, per)), cm

    # ponytail: ceiling = API error rows mapped to a pseudo-label never in
    # CLASSES, so they always count as misses. Upgrade path = explicit
    # sample-weighted metrics that keep error rows separate.
    return {
        "mp": stats(y_true, mp_pred),
        "rn": stats(y_true, rn_pred),
        "mp_ms": mp_ms, "mp_pred": mp_pred, "rn_pred": rn_pred,
        "det_rate": float(np.mean(det)),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    args = parser.parse_args()
    res_dir = os.path.join(args.dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    X, det, y_true, paths = load_test_features(args.dir)
    mp_clf, mp_name = load_mp_model(args.dir)
    resnet_rows_raw = load_resnet_preds(args.dir, list(paths))

    # keep order of split_test.csv
    ordered = [(resnet_rows_raw[p]["pred"], resnet_rows_raw[p]["ms"], p)
               for p in paths]
    rn_ms_vals = [float(ms) for _p, ms, _ in ordered if ms]
    rn_ms = float(np.mean(rn_ms_vals)) if rn_ms_vals else float("nan")
    rn_errors = sum(1 for _p, _ms, p in ordered
                    if not resnet_rows_raw[p]["pred"])

    r = evaluate(X, det, y_true, mp_clf, [(p, ms) for p, ms, _ in ordered])
    mp_acc, mp_macro, mp_f1, mp_cm = r["mp"]
    rn_acc, rn_macro, rn_f1, rn_cm = r["rn"]

    plot_cm(mp_cm, CLASSES, f"MediaPipe+{mp_name} confusion (test)",
            os.path.join(res_dir, "cm_mp.png"))
    plot_cm(rn_cm, CLASSES, "ResNet50 confusion (test)",
            os.path.join(res_dir, "cm_resnet.png"))

    mp_pairs = top_confused(mp_cm, CLASSES)
    rn_pairs = top_confused(rn_cm, CLASSES)

    L = []
    L.append("# ResNet50 vs MediaPipe comparison (FULL test split)")
    L.append("")
    L.append("**Caution / optimism note:** ResNet50 is evaluated through the "
             "hosted API on the same Kaggle images the project was trained "
             "from on Roboflow, so the hosted model may have seen these test "
             "images during training. **ResNet50 numbers may be optimistic** "
             "and are not a clean held-out estimate; MediaPipe features are "
             "extracted locally and never uploaded, so its numbers are "
             "unaffected by that risk.")
    L.append("")
    L.append(f"- Test size: {len(y_true)} images, {len(CLASSES)} classes "
             f"(stratified split, seed 42).")
    L.append(f"- MP model: `{mp_name}`; no-hand -> \"nothing\".")
    L.append("- ResNet50: `sign-language-2c6kk/2` via "
             "`classify.roboflow.com` (server-side preprocessing).")
    L.append(f"- MediaPipe detection rate on test: {r['det_rate']:.1%} "
             f"({int(np.sum(det))}/{len(det)}).")
    L.append(f"- ResNet50 API rows with error/missing: {rn_errors} (counted "
             "as misses).")
    L.append("")
    L.append("| model | overall accuracy | macro F1 | inference ms/image |")
    L.append("|---|---|---|---|")
    L.append(f"| ResNet50 (hosted) | {rn_acc:.4f} | {rn_macro:.4f} | "
             f"{rn_ms:.1f} (incl. network) |")
    L.append(f"| MediaPipe + {mp_name} | {mp_acc:.4f} | {mp_macro:.4f} | "
             f"{r['mp_ms']:.2f} (classifier only, excl. landmark "
             "extraction) |")
    L.append("")
    L.append("## Per-class F1")
    L.append("")
    L.append("| class | ResNet50 F1 | MediaPipe F1 |")
    L.append("|---|---|---|")
    for c in CLASSES:
        L.append(f"| {c} | {rn_f1[c]:.4f} | {mp_f1[c]:.4f} |")
    L.append("")
    L.append("## Top 10 most confused class pairs (both directions summed)")
    L.append("")
    L.append("**ResNet50:**")
    for n, a, b in rn_pairs:
        L.append(f"- {a} <-> {b}: {n}")
    L.append("")
    L.append("**MediaPipe:**")
    for n, a, b in mp_pairs:
        L.append(f"- {a} <-> {b}: {n}")
    L.append("")
    L.append("## J and Z only (static frames)")
    L.append("")
    for c in ("J", "Z"):
        m = y_true == c
        rn_c = int(np.sum(r["rn_pred"][m] == c))
        mp_c = int(np.sum(r["mp_pred"][m] == c))
        L.append(f"- **{c}**: ResNet50 {rn_c}/{int(m.sum())} "
                 f"({rn_c / m.sum():.4f}); MediaPipe {mp_c}/{int(m.sum())} "
                 f"({mp_c / m.sum():.4f})")
    L.append("")
    L.append("**Note:** J and Z are dynamic letters (motion), but this "
             "dataset provides single static frames only, so these numbers "
             "do not measure motion recognition. They only reflect how "
             "distinct the static J/Z handshape looks against other "
             "classes.")
    L.append("")
    L.append("## Out-of-dataset (eval_external)")
    L.append("")
    if os.path.isdir(EXTERNAL_DIR):
        L.append("`eval_external/` exists but was not evaluated in this run "
                 "(no predictions cached). Re-run with an external eval "
                 "pass to populate.")
    else:
        L.append("Skipped: folder `eval_external/<class_name>/*.jpg` does "
                 "not exist.")
    L.append("")

    md = os.path.join(res_dir, "compare.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"saved -> {md}")
    print(f"ResNet50 acc={rn_acc:.4f} macroF1={rn_macro:.4f} ms={rn_ms:.1f}")
    print(f"MediaPipe acc={mp_acc:.4f} macroF1={mp_macro:.4f} "
          f"ms={r['mp_ms']:.2f} det={r['det_rate']:.1%}")


if __name__ == "__main__":
    main()

