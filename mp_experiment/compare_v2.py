"""MediaPipe v2 (cascade) evaluation on the FULL test split.

Reads features_v2_{train,test}.npz + mp_model_v2.joblib (never modifies
them). Writes results/compare_v2.md and confusion PNGs. No API calls.

Usage:
    python compare_v2.py [--dir E:/ASL/mp_experiment]
"""
import argparse
import os
import time

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = sorted(
    [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["del", "nothing", "space"]
)
LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
TOP_N = 10
NOTHING = "nothing"


def load_npz(directory, split):
    path = os.path.join(directory, f"features_v2_{split}.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"features not found: {path} "
                         "(run extract_landmarks_v2.py)")
    d = np.load(path, allow_pickle=True)
    return {
        "X": np.asarray(d["features"], dtype=np.float64),
        "detected": np.asarray(d["detected"], dtype=bool),
        "stage": np.asarray(d["stage"], dtype=np.int8),
        "y": np.asarray(d["label"]).astype(str),
        "path": np.asarray(d["path"]).astype(str),
    }


def top_pairs(cm, labels, k, exclude=()):
    """Top-k confused pairs (both directions summed)."""
    idx = [i for i, l in enumerate(labels) if l not in exclude]
    pairs = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    args = parser.parse_args()
    res_dir = os.path.join(args.dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    test = load_npz(args.dir, "test")
    train = load_npz(args.dir, "train")
    blob_path = os.path.join(args.dir, "mp_model_v2.joblib")
    if not os.path.isfile(blob_path):
        raise SystemExit(f"model not found: {blob_path} (run train_mp_v2.py)")
    blob = joblib.load(blob_path)
    clf = blob["model"]
    print(f"model: {blob.get('kind', '?')} "
          f"(val acc {blob.get('val_accuracy', '?')})")

    y = test["y"]
    det = test["detected"]
    stage = test["stage"]
    X = test["X"]
    n = len(y)
    n_det = int(det.sum())

    # --- predictions on detected samples ---
    t0 = time.perf_counter()
    pred = np.array([str(clf.predict(X[i].reshape(1, -1))[0])
                     for i in range(n) if det[i]])
    det_ms = (time.perf_counter() - t0) * 1000.0 / max(n_det, 1)
    y_det = y[det]
    stage_det = stage[det]
    acc_det = accuracy_score(y_det, pred)
    s1m, s2m = stage_det == 1, stage_det == 2
    acc_s1 = accuracy_score(y_det[s1m], pred[s1m]) if s1m.any() else None
    acc_s2 = accuracy_score(y_det[s2m], pred[s2m]) if s2m.any() else None
    f1_det = f1_score(y_det, pred, labels=CLASSES, average=None,
                      zero_division=0)
    f1_map = dict(zip(CLASSES, f1_det))
    cm_det = confusion_matrix(y_det, pred, labels=CLASSES)

    # --- coverage ---
    not_mask = y == NOTHING
    n_not = int(not_mask.sum())
    n_not_det = int(det[not_mask].sum())
    cov_all = n_det / n
    cov_excl = (n_det - n_not_det) / max(n - n_not, 1)

    plot_cm(cm_det, CLASSES,
            "MediaPipe v2 cascade - confusion (detected-only test)",
            os.path.join(res_dir, "cm_v2_detected.png"))

    # --- policy A: undetected -> "nothing" ---
    pred_a = np.array(["" for _ in range(n)], dtype=object)
    pred_a[det] = pred
    pred_a[~det] = NOTHING
    acc_a = accuracy_score(y, pred_a)
    ni = CLASSES.index(NOTHING)
    cm_a = confusion_matrix(y, pred_a, labels=CLASSES)
    nothing_recall = cm_a[ni, ni] / cm_a[ni].sum() if cm_a[ni].sum() else 0.0
    nothing_prec = cm_a[ni, ni] / cm_a[:, ni].sum() if cm_a[:, ni].sum() else 0.0

    # --- policy B: undetected = abstain ---
    abstain_rate = 1.0 - cov_all
    acc_b = acc_det  # answered samples == detected samples

    # --- leakage check: NN distance in 63-d feature space ---
    # ponytail: ceiling = exact brute-force NN over detected train samples
    # (train∩test overlap by filename is not checked). Upgrade path =
    # dedupe by perceptual hash before splitting + NN against train+val.
    from scipy.spatial.distance import cdist
    Xtr = train["X"][train["detected"]]
    dmin = cdist(X[det], Xtr).min(axis=1)
    med = float(np.median(dmin))
    k = max(1, len(dmin) // 4)
    order = np.argsort(dmin)
    near, far = order[:k], order[-k:]
    acc_near = accuracy_score(y_det[near], pred[near])
    acc_far = accuracy_score(y_det[far], pred[far])

    # --- markdown ---
    L = ["# MediaPipe v2 (cascade) — TEST split report", ""]
    L.append("Cascade: stage 1 = HandLandmarker conf 0.5, no preprocessing; "
             "stage 2 = conf 0.1 + 25% gray pad (only on stage-1 misses). "
             "Model `mp_model_v2.joblib` trained on detected train samples "
             "only; nothing retrained beyond that.")
    L.append("")
    L.append(f"- Test size: {n}; detected: {n_det} "
             f"({cov_all:.1%}); unreadable counted as undetected.")
    L.append(f"- Coverage overall: **{cov_all:.1%}**; excluding "
             f"`nothing`: **{cov_excl:.1%}** "
             f"({n_det - n_not_det}/{n - n_not}).")
    L.append(f"- Detected by stage: stage 1 = {int(s1m.sum())}, "
             f"stage 2 = {int(s2m.sum())}.")
    L.append("")
    L.append("## Per-class coverage (detected / total)")
    L.append("")
    L.append("| class | detected / total | rate |")
    L.append("|---|---|---|")
    for c in CLASSES:
        m = y == c
        d = int(det[m].sum())
        L.append(f"| {c} | {d}/{int(m.sum())} | {d / m.sum():.1%} |")
    L.append("")
    L.append("## Accuracy on detected-only samples")
    L.append("")
    L.append(f"- Overall: **{acc_det:.4f}** ({n_det} samples, "
             f"{det_ms:.2f} ms/sample classifier-only)")
    L.append(f"- Stage 1 only: **{acc_s1:.4f}** ({int(s1m.sum())} samples)"
             if acc_s1 is not None else "- Stage 1 only: n/a")
    L.append(f"- Stage 2 only: **{acc_s2:.4f}** ({int(s2m.sum())} samples)"
             if acc_s2 is not None else "- Stage 2 only: n/a")
    L.append("")
    L.append("## Per-class F1 (detected-only)")
    L.append("")
    L.append("| class | F1 |")
    L.append("|---|---|")
    for c in CLASSES:
        L.append(f"| {c} | {f1_map[c]:.4f} |")
    L.append("")
    L.append("## Top 10 confused LETTER pairs (detected-only; "
             "`nothing`, `del`, `space` excluded)")
    L.append("")
    for cnt, a, b in top_pairs(cm_det, CLASSES, TOP_N, exclude=(NOTHING, "del", "space")):
        L.append(f"- {a} <-> {b}: {cnt}")
    L.append("")
    L.append("### The three non-letter classes, separately")
    L.append("")
    for c in (NOTHING, "del", "space"):
        ci = CLASSES.index(c)
        own = [(int(cm_det[ci, j] + cm_det[j, ci]), CLASSES[j])
               for j in range(len(CLASSES)) if j != ci]
        own = [p for p in own if p[0] > 0]
        own.sort(reverse=True)
        top = ", ".join(f"{cls}: {cnt}" for cnt, cls in own[:5]) or "none"
        n_c = int((y_det == c).sum())
        L.append(f"- **{c}** (n={n_c} detected): top confused with -> {top}")
    L.append("")
    L.append("## Policies for undetected images")
    L.append("")
    L.append(f"**(A) undetected -> \"{NOTHING}\":** overall accuracy = "
             f"**{acc_a:.4f}**; `nothing` recall = "
             f"**{nothing_recall:.4f}**, `nothing` precision = "
             f"**{nothing_prec:.4f}**.")
    L.append("")
    L.append(f"**(B) undetected = abstain:** abstain rate = "
             f"**{abstain_rate:.1%}** ({n - n_det}/{n}); accuracy on "
             f"answered samples = **{acc_b:.4f}**.")
    L.append("")
    L.append("## J and Z only (detected-only accuracy)")
    L.append("")
    for c in ("J", "Z"):
        m = y_det == c
        if not m.any():
            L.append(f"- **{c}**: n=0 detected")
            continue
        ok = int((pred[m] == c).sum())
        L.append(f"- **{c}**: {ok}/{int(m.sum())} "
                 f"({ok / m.sum():.4f})")
    L.append("")
    L.append("**Note:** J and Z are dynamic letters (motion), but this "
             "dataset provides single static frames only, so these numbers "
             "do not measure motion recognition.")
    L.append("")
    L.append("## Leakage check (nearest-train distance)")
    L.append("")
    L.append(f"- Reference set: {len(Xtr)} detected TRAIN samples; query: "
             f"{len(dmin)} detected TEST samples; exact brute-force "
             "Euclidean NN in the normalized 63-d feature space.")
    L.append(f"- Median nearest-train distance: **{med:.4f}** "
             "(0 = identical landmark geometry).")
    L.append(f"- Accuracy on nearest 25% ({k} samples): "
             f"**{acc_near:.4f}**")
    L.append(f"- Accuracy on farthest 25% ({k} samples): "
             f"**{acc_far:.4f}**")
    L.append(f"- Gap (near - far): **{acc_near - acc_far:+.4f}**")
    L.append("")
    L.append("**How to read this:** Kaggle ASL Alphabet is burst-shot "
             "video, so consecutive frames per subject are near-"
             "duplicates; the stratified split puts such siblings into "
             "both train and test. A large positive gap (near >> far) "
             "means the score is inflated by near-duplicate frames "
             "leaking across the split, and true generalization to new "
             "hands/positions is closer to the far-quartile number. A "
             "small gap suggests the model generalizes rather than "
             "memorizes neighbors.")
    L.append("")

    md = os.path.join(res_dir, "compare_v2.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"saved -> {md}")
    print(f"coverage={cov_all:.1%} excl_nothing={cov_excl:.1%} "
          f"acc_det={acc_det:.4f} s1={acc_s1} s2={acc_s2} "
          f"policyA={acc_a:.4f} abstain={abstain_rate:.1%} "
          f"med_dist={med:.4f} near={acc_near:.4f} far={acc_far:.4f}")


if __name__ == "__main__":
    main()
