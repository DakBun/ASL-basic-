"""Train a mirror-augmented MediaPipe classifier (v3); compare it with v2.

Mirror = negate the x columns of the 63-d normalized landmark vector (y and z
unchanged): a left hand is the mirror image of a right hand. The x columns are
0::3 -- verified in extract_landmarks_v2.normalize(), which returns
(centered / scale).reshape(-1) of the (21, 3) [x, y, z] array, so C-order
flattening puts the x of landmark i at index 3*i.

Train = detected train samples + mirrored copies (same labels); val likewise.
Test samples (original or mirrored) never touch training or selection. Model
is trained exactly like train_mp_v2.py (same hyperparameters, seed 42, same
MLP-vs-RF selection) and saved in the same dict format.

Usage:
    python train_mp_v3.py [--dir E:/ASL/mp_experiment]
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

CAVEAT = ("The mirrored test is synthetic: it simulates the opposite hand "
          "by negating x. It is not a substitute for real left-hand data.")
LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_split(directory, split):
    path = os.path.join(directory, f"features_v2_{split}.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"features not found: {path} "
                         "(run extract_landmarks_v2.py)")
    d = np.load(path, allow_pickle=True)
    missing = [k for k in ("features", "label", "detected") if k not in d.files]
    if missing:
        raise SystemExit(f"missing keys {missing} in {path}")
    X = np.asarray(d["features"], dtype=np.float64)
    det = np.asarray(d["detected"], dtype=bool)
    y = np.asarray(d["label"]).astype(str)
    if len(X) == 0:
        raise SystemExit(f"features file is empty: {path}")
    if X.ndim != 2 or X.shape[1] != 63:
        raise SystemExit(f"expected (N, 63) features in {path}, got {X.shape}")
    if not (len(det) == len(y) == len(X)):
        raise SystemExit(f"length mismatch in {path}: features={len(X)} "
                         f"detected={len(det)} label={len(y)}")
    return X, det, y


def mirror(X):
    """Negate the x columns (0::3); involution: mirror(mirror(X)) == X."""
    out = np.array(X, dtype=np.float64, copy=True)
    out[:, 0::3] = -out[:, 0::3]
    return out


def augment(X, y):
    return np.concatenate([X, mirror(X)]), np.concatenate([y, y])


def top_confusions(y_true, y_pred, k=10):
    """Top-k unordered letter pairs (A-Z) by symmetric confusion count."""
    pair = {}
    for t, p in zip(y_true, y_pred):
        if t != p and t in LETTERS and p in LETTERS:
            key = tuple(sorted((t, p)))
            pair[key] = pair.get(key, 0) + 1
    return sorted(pair.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    args = parser.parse_args()

    Xtr, dtr, ytr = load_split(args.dir, "train")
    Xva, dva, yva = load_split(args.dir, "val")
    Xte, dte, yte = load_split(args.dir, "test")
    Xtr, ytr = Xtr[dtr], ytr[dtr]
    Xva, yva = Xva[dva], yva[dva]
    Xte, yte = Xte[dte], yte[dte]
    if not (len(ytr) and len(yva) and len(yte)):
        raise SystemExit("no detected samples: "
                         f"train={len(ytr)} val={len(yva)} test={len(yte)}")
    print(f"detected: train={len(ytr)} val={len(yva)} test={len(yte)}")

    # sanity: 5 training samples mirrored twice must come back unchanged
    five = Xtr[:5]
    assert np.allclose(mirror(mirror(five)), five), "double mirror != original"
    print("mirror sanity: OK (5 train samples, mirrored twice == original)")

    # ponytail: ceiling = the mirror is a synthetic proxy for the other hand
    # (only x is negated; no real left-hand pose/skin/camera variation).
    # Upgrade path = real left-hand data via collect_landmarks.py.
    Xtra, ytra = augment(Xtr, ytr)
    Xvaa, yvaa = augment(Xva, yva)
    print(f"augmented: train={len(ytra)} val={len(yvaa)} (original + mirrored)")

    # identical to train_mp_v2.py: same classes, hyperparameters, seed, rule
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xtra, ytra)
    rf_acc = accuracy_score(yvaa, rf.predict(Xvaa))
    print(f"RandomForest val acc: {rf_acc:.4f}")

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                        random_state=42)
    mlp.fit(Xtra, ytra)
    mlp_acc = accuracy_score(yvaa, mlp.predict(Xvaa))
    print(f"MLP val acc: {mlp_acc:.4f}")

    if mlp_acc >= rf_acc:
        best, name, acc = mlp, "mlp", mlp_acc
    else:
        best, name, acc = rf, "randomforest", rf_acc

    out = os.path.join(args.dir, "mp_model_v3.joblib")
    joblib.dump({"model": best, "kind": name, "classes": sorted(set(ytra)),
                 "val_accuracy": float(acc)}, out)
    print(f"winner: {name} ({acc:.4f}) -> {out}")

    p2 = os.path.join(args.dir, "mp_model_v2.joblib")
    if not os.path.isfile(p2):
        raise SystemExit(f"model not found: {p2} (run train_mp_v2.py)")
    m2, m3 = joblib.load(p2), joblib.load(out)

    # 2 models x {original, mirrored} detected test; same samples either way
    Xte_m = mirror(Xte)
    combos = (("v2 original test", m2, Xte), ("v2 mirrored test", m2, Xte_m),
              ("v3 original test", m3, Xte), ("v3 mirrored test", m3, Xte_m))
    accs, rows = {}, []
    for label, blob, X in combos:
        pred = blob["model"].predict(X)
        correct = int((pred == yte).sum())
        accs[label] = correct / len(yte)
        rows.append((label, accs[label], correct, len(yte)))
        print(f"{label}: acc={accs[label]:.4f} ({correct}/{len(yte)})")

    a2o, a3o = accs["v2 original test"], accs["v3 original test"]
    delta = a3o - a2o
    regressed = delta < -0.02
    if regressed:
        print(f"REGRESSION: v3 on original test is {-delta:.4f} lower than "
              f"v2 ({a3o:.4f} vs {a2o:.4f}) -- more than 0.02")

    # per-class accuracy on the mirrored test, both models
    pred2m = m2["model"].predict(Xte_m)
    pred3m = m3["model"].predict(Xte_m)
    per_class = []
    print("per-class accuracy on MIRRORED test (class n v2 v3):")
    for c in sorted(set(yte)):
        mask = yte == c
        n = int(mask.sum())
        a2 = float((pred2m[mask] == yte[mask]).mean())
        a3 = float((pred3m[mask] == yte[mask]).mean())
        per_class.append((c, n, a2, a3))
        print(f"  {c}: n={n} v2={a2:.4f} v3={a3:.4f}")

    # top-10 confused letter pairs for v3 on the original test
    top = top_confusions(yte, m3["model"].predict(Xte), 10)
    print("v3 original test top-10 confused letter pairs:")
    for (a, b), n in top:
        print(f"  {a} <-> {b}: {n}")

    md = ["# MediaPipe v3 (mirror-augmented) vs v2 -- TEST split comparison",
          "", CAVEAT, ""]
    md.append(f"- Detected test samples: {len(yte)}; the mirrored test is "
              "the same set with x negated (counts identical by construction).")
    md.append(f"- Train {len(ytr)} -> {len(ytra)} after mirror augmentation, "
              f"val {len(yva)} -> {len(yvaa)}; test never used for training "
              "or selection.")
    md.append("- Same recipe as train_mp_v2.py: RandomForest(n_estimators=300, "
              "random_state=42, n_jobs=-1) and MLPClassifier"
              "(hidden_layer_sizes=(128, 64), max_iter=800, random_state=42); "
              "winner = higher val accuracy, MLP wins ties.")
    md.append(f"- v2: kind={m2['kind']}, val_accuracy={m2['val_accuracy']:.4f}; "
              f"v3: kind={m3['kind']}, val_accuracy={m3['val_accuracy']:.4f} "
              "(v3 val = original + mirrored).")
    md.append("- Sanity check: 5 training samples mirrored twice == original: "
              "**OK**.")
    md.append("")
    md.append("## Four accuracy combinations (detected test)")
    md.append("")
    md.append("| model | test | accuracy | correct/n |")
    md.append("|---|---|---|---|")
    for label, a, c, n in rows:
        model, which = label.split(" ", 1)
        md.append(f"| {model} | {which} | {a:.4f} | {c}/{n} |")
    md.append("")
    md.append(f"- v3 - v2 on the ORIGINAL test: **{delta:+.4f}**")
    if regressed:
        md.append(f"\n**WARNING: v3 on the original test ({a3o:.4f}) is "
                  f"{-delta:.4f} LOWER than v2 ({a2o:.4f}) -- a drop of more "
                  "than 0.02. Mirror augmentation hurt accuracy on the "
                  "original test set; do not treat v3 as an upgrade over v2 "
                  "on this evidence.**")
    md.append("")
    md.append("## Per-class accuracy on the MIRRORED test")
    md.append("")
    md.append("| class | n | v2 | v3 |")
    md.append("|---|---|---|---|")
    for c, n, a2, a3 in per_class:
        md.append(f"| {c} | {n} | {a2:.4f} | {a3:.4f} |")
    md.append("")
    md.append("## Top 10 confused letter pairs (v3, original test; A-Z, "
              "unordered)")
    md.append("")
    if top:
        md.extend(f"- {a} <-> {b}: {n}" for (a, b), n in top)
    else:
        md.append("- none")
    md.append("")
    md.append("## ponytail")
    md.append("")
    md.append("- ceiling: the mirror is a synthetic proxy for the other hand "
              "(x negated only). Upgrade path: real left-hand data via "
              "collect_landmarks.py.")
    md.append("")
    md.append("## Reproduce")
    md.append("")
    md.append("```")
    md.append("python -m py_compile train_mp_v3.py")
    md.append("python train_mp_v3.py --dir E:/ASL/mp_experiment")
    md.append("```")
    rep = os.path.join(args.dir, "results", "compare_v3.md")
    os.makedirs(os.path.dirname(rep), exist_ok=True)
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"report -> {rep}")


if __name__ == "__main__":
    main()