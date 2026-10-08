"""Train small classifiers on MediaPipe landmark features (detected only).

Trains RandomForest and a small MLP, picks the better by val accuracy,
saves the winner with joblib. Inference rule: no hand -> "nothing".

Usage:
    python train_mp.py [--dir E:/ASL/mp_experiment]
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

CLASSES = sorted(
    [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["del", "nothing", "space"]
)


def load_split(directory, split):
    path = os.path.join(directory, f"features_{split}.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"features not found: {path} (run extract_landmarks.py)")
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["features"], dtype=np.float64)
    det = np.asarray(d["detected"], dtype=bool)
    y = np.asarray(d["label"]).astype(str)
    if len(X) == 0:
        raise SystemExit(f"features file is empty: {path}")
    return X, det, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    args = parser.parse_args()

    Xtr, dtr, ytr = load_split(args.dir, "train")
    Xva, dva, yva = load_split(args.dir, "val")

    Xtr, ytr = Xtr[dtr], ytr[dtr]
    Xva, yva = Xva[dva], yva[dva]
    print(f"train detected: {len(ytr)}, val detected: {len(yva)}")
    if len(ytr) == 0 or len(yva) == 0:
        raise SystemExit("no detected samples to train on")

    # ponytail: ceiling = RF(300 trees) + MLP(128,64) on 63 raw features, no tuning.
    # Upgrade path = GridSearchCV + feature engineering (angles/bone lengths).
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_acc = accuracy_score(yva, rf.predict(Xva))
    print(f"RandomForest val acc: {rf_acc:.4f}")

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                        random_state=42)
    mlp.fit(Xtr, ytr)
    mlp_acc = accuracy_score(yva, mlp.predict(Xva))
    print(f"MLP val acc: {mlp_acc:.4f}")

    if mlp_acc >= rf_acc:
        best, name, acc = mlp, "mlp", mlp_acc
    else:
        best, name, acc = rf, "randomforest", rf_acc

    out = os.path.join(args.dir, f"mp_model_{name}.joblib")
    joblib.dump({"model": best, "classes": CLASSES,
                 "val_accuracy": float(acc)}, out)
    print(f"winner: {name} ({acc:.4f}) -> {out}")


if __name__ == "__main__":
    main()
