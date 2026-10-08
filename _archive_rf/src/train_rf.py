"""Train a Random Forest classifier on hand-landmark features extracted by
extract_landmarks.py.

Usage:
    python train_rf.py <landmarks_csv> [--model-out model.joblib]
"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("landmarks_csv")
    parser.add_argument("--model-out", default="rf_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.landmarks_csv)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred))

    labels_sorted = sorted(y.unique())
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels_sorted),
        index=labels_sorted,
        columns=labels_sorted,
    )
    cm_path = args.landmarks_csv.replace(".csv", "_confusion_matrix.csv")
    cm.to_csv(cm_path)
    print(f"Confusion matrix saved -> {cm_path}")

    joblib.dump(clf, args.model_out)
    print(f"Model saved -> {args.model_out}")


if __name__ == "__main__":
    main()
