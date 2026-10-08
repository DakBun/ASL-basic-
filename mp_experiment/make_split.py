"""Create a persistent stratified train/val/test split of kaggle_sample.

Saves one CSV per split: relpath,label  (relpath like "A/A1021.jpg").
Used by BOTH models so the comparison is on identical data.

Usage:
    python make_split.py [--dataset E:/ASL/kaggle_sample] [--out mp_experiment]
"""
import argparse
import csv
import os

from sklearn.model_selection import train_test_split

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

CLASSES = sorted(
    [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["del", "nothing", "space"]
)


def collect(dataset_root):
    """Return (paths, labels); validate dataset layout at the boundary."""
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"dataset root not found: {dataset_root}")
    paths, labels = [], []
    missing = [c for c in CLASSES if not os.path.isdir(os.path.join(dataset_root, c))]
    if missing:
        raise SystemExit(f"missing class folders under {dataset_root}: {missing}")
    for cls in CLASSES:
        cls_dir = os.path.join(dataset_root, cls)
        files = sorted(
            f for f in os.listdir(cls_dir) if f.lower().endswith(IMAGE_EXTS)
        )
        if not files:
            print(f"WARNING: class {cls} has no images, skipped")
            continue
        for fn in files:
            paths.append(os.path.join(cls, fn))  # relative, portable
            labels.append(cls)
    if not paths:
        raise SystemExit(f"no images found under {dataset_root}")
    return paths, labels


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relpath", "label"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="E:/ASL/kaggle_sample")
    parser.add_argument("--out", default="E:/ASL/mp_experiment")
    args = parser.parse_args()

    paths, labels = collect(args.dataset)

    train_p, tmp_p, train_y, tmp_y = train_test_split(
        paths, labels, train_size=TRAIN_RATIO, stratify=labels, random_state=SEED
    )
    val_p, test_p, val_y, test_y = train_test_split(
        tmp_p, tmp_y,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=tmp_y, random_state=SEED,
    )

    write_csv(os.path.join(args.out, "split_train.csv"),
              sorted(zip(train_p, train_y)))
    write_csv(os.path.join(args.out, "split_val.csv"),
              sorted(zip(val_p, val_y)))
    write_csv(os.path.join(args.out, "split_test.csv"),
              sorted(zip(test_p, test_y)))

    from collections import Counter
    for name, y in [("train", train_y), ("val", val_y), ("test", test_y)]:
        c = Counter(y)
        print(f"{name}: n={len(y)} per-class={min(c.values())}-{max(c.values())} "
              f"classes={len(c)}")


if __name__ == "__main__":
    main()
