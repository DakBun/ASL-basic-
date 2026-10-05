"""Sample N random images per class from the downloaded Kaggle ASL Alphabet
dataset into a smaller folder, ready to upload to Roboflow.

Usage:
    python sample_kaggle_asl.py <kaggle_dataset_root> <output_dir> [--per-class N]
"""
import argparse
import os
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path to asl_alphabet_train folder (root/<class>/*.jpg)")
    parser.add_argument("output_dir", help="Where to write the sampled subset")
    parser.add_argument("--per-class", type=int, default=150, help="How many images to sample per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    classes = sorted(
        d for d in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, d))
    )
    if not classes:
        raise SystemExit(f"No class subfolders found under {args.dataset_root}")

    for cls in classes:
        src_dir = os.path.join(args.dataset_root, cls)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        sample = random.sample(files, min(args.per_class, len(files)))

        dst_dir = os.path.join(args.output_dir, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in sample:
            shutil.copy2(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))

        print(f"{cls}: copied {len(sample)} / {len(files)} images")

    print(f"Done -> {args.output_dir}")


if __name__ == "__main__":
    main()
