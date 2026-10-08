"""Extract MediaPipe hand landmarks from an ImageFolder-style dataset
(root/<class_name>/*.jpg) into a CSV of features + label.

Works for both the ASL Alphabet Kaggle dataset (prototype) and a
self-collected VSL alphabet dataset later, as long as it's organized
the same way: one subfolder per class, images inside.

Usage:
    python extract_landmarks.py <dataset_root> <output_csv> [--limit-per-class N]
"""

import argparse
import csv
import os
import sys
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
N_LANDMARKS = 21  # MediaPipe hand model: 21 points/hand


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand landmark model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def build_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    # num_hands=1: fingerspelling/alphabet signs here are one-handed.
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    return vision.HandLandmarker.create_from_options(options)


def extract_one(detector, image_path):
    """Return a flat list of 63 floats (21 landmarks x x,y,z), or None
    if no hand was detected in the image."""
    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)
    if not result.hand_landmarks:
        return None
    landmarks = result.hand_landmarks[0]  # first detected hand
    feats = []
    for lm in landmarks:
        feats.extend([lm.x, lm.y, lm.z])
    return feats


def iter_class_dirs(dataset_root):
    classes = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )
    if not classes:
        sys.exit(f"No class subfolders found under {dataset_root}")
    return classes


def run(dataset_root, output_csv, limit_per_class=None, detector=None):
    """Core loop, separated from main() so it can be unit-tested with a
    fake detector (no real MediaPipe/model needed)."""
    classes = iter_class_dirs(dataset_root)
    n_written = 0
    n_skipped = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(3 * N_LANDMARKS)] + ["label"])

        for cls in classes:
            cls_dir = os.path.join(dataset_root, cls)
            files = sorted(
                fn for fn in os.listdir(cls_dir)
                if os.path.splitext(fn)[1].lower() in IMAGE_EXTS
            )
            if limit_per_class:
                files = files[:limit_per_class]

            for fn in files:
                feats = extract_one(detector, os.path.join(cls_dir, fn))
                if feats is None:
                    n_skipped += 1
                    continue
                writer.writerow(feats + [cls])
                n_written += 1

            print(f"{cls}: done ({len(files)} images)")

    print(f"Wrote {n_written} rows, skipped {n_skipped} (no hand detected) -> {output_csv}")
    return n_written, n_skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path to ImageFolder-style dataset (root/<class>/*.jpg)")
    parser.add_argument("output_csv", help="Where to write the extracted landmarks CSV")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Cap images per class (useful for a quick test run)")
    args = parser.parse_args()

    ensure_model()
    detector = build_detector()
    run(args.dataset_root, args.output_csv, args.limit_per_class, detector)


if __name__ == "__main__":
    main()
