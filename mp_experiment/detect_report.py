"""Report MediaPipe detection rates by class for all splits and save
undetected TEST samples for visual inspection.

Reads the existing features_*.npz (never rewrites them).

Usage:
    python detect_report.py [--dataset E:/ASL/kaggle_sample] [--dir E:/ASL/mp_experiment]
"""
import argparse
import os
import shutil
from collections import Counter, defaultdict

import numpy as np

SPLITS = ("train", "val", "test")
MAX_SAMPLES = 24
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_split(directory, split):
    path = os.path.join(directory, f"features_{split}.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"features not found: {path} (run extract_landmarks.py)")
    d = np.load(path, allow_pickle=True)
    return (np.asarray(d["detected"], dtype=bool),
            np.asarray(d["label"]).astype(str),
            np.asarray(d["path"]).astype(str))


def rates_by_class(det, labels):
    """Return [(cls, hit, total, rate)] sorted ascending by rate (then class)."""
    total = Counter(labels.tolist())
    hit = Counter(labels[det].tolist())
    rows = [(c, hit[c], total[c], hit[c] / total[c]) for c in total]
    rows.sort(key=lambda r: (r[3], r[0]))
    return rows


def size_distribution(dataset_root):
    """(w,h) -> count over every image; unreadable images are counted, not raised."""
    from PIL import Image  # lazy import: only needed here

    sizes = Counter()
    n_unreadable = 0
    n_total = 0
    for cls in sorted(os.listdir(dataset_root)):
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if not fn.lower().endswith(IMAGE_EXTS):
                continue
            n_total += 1
            try:
                with Image.open(os.path.join(cls_dir, fn)) as im:
                    sizes[im.size] += 1
            except Exception:
                n_unreadable += 1
    return sizes, n_total, n_unreadable


def save_undetected(dataset_root, out_dir, det, labels, paths, test_rates):
    """Copy up to MAX_SAMPLES undetected test images, mixed classes,
    lowest-detection classes first (round-robin so classes stay mixed)."""
    undet = defaultdict(list)
    for relpath, label, d in zip(paths, labels, det):
        if not d:
            undet[label].append(relpath)
    order = [cls for cls, *_ in test_rates]  # ascending detection rate
    os.makedirs(out_dir, exist_ok=True)
    saved, missing, order_pos = [], 0, 0
    while len(saved) < MAX_SAMPLES and order_pos < len(order) * 4:
        cls = order[order_pos % len(order)]
        order_pos += 1
        if not undet[cls]:
            continue
        relpath = undet[cls].pop(0)
        src = os.path.join(dataset_root, relpath)
        if not os.path.isfile(src):
            missing += 1
            continue
        stem = os.path.splitext(os.path.basename(relpath))[0]
        ext = os.path.splitext(relpath)[1] or ".jpg"
        dst = os.path.join(out_dir, f"{cls}_{stem}{ext}")
        shutil.copy2(src, dst)
        saved.append(os.path.basename(dst))
        if len(saved) >= MAX_SAMPLES:
            break
    return saved, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="E:/ASL/kaggle_sample")
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    args = parser.parse_args()

    data = {s: load_split(args.dir, s) for s in SPLITS}

    L = ["# MediaPipe detection rate by class", ""]
    L.append("Settings: HandLandmarker (Tasks API, IMAGE mode), num_hands=1, "
             "min_hand_detection_confidence=0.5, no preprocessing — identical "
             "to `extract_landmarks.py`. Source: existing `features_*.npz` "
             "(not regenerated).")
    L.append("")

    for split in SPLITS:
        det, labels, _paths = data[split]
        rows = rates_by_class(det, labels)
        n_det = int(det.sum())
        L.append(f"## {split} — {n_det}/{len(det)} ({n_det / len(det):.1%})")
        L.append("")
        L.append("| class | detected / total | rate |")
        L.append("|---|---|---|")
        for cls, hit, total, rate in rows:
            L.append(f"| {cls} | {hit}/{total} | {rate:.1%} |")
        L.append("")

    # --- undetected TEST samples ---
    det_t, labels_t, paths_t = data["test"]
    test_rates = rates_by_class(det_t, labels_t)
    out_dir = os.path.join(args.dir, "results", "undetected_samples")
    saved, missing = save_undetected(args.dataset, out_dir, det_t, labels_t,
                                     paths_t, test_rates)
    classes_saved = sorted({s.split("_", 1)[0] for s in saved})
    L.append(f"## Undetected test samples saved: {len(saved)} "
             f"-> results/undetected_samples/")
    L.append("")
    L.append(f"Classes covered ({len(classes_saved)}): "
             + ", ".join(classes_saved))
    if missing:
        L.append(f"Missing on disk (skipped): {missing}")
    L.append("")

    # --- image size distribution ---
    sizes, n_total, n_unreadable = size_distribution(args.dataset)
    L.append(f"## Image size distribution (kaggle_sample, {n_total} images)")
    L.append("")
    L.append("| width x height | count |")
    L.append("|---|---|")
    for (w, h), n in sizes.most_common():
        L.append(f"| {w} x {h} | {n} |")
    if n_unreadable:
        L.append("")
        L.append(f"Unreadable (skipped): {n_unreadable}")
    L.append("")

    md = os.path.join(args.dir, "results", "detection_by_class.md")
    os.makedirs(os.path.dirname(md), exist_ok=True)
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"saved -> {md}")
    print(f"undetected samples saved: {len(saved)} (missing: {missing})")
    print(f"size distribution ({n_total} images, unreadable={n_unreadable}):")
    for (w, h), n in sizes.most_common():
        print(f"  {w} x {h}: {n}")


if __name__ == "__main__":
    main()
