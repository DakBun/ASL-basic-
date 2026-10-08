"""Evaluate the hosted ResNet50 on the FULL test split, with resume support.

Caches every prediction to resnet_preds_test.csv. On re-run, rows that already
have a successful prediction are skipped; rows with empty pred/error are retried.
Retry with exponential backoff; a run crash never loses finished calls.

Usage:
    python eval_resnet.py [--dataset E:/ASL/kaggle_sample] [--dir E:/ASL/mp_experiment]
"""
import argparse
import csv
import os
import time

MODEL_ID = "sign-language-2c6kk/2"  # same hosted model as webcam_demo.py
MAX_RETRIES = 4
BACKOFF_BASE = 2.0  # ponytail: ceiling = linear schedule, no jitter.
                    # Upgrade path = Retry-After header parsing + jitter.


def load_test_rows(directory):
    path = os.path.join(directory, "split_test.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"split file not found: {path} (run make_split.py)")
    with open(path, newline="") as f:
        rows = [(r["relpath"], r["label"]) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"split file is empty: {path}")
    return rows


def load_done(pred_csv):
    """Return relpath -> row for rows with a SUCCESSFUL prediction only."""
    done = {}
    if os.path.isfile(pred_csv):
        with open(pred_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("pred") and not r.get("error"):
                    done[r["relpath"]] = r
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="E:/ASL/kaggle_sample")
    parser.add_argument("--dir", default="E:/ASL/mp_experiment")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from inference_sdk import InferenceHTTPClient

    load_dotenv("E:/ASL/.env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY not found (check E:/ASL/.env)")

    rows = load_test_rows(args.dir)
    pred_csv = os.path.join(args.dir, "resnet_preds_test.csv")
    done = load_done(pred_csv)
    print(f"test rows: {len(rows)}, already cached: {len(done)}")

    client = InferenceHTTPClient(api_url="https://classify.roboflow.com",
                                 api_key=api_key)
    new_file = not os.path.isfile(pred_csv)
    f = open(pred_csv, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=["relpath", "label", "pred",
                                           "confidence", "ms", "error"])
    if new_file:
        writer.writeheader()

    n_new = 0
    try:
        for i, (relpath, label) in enumerate(rows):
            if relpath in done:
                continue
            abs_path = os.path.join(args.dataset, relpath)
            if not os.path.isfile(abs_path):
                writer.writerow({"relpath": relpath, "label": label,
                                 "pred": "", "confidence": "",
                                 "ms": "", "error": "missing_file"})
                continue
            # path string input (BufferedReader is rejected by the SDK);
            # exponential backoff on transient failures.
            last_err = ""
            for attempt in range(MAX_RETRIES):
                t0 = time.perf_counter()
                try:
                    # ponytail: ceiling = one blocking infer() per image over
                    # hosted API. Upgrade path = batch/parallel requests +
                    # local checkpoint eval.
                    result = client.infer(abs_path, model_id=MODEL_ID)
                    ms = (time.perf_counter() - t0) * 1000.0
                    writer.writerow({"relpath": relpath, "label": label,
                                     "pred": result.get("top", ""),
                                     "confidence": result.get("confidence", ""),
                                     "ms": f"{ms:.1f}", "error": ""})
                    last_err = ""
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if attempt < MAX_RETRIES - 1:
                        wait = BACKOFF_BASE * (2 ** attempt)
                        print(f"[{i + 1}/{len(rows)}] retry {attempt + 1} "
                              f"{relpath} after {wait:.0f}s: {e}")
                        time.sleep(wait)
            if last_err:
                ms = (time.perf_counter() - t0) * 1000.0
                writer.writerow({"relpath": relpath, "label": label,
                                 "pred": "", "confidence": "",
                                 "ms": f"{ms:.1f}", "error": last_err})
                print(f"[{i + 1}/{len(rows)}] ERROR {relpath}: {last_err}")
                f.flush()
                continue
            n_new += 1
            if (i + 1) % 25 == 0:
                print(f"[{i + 1}/{len(rows)}] ok (new={n_new})")
                f.flush()
            time.sleep(0.1)  # be gentle to the hosted API
    finally:
        f.close()
    print(f"done: new={n_new}, total_cached_rows={len(done) + n_new} -> {pred_csv}")


if __name__ == "__main__":
    main()
