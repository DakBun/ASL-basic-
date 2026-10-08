"""Collect labeled landmark samples from the webcam (MediaPipe v2 pipeline).

Per saved sample:
  * one row in <out>/landmarks.csv with columns session, person, hand_used,
    label, handedness_label, stage, f0..f62, timestamp (header created when
    the file does not exist, appended otherwise);
  * a 200x200 center-square crop of the UN-mirrored frame at
    <out>/images/<session>/<label>/<label>_<timestamp_ms>.jpg.

f0..f62 are the plain normalized 63 features: cascade(canon=False), so NO
canonicalization is baked in; handedness_label keeps MediaPipe's raw label
and training later decides whether to flip (see webcam_mp_v2.features).

Detection/drawing imported from webcam_mp_v2 (import-safe, main guard):
cascade, draw_hand, draw_hud + DATASET/MP_DIR constants. Preview is mirrored,
model input is the UN-mirrored frame -- same convention as webcam_mp_v2.py.

Keys: A-Z select a class (class U is taken by undo, step to it with arrows);
type del/space + Enter selects those (capture_external.py idea); LEFT/RIGHT
step the class list; SPACE toggles a burst (15 attempts, 0.2 s apart, only
detected frames saved, no-hand frames skipped and counted); U undoes the last
burst; Q quits (releases camera, closes windows).

Classes: A-Z, del, space (no "nothing" -- only detected frames are saved).
J and Z are motion letters; a single frame stores the static handshape only.

Usage:
    python collect_landmarks.py --session s1 --hand right [--camera 0]
    python collect_landmarks.py --selftest --session t1 --hand right
"""
import argparse
import csv
import os
import re
import sys
import tempfile
import time
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import extract_landmarks_v2 as ext  # noqa: E402  (main guard, import-safe)
import webcam_mp_v2 as w2           # noqa: E402  (main guard, import-safe)

# --- constants (no config system) ---
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "space"]
OUT_DEFAULT = "E:/ASL/my_data"
CSV_NAME = "landmarks.csv"
FEATURES = [f"f{i}" for i in range(63)]
CSV_HEADER = ["session", "person", "hand_used", "label", "handedness_label",
              "stage"] + FEATURES + ["timestamp"]
TS_COL = CSV_HEADER.index("timestamp")  # 6 + 63
IMG = 200                    # saved crop is IMG x IMG
SEED = 42
N_SELFTEST = 3
# ponytail: ceiling = attempt-based burst can save fewer than 15 when no-hand
# frames are skipped. Upgrade path = keep sampling until 15 saves land.
BURST_FRAMES = 15             # attempts per burst
BURST_INTERVAL = 0.2          # seconds between attempts


def sanitize_session(tag):
    """Session becomes a directory name: keep it filesystem-safe."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag or ""):
        raise SystemExit("--session must be non-empty and use only letters, "
                         "digits, '-' or '_' (e.g. s1_warmup)")
    return tag


def center_square(shape):
    """(x0, y0, side) of the centered square used for the 200x200 crop."""
    h, w = shape[:2]
    side = min(h, w)
    return (w - side) // 2, (h - side) // 2, side


def append_row(path, row):
    """Append one CSV row; write the header when the file is new or empty."""
    need_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


def load_counts(path, session):
    """Rows already saved by this session, per label (supports resuming)."""
    counts = Counter()
    if not os.path.isfile(path):
        return counts
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("session") == session and row.get("label"):
                counts[row["label"]] += 1
    return counts


def save_sample(raw, feat, stage, hand_label, label, ctx):
    """Write image + CSV row for one detected frame; None = skipped."""
    ts = int(time.time() * 1000)
    img_dir = os.path.join(ctx["out"], "images", ctx["session"], label)
    try:
        os.makedirs(img_dir, exist_ok=True)
        x0, y0, side = center_square(raw.shape)
        crop = raw[y0:y0 + side, x0:x0 + side]
        crop = cv2.resize(crop, (IMG, IMG), interpolation=cv2.INTER_AREA)
        img_path = os.path.join(img_dir, f"{label}_{ts}.jpg")
        if not cv2.imwrite(img_path, crop):
            print(f"WARNING: could not write {img_path} (sample skipped)")
            return None
    except OSError as e:
        print(f"WARNING: image save failed ({e}) (sample skipped)")
        return None
    row = [ctx["session"], ctx["args"].person, ctx["args"].hand, label,
           hand_label or "none", stage] + [float(v) for v in feat] + [ts]
    append_row(ctx["csv"], row)
    ctx["counts"][label] += 1
    return {"path": img_path, "ts": str(ts)}


def undo_burst(ctx, entries):
    """Remove the last burst's rows from the CSV and delete its images."""
    if not entries:
        print("undo: nothing to undo")
        return
    ts_set = {e["ts"] for e in entries}
    removed = deleted = 0
    if os.path.isfile(ctx["csv"]):
        with open(ctx["csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        keep = []
        for row in rows:
            hit = (len(row) == len(CSV_HEADER) and row[0] == ctx["session"]
                   and row[TS_COL] in ts_set)
            if hit:
                removed += 1
            else:
                keep.append(row)
        # ponytail: ceiling = full CSV rewrite, undo covers only the last
        # burst. Upgrade path = journaled appends / delete-by-id store.
        with open(ctx["csv"], "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(keep)
    for e in entries:
        try:
            os.remove(e["path"])
            deleted += 1
        except OSError:
            pass
    ctx["counts"] = load_counts(ctx["csv"], ctx["session"])  # resync
    print(f"undo: removed {removed} row(s), deleted {deleted} image(s)")


def on_key(key, state):
    """Mutate selection state from one keypress; returns an action or None."""
    if key in (ord("q"), ord("Q")):
        return "quit"
    if key in (ord("u"), ord("U")):
        # ponytail: ceiling = U cannot double as the letter-U selector (OpenCV
        # cannot distinguish Shift/Caps), so class U is reached with arrows.
        # Upgrade path = modifier-aware input or a separate undo key.
        return "undo"
    if key == 32:  # SPACE
        return "burst"
    if key in (2424832, 81, ord("[")):  # LEFT
        state["idx"] = (state["idx"] - 1) % len(CLASSES)
        state["buffer"] = ""
        return "step"
    if key in (2555904, 83, ord("]")):  # RIGHT
        state["idx"] = (state["idx"] + 1) % len(CLASSES)
        state["buffer"] = ""
        return "step"
    if key in (13, 10):  # Enter: commit typed del/space (or a lone letter)
        name = state["buffer"].strip().lower()
        state["buffer"] = ""
        names = [c.lower() for c in CLASSES]
        if name in names:
            state["idx"] = names.index(name)
        elif name:
            print(f"unknown class '{name}' (valid: A-Z, del, space)")
        return None
    if key in (8, 127):  # Backspace
        state["buffer"] = state["buffer"][:-1]
        return None
    if key == 27:  # Esc clears the buffer
        state["buffer"] = ""
        return None
    if 32 < key < 127 and chr(key).isalpha():  # letter (u/U handled above)
        if not state["buffer"]:
            state["idx"] = CLASSES.index(chr(key).upper())
        state["buffer"] = (state["buffer"] + chr(key).lower())[-12:]
        state["typed_at"] = time.monotonic()
    return None


def selftest(args):
    """Round-trip 3 split_test.csv samples through cascade()+CSV; no my_data."""
    if not os.path.isfile(ext.MODEL_PATH):
        raise SystemExit(f"hand landmarker task not found: {ext.MODEL_PATH}")
    model = os.path.join(w2.MP_DIR, w2.MODEL_NAME)
    if not os.path.isfile(model):
        raise SystemExit(f"model not found: {model} (run train_mp_v2.py)")
    rows = ext.load_split(w2.MP_DIR, "test")  # split_test.csv (relpath,label)
    idx = np.random.default_rng(SEED).choice(len(rows), N_SELFTEST,
                                             replace=False)
    sample = [rows[i] for i in idx]
    tmp = os.path.join(tempfile.gettempdir(), "collect_landmarks_selftest.csv")
    if os.path.isfile(tmp):
        os.remove(tmp)  # fresh file -> header written by append_row
    expected, detected = [], 0
    det1 = ext.build_detector(ext.CONF_STAGE1)
    det2 = ext.build_detector(ext.CONF_STAGE2)
    try:
        for relpath, label in sample:
            path = os.path.join(w2.DATASET, relpath)
            img = cv2.imread(path) if os.path.isfile(path) else None
            # canon=False: stored features must NOT be canonicalized
            feat, stage, hand, _ = w2.cascade(img, det1, det2, canon=False)
            if feat is None:
                feat = [0.0] * 63  # keep the 63-column contract (stage 0)
            else:
                detected += 1
            row = [args.session, args.person, args.hand, label,
                   hand or "none", stage] + [float(v) for v in feat] + \
                  [int(time.time() * 1000)]
            append_row(tmp, row)
            expected.append(label)
            print(f"  {relpath}: stage={stage} hand={hand or 'none'} "
                  f"label={label}")
    finally:
        det1.close()
        det2.close()
    with open(tmp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        got = list(reader)
    feat_cols = [c for c in fields if c.startswith("f") and c[1:].isdigit()]
    numeric = True
    for r in got:
        try:
            [float(r[c]) for c in FEATURES]
        except (KeyError, TypeError, ValueError):
            numeric = False
    checks = [
        (f"rows == {N_SELFTEST}", len(got) == N_SELFTEST),
        (f"feature columns == 63 (got {len(feat_cols)})",
         len(feat_cols) == 63 and all(c in fields for c in FEATURES)),
        (f"labels match {expected}",
         [r.get("label") for r in got] == expected),
        ("feature values numeric", numeric),
    ]
    print(f"selftest: wrote {len(got)} rows -> {tmp} "
          f"(detected {detected}/{N_SELFTEST})")
    ok = True
    for name, passed in checks:
        print(f"selftest: {name}: {'OK' if passed else 'FAIL'}")
        ok = ok and passed
    if not ok:
        raise SystemExit("selftest FAIL")
    print("selftest OK")


def run_live(args):
    """Mirrored preview, UN-mirrored model input (same as webcam_mp_v2)."""
    if not os.path.isfile(ext.MODEL_PATH):
        raise SystemExit(f"hand landmarker task not found: {ext.MODEL_PATH}")
    model = os.path.join(w2.MP_DIR, w2.MODEL_NAME)
    if not os.path.isfile(model):
        raise SystemExit(f"model not found: {model} (run train_mp_v2.py)")
    session = sanitize_session(args.session)
    out = os.path.abspath(args.out)
    try:
        os.makedirs(out, exist_ok=True)
        probe = os.path.join(out, ".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        raise SystemExit(f"output dir not writable: {out} ({e})")
    csv_file = os.path.join(out, CSV_NAME)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera} "
                         "(plugged in? in use by another app?)")
    det1 = det2 = None
    try:
        det1 = ext.build_detector(ext.CONF_STAGE1)
        det2 = ext.build_detector(ext.CONF_STAGE2)
        ctx = {"out": out, "csv": csv_file, "session": session,
               "counts": load_counts(csv_file, session), "args": args}
        state = {"idx": 0, "buffer": "", "typed_at": 0.0}
        bursting, attempts, saved, skipped = False, 0, 0, 0
        next_due = 0.0
        burst_entries, last_burst = [], []
        print(f"collecting: session={session} person={args.person} "
              f"hand={args.hand} out={out}")
        print("keys: A-Z class | type del/space+Enter | arrows step | "
              "SPACE burst | U undo | Q quit")
        # ponytail: ceiling = fixed centered square at capture resolution.
        # Upgrade path = request 720p+ and an adjustable ROI.
        while True:
            ok, raw = cap.read()
            if not ok:
                print("camera read failed, quitting")
                break
            now = time.monotonic()
            display = cv2.flip(raw, 1)   # preview is mirrored
            feed = raw                   # model input is NOT mirrored
            feat, stage, hand, _, lm = w2.cascade(
                feed, det1, det2, canon=False, return_lm=True)
            if bursting and now >= next_due:  # burst attempt on this frame
                next_due = now + BURST_INTERVAL
                attempts += 1
                if feat is None:
                    skipped += 1
                else:
                    entry = save_sample(raw, feat, stage, hand,
                                        CLASSES[state["idx"]], ctx)
                    if entry:
                        burst_entries.append(entry)
                        saved += 1
                if attempts >= BURST_FRAMES:
                    bursting = False
                    if burst_entries:
                        last_burst = burst_entries
                    burst_entries = []
                    print(f"burst done: saved {saved}, skipped {skipped} "
                          f"(attempts {attempts})")
            x0, y0, side = center_square(raw.shape)
            cv2.rectangle(display, (x0, y0), (x0 + side, y0 + side),
                          (0, 255, 255), 2)  # keep the hand inside this box
            if feat is not None and lm is not None:
                w2.draw_hand(display, lm, stage, feed_flip=False)
            label = CLASSES[state["idx"]]
            lines = [
                f"target: {label}   saved this session: "
                f"{ctx['counts'][label]}",
                f"session: {session}  person: {args.person}  hand: {args.hand}",
                f"stage: {stage}  handedness: {hand or 'none'}",
                (f"burst {attempts}/{BURST_FRAMES}  saved {saved}  "
                 f"skipped {skipped}") if bursting else "burst: off",
                "A-Z class | type del/space + Enter | arrows step",
                "SPACE burst | U undo | Q quit",
            ]
            w2.draw_hud(display, lines, label, feat is not None)
            cv2.imshow("collect_landmarks", display)
            action = on_key(cv2.waitKeyEx(1), state)
            if action == "quit":
                break
            if action == "burst":
                if bursting:
                    bursting = False
                    if burst_entries:
                        last_burst = burst_entries
                    burst_entries = []
                    print(f"burst stopped: saved {saved}, skipped {skipped}")
                else:
                    bursting, attempts, saved, skipped = True, 0, 0, 0
                    burst_entries = []
                    next_due = now  # first attempt on the next frame
                    print(f"burst start: {BURST_FRAMES} attempts, "
                          f"{BURST_INTERVAL}s apart, no-hand frames skipped")
            elif action == "undo":
                if bursting:  # stop the burst; its saves become undoable
                    bursting = False
                    if burst_entries:
                        last_burst = burst_entries
                    burst_entries = []
                undo_burst(ctx, last_burst)
                last_burst = []
            # clear a lone letter buffer after a pause (capture_external idea)
            if (state["buffer"] and len(state["buffer"]) == 1
                    and now - state["typed_at"] > 0.7):
                state["buffer"] = ""
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if det1:
            det1.close()
        if det2:
            det2.close()


def main():
    parser = argparse.ArgumentParser(
        description="Collect labeled landmark samples (MediaPipe v2 pipeline)")
    parser.add_argument("--session", required=True,
                        help="session tag; also the images subfolder name")
    parser.add_argument("--person", default="me", help="who is signing")
    parser.add_argument("--hand", required=True, choices=("left", "right"),
                        help="the hand you are actually using")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", default=OUT_DEFAULT,
                        help="output root (git-ignored)")
    parser.add_argument("--selftest", action="store_true",
                        help="offline CSV round-trip test; writes only to temp")
    args = parser.parse_args()
    if args.selftest:
        selftest(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()