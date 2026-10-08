"""Webcam burst-capture tool for out-of-dataset evaluation images.

Opens an OpenCV preview with a centered crop square (same framing as the
200x200 Kaggle images — put your hand inside the square). Keyboard:
  A-Z                              jump to that class immediately
  type del / space / nothing + Enter   select a special class
  Backspace / Esc                  edit / clear the typed buffer
  LEFT / RIGHT (or [ ])            step through the 29 classes in order
  SPACE                            start / stop a burst
  Q (or Esc on empty buffer)       quit

A burst saves N frames (default 10, one every ~0.3 s) to
eval_external/<class>/<class>_<session>_<idx>.jpg. Each frame is cropped
to a centered square first, then resized to 200x200 (INTER_AREA).

Note for J and Z: capture the final handshape only (static frames).
Motion (the J hook, the Z trace) is out of scope here.

Usage:
    python capture_external.py --session bg1_light [--n 10] [--interval 0.3] [--cam 0]
"""
import argparse
import os
import re
import time

try:
    import cv2
except ImportError:
    raise SystemExit("OpenCV (cv2) is not installed. Install it first, e.g.: "
                     "pip install opencv-python")

CLASSES = sorted(
    [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["del", "nothing", "space"]
)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MAX_FAILS = 30  # consecutive bad frames before giving up
FONT = cv2.FONT_HERSHEY_SIMPLEX


def sanitize_session(tag):
    tag = (tag or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag or ""):
        raise SystemExit("--session must be non-empty and use only letters, "
                         "digits, '-' or '_' (e.g. bg1_light)")
    return tag


def class_saved_counts(out_root):
    """Existing jpg counts per class (missing dirs count as 0)."""
    counts = {c: 0 for c in CLASSES}
    for c in CLASSES:
        d = os.path.join(out_root, c)
        if os.path.isdir(d):
            counts[c] = sum(1 for f in os.listdir(d)
                            if f.lower().endswith(IMAGE_EXTS))
    return counts


def next_index(cls_dir, cls, session):
    """Next free idx for this class+session pair (never overwrites)."""
    best = -1
    if os.path.isdir(cls_dir):
        for fn in os.listdir(cls_dir):
            stem = os.path.splitext(fn)[0]
            parts = stem.split("_")
            if (len(parts) >= 3 and parts[0] == cls
                    and "_".join(parts[1:-1]) == session
                    and parts[-1].isdigit()):
                best = max(best, int(parts[-1]))
    return best + 1


def center_square(frame):
    """Return (crop, x0, y0, side) of the centered square region."""
    h, w = frame.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return frame[y0:y0 + side, x0:x0 + side], x0, y0, side


def draw_hud(frame, x0, y0, side, cls, saved, session, buffer, bursting,
             burst_i, burst_n):
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
    pad = 12
    lines = [
        f"class: {cls}  (saved: {saved})",
        f"session: {session}",
        f"type: {buffer}" if buffer else "type A-Z / del / space / nothing",
        "SPACE burst | LEFT/RIGHT step | Q quit",
    ]
    if bursting:
        lines.append(f"BURST {burst_i + 1}/{burst_n}  REC")
    y = 28
    for i, text in enumerate(lines):
        color = (0, 0, 255) if (bursting and i == len(lines) - 1) else (0, 255, 0)
        cv2.putText(frame, text, (pad, y), FONT, 0.7, color, 2, cv2.LINE_AA)
        y += 30


def on_key(key, state):
    """Update selection state from one keypress. Returns (action, idx, buf)."""
    idx, buffer = state["idx"], state["buffer"]
    if key in (ord("q"), ord("Q")):
        return "quit", idx, ""
    if key == 32:  # SPACE: burst toggle handled by caller
        return "burst", idx, ""
    if key in (2424832, 81, ord("[")):  # LEFT
        return None, (idx - 1) % len(CLASSES), ""
    if key in (2555904, 83, ord("]")):  # RIGHT
        return None, (idx + 1) % len(CLASSES), ""
    if key in (13, 10):  # Enter: commit buffer
        name = buffer.strip().lower()
        names = [c.lower() for c in CLASSES]
        if name in names:
            return None, names.index(name), ""
        if name:
            print(f"unknown class '{buffer}' (valid: A-Z, del, space, nothing)")
        return None, idx, ""
    if key in (8, 127):  # Backspace
        return None, idx, buffer[:-1]
    if key == 27:  # Esc: clear buffer
        return None, idx, ""
    if 32 < key < 127 and chr(key).isalpha():  # typed letter
        if buffer == "" and chr(key).upper() in CLASSES:
            idx = CLASSES.index(chr(key).upper())
        buffer = (buffer + chr(key).lower())[-12:]
        state["typed_at"] = time.monotonic()
        return None, idx, buffer
    return None, idx, buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True,
                        help="e.g. bg1_light (letters/digits/-/_ only)")
    parser.add_argument("--n", type=int, default=10,
                        help="frames per burst")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="seconds between burst frames")
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()
    session = sanitize_session(args.session)
    if args.n <= 0 or args.interval <= 0:
        raise SystemExit("--n and --interval must be positive")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.cam} "
                         "(check it is plugged in / not used by another app)")
    # ponytail: ceiling = default capture size, centered square crop only.
    # Upgrade path = request 720p+ and a movable/resizeable crop box.

    out_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "eval_external"))
    counts = class_saved_counts(out_root)

    state = {"idx": 0, "buffer": "", "typed_at": 0.0}
    bursting, burst_i, next_due, fails = False, 0, 0.0, 0

    print("keys: A-Z jump | type del/space/nothing + Enter | "
          "LEFT/RIGHT step | SPACE burst | Q quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= MAX_FAILS:
                    print(f"camera gave {MAX_FAILS} bad frames in a row, "
                          "quitting (this burst step saved nothing)")
                    break
                continue
            fails = 0

            crop, x0, y0, side = center_square(frame)
            cls = CLASSES[state["idx"]]
            draw_hud(frame, x0, y0, side, cls, counts[cls], session,
                     state["buffer"], bursting, burst_i, args.n)
            cv2.imshow("capture_external (hand inside the green square)",
                       frame)

            now = time.monotonic()
            if bursting and now >= next_due:
                small = cv2.resize(crop, (200, 200),
                                   interpolation=cv2.INTER_AREA)
                cls_dir = os.path.join(out_root, cls)
                os.makedirs(cls_dir, exist_ok=True)
                fn = (f"{cls}_{session}_"
                      f"{next_index(cls_dir, cls, session)}.jpg")
                if cv2.imwrite(os.path.join(cls_dir, fn), small):
                    counts[cls] += 1
                else:
                    print(f"WARNING: could not write {fn} (skipped)")
                burst_i += 1
                next_due = now + args.interval
                if burst_i >= args.n:
                    bursting = False
                    print(f"burst done: {cls} now has {counts[cls]} image(s)")

            key = cv2.waitKeyEx(30)
            if key == -1:
                pass
            else:
                action, state["idx"], state["buffer"] = on_key(key, state)
                if action == "quit":
                    break
                if action == "burst":
                    if bursting:
                        bursting = False
                        print("burst stopped early")
                    else:
                        bursting, burst_i = True, 0
                        next_due = time.monotonic()

            # commit a lone single-letter buffer after a 0.7 s pause
            if (state["buffer"] and len(state["buffer"]) == 1
                    and state["buffer"].upper() in CLASSES
                    and now - state["typed_at"] > 0.7):
                state["idx"] = CLASSES.index(state["buffer"].upper())
                state["buffer"] = ""
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("quit. counts per class:",
          {c: counts[c] for c in CLASSES if counts[c]})


if __name__ == "__main__":
    main()
