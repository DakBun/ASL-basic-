"""Live ASL letter recognition demo using a webcam + your Roboflow-hosted model.

Setup:
    pip install opencv-python inference-sdk

Set ROBOFLOW_API_KEY as an environment variable before running.

Usage:
    python webcam_demo.py
    (press 'q' in the video window to quit)
"""
import os
import threading
import time

import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY")

WORKSPACE_NAME = "dakbuns-workspace"
WORKFLOW_ID = "american-sign-language-letters-vamerican-sign-language-letters-bpoi9-1-rfdetr-large-t1-logic"

INFER_INTERVAL_SECONDS = 1.0
CONFIDENCE_THRESHOLD = 0.3


class InferenceWorker:
    """Runs run_workflow() calls on a background thread so the video loop
    never blocks waiting on the network."""

    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.pending_frame = None
        self.predictions = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.pending_frame = frame.copy()

    def get_predictions(self):
        with self.lock:
            return self.predictions

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def _loop(self):
        while self.running:
            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                result = self.client.run_workflow(
                    workspace_name=WORKSPACE_NAME,
                    workflow_id=WORKFLOW_ID,
                    images={"image": frame},
                    use_cache=True,
                )
                raw_preds = result[0]["predictions"]["predictions"]
                preds = [p for p in raw_preds if p["confidence"] >= CONFIDENCE_THRESHOLD]
                with self.lock:
                    self.predictions = preds
            except Exception as e:
                print(f"Inference error: {e}")


def draw_predictions(frame, predictions):
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = f'{pred["class"]} {pred["confidence"]:.0%}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def main():
    if API_KEY == "YOUR_API_KEY":
        raise SystemExit("Set ROBOFLOW_API_KEY first (https://app.roboflow.com/settings/api)")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY,
    ).configure(InferenceConfiguration(api_key_transport="header"))

    worker = InferenceWorker(client)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try 1 if you have more than one camera.")

    last_submit_time = 0.0

    print("Running. Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            now = time.time()
            if now - last_submit_time >= INFER_INTERVAL_SECONDS:
                last_submit_time = now
                worker.submit(frame)

            draw_predictions(frame, worker.get_predictions())
            cv2.imshow("ASL Letter Recognition (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


