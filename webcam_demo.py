"""Live ASL letter classification demo using a webcam + your Roboflow-hosted model.

Setup:
    pip install opencv-python inference-sdk

Set ROBOFLOW_API_KEY in your .env file or as an environment variable before running.

Usage:
    python webcam_demo.py
    (press 'q' in the video window to quit)
"""
import os
import threading
import time
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Tải biến môi trường từ file .env (chứa ROBOFLOW_API_KEY)
load_dotenv()
API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# THAY ĐỔI 1: Cấu hình Model Classification thay cho Workflow
MODEL_ID = "sign-language-2c6kk/2"

INFER_INTERVAL_SECONDS = 0.5 # Gửi ảnh lên server mỗi 0.5s cho mượt
CONFIDENCE_THRESHOLD = 0.4   # Ngưỡng tin cậy 40%


class InferenceWorker:
    """Chạy API infer() trên background thread để webcam không bị giật lag."""

    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.pending_frame = None
        self.prediction = None  # Phân loại ảnh chỉ trả về 1 kết quả duy nhất
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.pending_frame = frame.copy()

    def get_prediction(self):
        with self.lock:
            return self.prediction

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
                # THAY ĐỔI 2: Gọi hàm infer() của Classification thay vì run_workflow
                result = self.client.infer(frame, model_id=MODEL_ID)
                
                # Bóc tách kết quả của Classification (chỉ lấy Top 1)
                top_class = result.get("top", "Unknown")
                confidence = result.get("confidence", 0.0)

                with self.lock:
                    if confidence >= CONFIDENCE_THRESHOLD:
                        self.prediction = {"class": top_class, "confidence": confidence}
                    else:
                        self.prediction = None
            except Exception as e:
                print(f"Lỗi Inference: {e}")


def draw_prediction(frame, prediction):
    # THAY ĐỔI 3: Không còn tọa độ x,y,w,h. Chỉ in Text lên góc màn hình.
    if prediction:
        label = f'{prediction["class"]}: {prediction["confidence"]:.0%}'
        
        # Vẽ nền đen mờ
        cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
        # Viết chữ
        cv2.putText(frame, label, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)


def main():
    if not API_KEY:
        raise SystemExit("Lỗi: Không tìm thấy ROBOFLOW_API_KEY. Hãy kiểm tra lại file .env")

    # Khởi tạo Client cho Classification API
    client = InferenceHTTPClient(
        api_url="https://classify.roboflow.com", 
        api_key=API_KEY,
    )

    worker = InferenceWorker(client)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở webcam (index 0).")

    last_submit_time = 0.0

    print(f"🚀 Đang chạy dự án: {MODEL_ID}")
    print("Nhấn phím 'q' trong cửa sổ video để thoát.")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lỗi không đọc được khung hình từ webcam.")
                break
                
            # Lật ngược ảnh để giống như đang soi gương
            frame = cv2.flip(frame, 1)

            now = time.time()
            if now - last_submit_time >= INFER_INTERVAL_SECONDS:
                last_submit_time = now
                worker.submit(frame)

            draw_prediction(frame, worker.get_prediction())
            cv2.imshow("Sign Language Classification - ResNet34", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()