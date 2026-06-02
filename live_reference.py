import cv2
import time
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\hp\Downloads\Obraz\best.onnx"  # IMPORTANT: use ONNX

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture("demo.mp4")

if not cap.isOpened():
    print("Camera error")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Preprocess timing ----
    t0 = time.time()

    # inference
    results = model(frame, verbose=False)

    t1 = time.time()

    # postprocess (NMS is inside ultralytics but approximated here)
    annotated = results[0].plot()

    t2 = time.time()

    # ---- Metrics ----
    inference_time = (t1 - t0) * 1000
    postprocess_time = (t2 - t1) * 1000

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # ---- Overlay ----
    cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated, f"Infer: {inference_time:.1f} ms", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.putText(annotated, f"Post: {postprocess_time:.1f} ms", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Edge AI Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()