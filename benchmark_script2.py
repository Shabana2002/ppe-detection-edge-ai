import time
from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\hp\Downloads\Obraz\best.onnx")

img = cv2.imread(r"C:\Users\hp\Downloads\Obraz\Dataset_no_person_glasses\val\images\4_jpg.rf.8c0629bf23d4911557ca5cdc92b45919.jpg")

for _ in range(10):
    model(img)

start = time.time()

for _ in range(100):
    model(img)

end = time.time()

fps = 100 / (end - start)

print("ONNX MODEL RESULTS")
print(f"FPS: {fps:.2f}")
print("Model: best.onnx (FP16)")