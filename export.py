from ultralytics import YOLO

MODEL_PATH = r"C:\Users\hp\Downloads\Obraz\best.pt"

model = YOLO(MODEL_PATH)

model.export(
    format="onnx",
    half=True,      # FP16 quantization
    simplify=True
)

print("✅ ONNX export complete")