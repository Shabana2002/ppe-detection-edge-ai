from ultralytics import YOLO

model = YOLO(r"C:\Users\hp\Downloads\Obraz\best.onnx")

metrics = model.val(
    data=r"C:\Users\hp\Downloads\Obraz\data.yaml")
print("===== ONNX METRICS =====")
print(metrics.results_dict)