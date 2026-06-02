from ultralytics import YOLO

model = YOLO(r"C:\Users\hp\Downloads\Obraz\best.pt")

metrics = model.val(
    data=r"C:\Users\hp\Downloads\Obraz\data.yaml"
)

print("===== FP32 METRICS =====")
print(metrics.results_dict)