

# 🚀 Edge AI PPE Detection System

### Junior Computer Vision Engineer Assignment Submission

---


---

# 📌 1. Problem Statement

This project focuses on an **industrial safety PPE detection system** designed to identify whether workers are wearing essential safety equipment such as:

* Helmet
* Mask
* Safety Vest
* Boots
* gloves

The system is designed for **real-time edge deployment** in industrial environments to improve safety compliance monitoring.

---

# 📊 2. Dataset Details

### Dataset Source:

Kaggle / Roboflow Public PPE Dataset

### Dataset Size:

* Total Images: **1423+**
* Total Instances: **3705**

### Classes:

* Helmet
* Mask
* Vest
* Boots
* gloves

### Dataset Structure:

```
Dataset_no_person_glasses/
│
├── train/
│   ├── images/
│   ├── labels/
│
├── val/
│   ├── images/
│   ├── labels/
│
└── data.yaml
```

---

# 🧠 3. Model Training (FP32 Baseline)

### Model Used:

* YOLOv8n (Ultralytics)

### Training Environment:

* Google Colab
* PyTorch FP32

### Output Model:

* `best.pt`

### Purpose:

Baseline high-accuracy detection model before optimization.

---

# ⚙️ 4. Edge Optimization (Model Conversion)

To make the model suitable for edge devices:

### Conversion Format:

* ONNX (Open Neural Network Exchange)

### Precision:

* FP16 (Half Precision Optimization)

### Converted Model:

* `best.onnx`

### Benefits:

* Reduced model size
* Faster inference
* Lower CPU usage

---

# 📈 5. Performance Benchmark

## 📊 Comparison Table

| Model       | Format | FPS  | Inference Time | Notes          |
| ----------- | ------ | ---- | -------------- | -------------- |
| YOLOv8 FP32 | .pt    | 6.53 | ~200 ms        | Baseline model |
| YOLOv8 ONNX | FP16   | 9.82 | ~90 ms         | Edge optimized |

---

## 📊 Accuracy Metrics

| Metric    | FP32 Model | ONNX Model |
| --------- | ---------- | ---------- |
| Precision | 0.708      | 0.715      |
| Recall    | 0.567      | 0.566      |
| mAP@50    | 0.609      | 0.609      |
| mAP@50-95 | 0.343      | 0.342      |

---

## 📌 Key Insight

> ONNX optimization improves inference speed by ~30% while maintaining almost identical accuracy.

---

# 🎥 6. Live Inference System

### Script:

`live_inference.py`

### Features:

* Real-time webcam detection
* Bounding box visualization
* Class labels + confidence scores
* FPS overlay
* Inference latency tracking
* Pre-processing & post-processing timing

### Example Output:

```
FPS: 9.82
Inference: 90 ms
Post-process: 2 ms
```

---

# 🧪 7. Evaluation (mAP Testing)

### FP32 Evaluation:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
metrics = model.val(data="data.yaml")
print(metrics.results_dict)
```

### ONNX Evaluation:

```python
model = YOLO("best.onnx")
metrics = model.val(data="data.yaml")
print(metrics.results_dict)
```

---

# 📁 8. Project Structure

```
Obraz_Project/
│
├── best.pt
├── best.onnx
├── data.yaml
│
├── scripts/
│   ├── live_inference.py
│   ├── evaluate_fp32.py
│   ├── evaluate_onnx.py
│   ├── benchmark_script.py
│
├── Dataset_no_person_glasses/
│
└── README.md
```

---

# 🎯 9. Results Summary

* Successfully trained YOLOv8 model on PPE dataset
* Converted model to ONNX for edge deployment
* Achieved real-time inference performance improvement
* Minimal accuracy loss after optimization
* Verified deployment on webcam (live inference)

---

# 🧾 10. Conclusion

This project demonstrates a complete **edge AI pipeline**:

✔ Dataset preparation
✔ Model training (FP32)
✔ Evaluation (mAP metrics)
✔ Optimization (ONNX FP16)
✔ Real-time inference system

The final ONNX model is suitable for **low-latency industrial edge applications** such as PPE monitoring systems.

---

# 🔗 11. Links (TO BE ADDED)

### GitHub Repository:

```
https://github.com/your-username/edge-ai-ppe-detection
```

### Model Weights:

```
Google Drive / HuggingFace link
```

### Demo Video:

```
YouTube / Google Drive (Unlisted)
```

