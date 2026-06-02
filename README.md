# 🚀 PPE Detection Edge AI System (YOLOv8 + ONNX)

## 📌 Submission for Junior Computer Vision Engineer Assignment

---

# 👤 Candidate Information
- Name: Shabana


---

# 🧠 1. Project Overview

This project implements a **real-time PPE (Personal Protective Equipment) detection system** using YOLOv8.  

The system detects:
- Helmet
- Mask
- Safety Vest
- Boots

It is optimized for **edge deployment using ONNX quantization (FP16)** for faster inference.

---

# 📁 2. GitHub Repository

👉 Source Code & Project Repo:
https://github.com/Shabana2002/ppe-detection-edge-ai

Includes:
- Training notebook (YOLOv8)
- Conversion scripts (ONNX export)
- live_inference.py
- Evaluation scripts

---

# 💾 3. Model Weights

### 🟢 FP32 Model (Baseline)
https://github.com/Shabana2002/ppe-detection-edge-ai/blob/main/best.pt

### 🔵 Edge Optimized Model (ONNX FP16)
https://github.com/Shabana2002/ppe-detection-edge-ai/blob/main/best.onnx

---

# 📊 4. Dataset Details

- Source: Kaggle / Roboflow PPE Dataset
- Total Images: 1423+
- Total Instances: 3705

### Classes:
- Helmet
- Mask
- Vest
- Boots
- gloves

---

# ⚙️ 5. Model Architecture

- Model: YOLOv8n
- Framework: Ultralytics YOLO
- Training Precision: FP32
- Edge Conversion: ONNX (FP16)

---

# 📈 6. Performance Benchmark Table

| Metric | YOLOv8 FP32 (.pt) | YOLOv8 ONNX (FP16) |
|--------|------------------|---------------------|
| Model Size | ~14 MB | ~7 MB |
| FPS | 6.53 | 9.82 |
| Inference Time | ~200 ms | ~90 ms |
| mAP@50-95 | 0.343 | 0.342 |
| Precision | 0.708 | 0.715 |
| Recall | 0.567 | 0.566 |

---

# 📌 7. Key Observations

- ONNX reduces inference time by ~30–40%
- Model size reduced by ~50%
- Accuracy drop is negligible
- Best suited for real-time edge deployment

---

# 🎯 8. Trade-off Analysis

The ONNX FP16 model improves speed significantly while maintaining nearly identical accuracy compared to FP32.

✔ Faster inference  
✔ Lower latency  
✔ Reduced memory usage  
⚠ Slight numerical precision trade-off

---

# 🎥 9. Live Inference System

Run file:


Features:
- Webcam detection
- Bounding boxes
- Confidence scores
- FPS display
- Pre-processing & post-processing latency

---

# 🧪 10. Evaluation

### FP32:
```python
model.val(data="data.yaml")