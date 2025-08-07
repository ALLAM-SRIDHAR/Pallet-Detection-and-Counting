# 📦 Pallet Detection and Counting Using Computer Vision

## 🧠 Project Overview
This project addresses the problem of inaccurate pallet counting in warehouses using deep learning and computer vision. A custom object detection model is trained to detect and count stacked pallets in real-time from warehouse images.

---

## 🎯 Problem Statement

Manual and semi-automated pallet counting methods often result in:
- Stock discrepancies
- Fulfillment delays
- High operational costs

Traditional scanners fail to detect **stacked pallets**, especially under low lighting or occlusions. As operations scale, accuracy becomes critical.

---

## 🚀 Objectives & Goals

- ✅ Achieve **≥95% model accuracy**
- ✅ Detect and count pallets with **≤1 sec inference time**
- ✅ Reduce warehouse operational costs by **25–30%**


**Business Objective:** Minimize the latency of count and Maximize Precision accuracy

**Business Constraint:** Minimize Processing Time

**Business Success Criteria:** Achieve 25% reduction in overall warehouse operational costs, including labor and counting inefficiencies.

**Machine Learning Success Criteria:** Achieve at least 95% model accuracy with an inference timeof I second per frame.

**Economic Success Criteria:** Achieve reduction in operational costs, including labor and inventory loss.

---

## 🛠️ Tech Stack

| Component      | Tools/Frameworks Used             |
|----------------|------------------------------------|
| Programming    | Python                             |
| Deep Learning  | PyTorch, Roboflow, OpenCV          |
| Models Used    | YOLOv5, YOLOv8, Faster R-CNN       |
| Deployment     | Flask, AWS EC2                     |
| Visualization  | Power BI, Streamlit                |
| Annotation     | Roboflow (COCO format)             |

---

## 📁 Dataset

- **Images Collected**: 200+ front-facing pallet images  
- **Annotated Using**: [Roboflow](https://roboflow.com)  
- **Format**: COCO JSON  
- **Classes**: Unified to `pallet` to prevent model confusion

---

## 🧪 Model Training

- Preprocessed data: resize, normalize, augment
- Trained on multiple object detection architectures:
  - YOLOv5 (speed optimized)
  - YOLOv8 (accuracy focused)
  - Faster R-CNN (robust detection)

---

## 📈 Results

| Metric          | Value             |
|-----------------|------------------|
| Accuracy        | 95.3%             |
| Inference Time  | ~0.8 sec/frame    |
| Cost Savings    | ~30% reduction    |

---

## 🧑‍💻 How to Run

```bash
# Clone the repository
git clone https://github.com/ALLAM-SRIDHAR/Pallet-Detection-and-Counting.git
cd Pallet-Detection-and-Counting

# Install dependencies
pip install -r requirements.txt

# Run detection
python detect.py --source path_to_image_or_video

# (Optional) Deploy API
python app.py


**Note:** My origial dataset consisted of 217 images which i cannot upload directly github due to length of file, so i am uploading it to my drive and here is its link: https://drive.google.com/file/d/1cHx6z0SrabNGrAot_GIBWTmRtywCFs48/view?usp=sharing

Data Preprocessing, Data Annotation, Data Augumentation, etc is done in Roboflow application.
