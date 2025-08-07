import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (make sure 'best.pt' is in the same directory)
model = YOLO("best.pt")

# Streamlit page settings
st.set_page_config(page_title="🧱 Pallet Detection and Counting", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🧱 Pallet Detection and Counting using YOLOv8</h1>", unsafe_allow_html=True)
st.markdown("---")

# Upload Section
st.markdown("### 📤 Upload an Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert and preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run detection
    results = model.predict(source=img_bgr, conf=0.25)

    # Extract detection data
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy()
    names = model.names

    unique, counts = np.unique(class_ids, return_counts=True)
    detection_counts = {
        names[int(cls_id)]: int(count)
        for cls_id, count in zip(unique, counts)
    }
    total_count = sum(detection_counts.values())

    # Render detected image
    rendered_img = results[0].plot()
    rendered_rgb = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)

    # Display Original and Detected images side-by-side
    st.markdown("### 🖼️ Visual Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(rendered_rgb, caption="Detected Pallets", use_column_width=True)

    # Display detection metrics
    st.markdown("### 🔍 Detection Summary")
    cols = st.columns(len(detection_counts))
    for idx, (cls_name, count) in enumerate(detection_counts.items()):
        with cols[idx]:
            st.metric(label=f"📦 {cls_name}", value=count)

    # Show total count
    st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'>🟩 Total Pallets Detected: {total_count}</h2>", unsafe_allow_html=True)

else:
    st.info("Please upload an image to start detection.")