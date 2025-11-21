import streamlit as st
import cv2
import numpy as np
import io
import os

# ---------- Paths ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_UTILS_PATH = os.path.join(PROJECT_ROOT, "streamlit")

# Correct import
from stream_lit.model_utils import (
    load_ud_model, load_yolo_model,
    enhance_image, run_detection
)

# ---------- App Title ----------
st.set_page_config(page_title="Underwater Enhancement + Detection", layout="wide")

st.title("🌊 Underwater Image Enhancement & Object Detection")

st.write("""
This application enhances underwater images using **UDNet** and detects objects using **YOLO**.
Follow the steps shown on the right panel to complete the process.
""")

# ---------- Load Models ----------
@st.cache_resource(show_spinner=True)
def get_ud_model():
    return load_ud_model()

@st.cache_resource(show_spinner=True)
def get_yolo_model():
    return load_yolo_model()

ud_model = get_ud_model()
yolo_model = get_yolo_model()


# ---------- Layout ----------
left_col, right_col = st.columns([1.2, 1])

# ----------------------------------------------------------
# LEFT PANEL → IMAGE UPLOAD + RESULTS
# ----------------------------------------------------------
with left_col:
    st.markdown("### 📤 Upload Image")

    uploaded = st.file_uploader(
        "Upload an underwater image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    enhance_btn = st.button("✨ Enhance Image", use_container_width=True)

    # Reset session when new image arrives
    if uploaded and (
            "file_name" not in st.session_state
            or st.session_state.file_name != uploaded.name
    ):
        st.session_state.file_name = uploaded.name
        st.session_state.enhanced = None
        st.session_state.detected = None

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.subheader("📸 Original Image")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_column_width=True)

        # --- Enhancement ---
        if enhance_btn and st.session_state.get("enhanced") is None:
            st.session_state.enhanced = enhance_image(ud_model, img_cv)

        if st.session_state.get("enhanced") is not None:
            enhanced = st.session_state.enhanced

            st.subheader("🔧 Enhanced Image")
            st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_column_width=True)

            _, buf = cv2.imencode(".png", enhanced)
            st.download_button(
                "📥 Download Enhanced Image",
                data=io.BytesIO(buf.tobytes()),
                file_name="enhanced.png",
                mime="image/png",
                use_container_width=True
            )

            # Detection
            if st.button("🎯 Run Object Detection", use_container_width=True):
                st.session_state.detected = run_detection(yolo_model, enhanced)

        # --- Detection Output ---
        if st.session_state.get("detected") is not None:
            det = st.session_state.detected

            st.subheader("🟦 Detection Result")
            st.image(cv2.cvtColor(det, cv2.COLOR_RGB2BGR), use_column_width=True)

            _, buf = cv2.imencode(".png", det)
            st.download_button(
                "📥 Download Detection Result",
                data=io.BytesIO(buf.tobytes()),
                file_name="detected.png",
                mime="image/png",
                use_container_width=True
            )



# ----------------------------------------------------------
# RIGHT PANEL → INFORMATION + STEPS
# ----------------------------------------------------------
with right_col:
    st.markdown("### ℹ️ About This Application")
    st.write("""
This tool performs **two major tasks** on underwater images:

#### **1️⃣ Image Enhancement (UDNet)**
Underwater images usually suffer from:
- Color distortion  
- Low contrast  
- Poor visibility  

UDNet restores image clarity, colors, and sharpness.

#### **2️⃣ Object Detection (YOLO)**
Once enhanced, the image is passed through a YOLO model to identify:
- Marine animals  
- Underwater structures  
- Objects of interest  

The detection result is displayed with bounding boxes.
""")

    st.markdown("---")

    st.markdown("### 🧭 How to Use")

    st.write("""
#### **Step 1 — Upload**
Upload any underwater image (JPG/PNG).

#### **Step 2 — Enhance**
Click **Enhance Image** to improve clarity and colors.

#### **Step 3 — Detect**
Click **Run Object Detection** to identify objects.

You can also download:
- Enhanced image  
- Detection output  
""")

    st.markdown("---")

    st.markdown("### ⚙️ Models Used")

    st.write("""
- **UDNet** – Deep learning model for underwater image enhancement  
- **YOLO** – High-performance object detection model  
Both models are automatically loaded and cached for speed.
""")