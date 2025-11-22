import streamlit as st
import cv2
import numpy as np
import io
import os

# ---------- Paths (optional, if you need project root) ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Correct import from your utilities
from stream_lit.model_utils import (
    load_ud_model, load_yolo_model,
    enhance_image, run_detection
)

# ---------- App Title ----------
st.set_page_config(page_title="Underwater Enhancement + Detection", layout="wide")

st.title("🌊 Underwater Image Enhancement & Object Detection")

st.write("""
This application enhances underwater images using **UDNet** and detects objects using **YOLO**.
Use the buttons below to:
- Run object detection on the original image
- Enhance the image
- Compare detection before and after enhancement
""")

# ---------- Load Models (cached) ----------
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
# LEFT PANEL → IMAGE UPLOAD + BUTTONS + RESULTS
# ----------------------------------------------------------
with left_col:
    st.markdown("### 📤 Upload Image")

    uploaded = st.file_uploader(
        "Upload an underwater image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    # Initialize session_state keys if needed
    if "file_name" not in st.session_state:
        st.session_state.file_name = None
    if "original" not in st.session_state:
        st.session_state.original = None
    if "enhanced" not in st.session_state:
        st.session_state.enhanced = None
    if "detected_original" not in st.session_state:
        st.session_state.detected_original = None
    if "detected_enhanced" not in st.session_state:
        st.session_state.detected_enhanced = None
    if "last_action" not in st.session_state:
        st.session_state.last_action = None

    # When a new image is uploaded, reset state
    if uploaded and (st.session_state.file_name != uploaded.name):
        st.session_state.file_name = uploaded.name

        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.session_state.original = img_cv
        st.session_state.enhanced = None
        st.session_state.detected_original = None
        st.session_state.detected_enhanced = None
        st.session_state.last_action = None

    # If we have an original image in memory, work with it
    if st.session_state.original is not None:
        img_cv = st.session_state.original

        st.subheader("📸 Original Image")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_column_width=True)

        # --- Three main buttons in a row ---
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            enhance_clicked = st.button("✨ Enhance", use_container_width=True)
        with btn_col2:
            detect_orig_clicked = st.button("🎯 Run Object Detection", use_container_width=True)
        with btn_col3:
            compare_clicked = st.button("🧪 Compare", use_container_width=True)

        # -----------------------
        # Actions: Enhance
        # -----------------------
        if enhance_clicked:
            st.session_state.enhanced = enhance_image(ud_model, img_cv)
            st.session_state.last_action = "enhance"

        # -----------------------
        # Actions: Detect on Original
        # -----------------------
        if detect_orig_clicked:
            st.session_state.detected_original = run_detection(yolo_model, img_cv)
            st.session_state.last_action = "detect_original"

        # -----------------------
        # Actions: Compare
        # -----------------------
        if compare_clicked:
            # 1) Detection on original
            if st.session_state.detected_original is None:
                st.session_state.detected_original = run_detection(yolo_model, img_cv)

            # 2) Enhancement
            if st.session_state.enhanced is None:
                st.session_state.enhanced = enhance_image(ud_model, img_cv)

            # 3) Detection on enhanced
            if st.session_state.detected_enhanced is None:
                st.session_state.detected_enhanced = run_detection(
                    yolo_model, st.session_state.enhanced
                )

            st.session_state.last_action = "compare"

        # --------------------------------------------------
        # Show Enhanced Image (if available)
        # --------------------------------------------------
        if st.session_state.enhanced is not None:
            enhanced = st.session_state.enhanced

            st.subheader("🔧 Enhanced Image")
            st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Download enhanced
            _, buf_enh = cv2.imencode(".png", enhanced)
            st.download_button(
                "📥 Download Enhanced Image",
                data=io.BytesIO(buf_enh.tobytes()),
                file_name="enhanced.png",
                mime="image/png",
                use_container_width=True
            )

            # Button for detection ON ENHANCED image
            if st.button("🎯 Run Object Detection on Enhanced", use_container_width=True, key="detect_on_enh"):
                st.session_state.detected_enhanced = run_detection(yolo_model, enhanced)
                st.session_state.last_action = "detect_enhanced"

        # --------------------------------------------------
        # Show Detection on Original (if exists)
        # --------------------------------------------------
        if st.session_state.detected_original is not None:
            det_orig = st.session_state.detected_original

            st.subheader("🟦 Detection on Original Image")
            # Assuming run_detection returns BGR image; convert to RGB for display
            st.image(cv2.cvtColor(det_orig, cv2.COLOR_BGR2RGB), use_column_width=True)

            _, buf_det_o = cv2.imencode(".png", det_orig)
            st.download_button(
                "📥 Download Original Detection Result",
                data=io.BytesIO(buf_det_o.tobytes()),
                file_name="detected_original.png",
                mime="image/png",
                use_container_width=True
            )

        # --------------------------------------------------
        # Show Detection on Enhanced (if exists)
        # --------------------------------------------------
        if st.session_state.detected_enhanced is not None:
            det_enh = st.session_state.detected_enhanced

            st.subheader("🟩 Detection on Enhanced Image")
            st.image(cv2.cvtColor(det_enh, cv2.COLOR_BGR2RGB), use_column_width=True)

            _, buf_det_e = cv2.imencode(".png", det_enh)
            st.download_button(
                "📥 Download Enhanced Detection Result",
                data=io.BytesIO(buf_det_e.tobytes()),
                file_name="detected_enhanced.png",
                mime="image/png",
                use_container_width=True
            )

        # --------------------------------------------------
        # Compare View: side-by-side detections
        # --------------------------------------------------
        if (
                st.session_state.detected_original is not None
                and st.session_state.detected_enhanced is not None
                and st.session_state.last_action == "compare"
        ):
            st.subheader("🧪 Comparison: Original vs Enhanced (Detection)")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original + Detection**")
                st.image(
                    cv2.cvtColor(st.session_state.detected_original, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )
            with col_b:
                st.markdown("**Enhanced + Detection**")
                st.image(
                    cv2.cvtColor(st.session_state.detected_enhanced, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )

    else:
        st.info("Please upload an underwater image to begin.")

# ----------------------------------------------------------
# RIGHT PANEL → INFORMATION + STEPS
# ----------------------------------------------------------
with right_col:
    st.markdown("### ℹ️ About This Application")
    st.write("""
This tool performs **two major tasks** on underwater images:

#### **1️⃣ Image Enhancement (UDNet)**
Underwater images often suffer from:
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

#### **Step 2 — Choose an Action**
- **🎯 Run Object Detection** → runs YOLO on the **original** image  
- **✨ Enhance** → enhances the image; then you can run detection on the enhanced version  
- **🧪 Compare** → in a single click:
  - runs detection on original  
  - enhances the image  
  - runs detection on enhanced  
  - shows both results side by side  

You can also download:
- Enhanced image  
- Detection outputs  
""")

    st.markdown("---")

    st.markdown("### ⚙️ Models Used")

    st.write("""
- **UDNet** – Deep learning model for underwater image enhancement  
- **YOLO** – High-performance object detection model  
Both models are automatically loaded and cached for speed.
""")
