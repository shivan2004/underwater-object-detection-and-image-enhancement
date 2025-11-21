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

# ---------- Load Models ----------
@st.cache_resource(show_spinner=False)
def get_ud_model():
    return load_ud_model()

@st.cache_resource(show_spinner=False)
def get_yolo_model():
    return load_yolo_model()


ud_model = get_ud_model()
yolo_model = get_yolo_model()


# ---------- UI ----------
st.title("UDnet Enhancement + Object Detection")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
enhance_btn = st.button("Enhance Image")

# Session reset for new uploads
if uploaded is not None and (
        'uploaded_name' not in st.session_state or
        st.session_state['uploaded_name'] != uploaded.name
):
    st.session_state['enhanced_image'] = None
    st.session_state['detected_image'] = None
    st.session_state['uploaded_name'] = uploaded.name


if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_cv is None:
        st.error("Could not read image.")
    else:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_column_width=True)

        # ---------- Enhancement ----------
        if enhance_btn and st.session_state.get("enhanced_image") is None:
            st.session_state['enhanced_image'] = enhance_image(ud_model, img_cv)

        if st.session_state.get("enhanced_image") is not None:
            enhanced = st.session_state['enhanced_image']
            st.subheader("Enhanced Image")
            st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_column_width=True)

            _, buffer = cv2.imencode(".png", enhanced)
            st.download_button("Download Enhanced Image", data=io.BytesIO(buffer.tobytes()),
                               file_name="enhanced.png", mime="image/png")

            # ---------- Detection ----------
            if st.button("Run Object Detection"):
                st.session_state['detected_image'] = run_detection(yolo_model, enhanced)

        # Show detected image
        if st.session_state.get("detected_image") is not None:
            det = st.session_state['detected_image']
            st.subheader("Object Detection Result")
            st.image(cv2.cvtColor(det, cv2.COLOR_RGB2BGR), use_column_width=True)

            _, buffer = cv2.imencode(".png", det)
            st.download_button("Download Detected Image", data=io.BytesIO(buffer.tobytes()),
                               file_name="detected.png", mime="image/png")