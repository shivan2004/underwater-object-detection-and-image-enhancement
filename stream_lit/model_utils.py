# stream_lit/model_utils.py

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ---------- Device & constants ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_MULTIPLE = 32

# ---------- Paths (relative to this file) ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UD_MODEL_PATH = os.path.join(THIS_DIR, "weights", "UDnet.pth")
YOLO_MODEL_PATH = os.path.join(THIS_DIR, "weights", "best.pt")


# ---------- Load Models ----------
def load_ud_model():
    """
    Load UDNet enhancement model.
    Expects your UDNet code at: imageEnhancement/model_utils/UDnet.py
    with classes: mynet(opt), Opt(device)
    """
    from imageEnhancement.model_utils.UDnet import mynet, Opt

    device = torch.device(DEVICE)
    opt = Opt(device)
    model = mynet(opt)

    # Load weights
    state = torch.load(UD_MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_yolo_model():
    """Load YOLO model for detection."""
    return YOLO(YOLO_MODEL_PATH)


# ---------- Enhancement Helpers ----------
def preprocess_np(img_np, multiple=32):
    """
    Pad image so H and W are multiples of `multiple`,
    convert to tensor [1,3,H,W] in RGB, normalized [0,1].
    Input: BGR numpy (cv2).
    """
    h, w = img_np.shape[:2]
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    pad_bottom = new_h - h
    pad_right = new_w - w

    img_pad = cv2.copyMakeBorder(
        img_np, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT
    )

    rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor, (0, pad_bottom, 0, pad_right)


def tensor_to_bgr_image(tensor):
    """Convert model tensor output [1,3,H,W] or [3,H,W] in [0,1] RGB to BGR uint8."""
    tensor = tensor.detach().cpu()
    arr = tensor.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)  # H,W,3 RGB
    bgr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


def crop_to_original(img_np, pads):
    """Remove padding to get back to original size."""
    top, bottom, left, right = pads
    h, w = img_np.shape[:2]
    y0, y1 = top, h - bottom if bottom != 0 else h
    x0, x1 = left, w - right if right != 0 else w
    return img_np[y0:y1, x0:x1]


# ---------- Full Enhancement Step ----------
def enhance_image(model, img_cv):
    """
    Run UDNet enhancement on a BGR image (cv2).
    Returns enhanced image in BGR.
    """
    tensor, pads = preprocess_np(img_cv, PAD_MULTIPLE)
    tensor = tensor.to(torch.device(DEVICE))

    with torch.no_grad():
        # Try different forward signatures, depending on your UDNet implementation
        try:
            model.forward(tensor, tensor, training=False)
        except Exception:
            try:
                model.forward(tensor, training=False)
            except Exception:
                pass

        out = model.sample(testing=True)
        out_tensor = out[0] if isinstance(out, (list, tuple)) else out
        enhanced_bgr = crop_to_original(tensor_to_bgr_image(out_tensor), pads)

    return enhanced_bgr


# ---------- YOLO Detection ----------
def run_detection(yolo_model, bgr_image):
    """
    Run YOLO detection on a BGR image and return an annotated image (BGR).
    """
    results = yolo_model.predict(source=bgr_image, conf=0.4, save=False, verbose=False)
    # results[0].plot() returns an RGB or BGR image depending on ultralytics version;
    # for Streamlit we treat it as BGR and convert to RGB before display in the app.
    annotated = results[0].plot()
    return annotated
