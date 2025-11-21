import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ---------- Paths ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UD_MODEL_PATH = os.path.join(PROJECT_ROOT, "stream_lit", "weights", "UDnet.pth")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "stream_lit", "weights", "best.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_MULTIPLE = 32


# ---------- Load Models ----------
def load_ud_model():
    from imageEnhancement.model_utils.UDnet import mynet, Opt

    device = torch.device(DEVICE)
    opt = Opt(device)
    model = mynet(opt)

    state = torch.load(UD_MODEL_PATH, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)


# ---------- Enhancement Helpers ----------
def preprocess_np(img_np, multiple=32):
    h, w = img_np.shape[:2]
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    pad_bottom = new_h - h
    pad_right = new_w - w

    img_pad = cv2.copyMakeBorder(
        img_np, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT
    )

    tensor = torch.from_numpy(
        cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    return tensor, (0, pad_bottom, 0, pad_right)


def tensor_to_bgr_image(tensor):
    tensor = tensor.detach().cpu()
    arr = tensor.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def crop_to_original(img_np, pads):
    top, bottom, left, right = pads
    h, w = img_np.shape[:2]
    y0, y1 = top, h - bottom if bottom != 0 else h
    x0, x1 = left, w - right if right != 0 else w
    return img_np[y0:y1, x0:x1]


# ---------- Full Enhancement Step ----------
def enhance_image(model, img_cv):
    tensor, pads = preprocess_np(img_cv, PAD_MULTIPLE)
    tensor = tensor.to(torch.device(DEVICE))

    with torch.no_grad():
        try:
            model.forward(tensor, tensor, training=False)
        except TypeError:
            model.forward(tensor, training=False)

        out = model.sample(testing=True)
        out_tensor = out[0] if isinstance(out, (list, tuple)) else out
        enhanced_bgr = crop_to_original(tensor_to_bgr_image(out_tensor), pads)

    return enhanced_bgr


# ---------- YOLO Detection ----------
def run_detection(yolo_model, enhanced_bgr):
    results = yolo_model.predict(source=enhanced_bgr, conf=0.4, save=False)
    return results[0].plot()