# src/streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

st.set_page_config(page_title="SolarGuard", layout="centered")

MODEL_DIR = Path("models")
ONNX_MODEL_PATH = MODEL_DIR / "resnet50_best.onnx"
LABELS = ['Clean','Dusty','Bird-Drop','Electrical-Damage','Physical-Damage','Snow-Covered']

# Only import torch inside a try/except, never at top-level
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    # onnxruntime will be imported lazily in load_onnx()

def preprocess(img, size=224):
    img = img.resize((size, size)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return arr[np.newaxis, :, :, :].astype(np.float32)

@st.cache_resource
def load_onnx():
    import onnxruntime as ort
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")
    sess = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    return sess

st.title("SolarGuard — Solar Panel Condition Detector")

uploaded = st.file_uploader("Upload a solar-panel image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image to get a prediction.")
else:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)
    x = preprocess(img)

    if TORCH_AVAILABLE:
        try:
            import torch
            from src.model import get_resnet50
            model = get_resnet50(num_classes=6, pretrained=False)
            model.load_state_dict(torch.load("models/resnet50_best.pth", map_location="cpu"))
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(x))
                probs = torch.nn.functional.softmax(out, dim=1).numpy().squeeze()
            idx = int(np.argmax(probs))
            st.success(f"Predicted Condition (PyTorch): **{LABELS[idx]}**")
            st.write(f"Confidence: {probs[idx]*100:.2f}%")
        except Exception as e:
            st.error(f"PyTorch inference error: {e}")
    else:
        try:
            sess = load_onnx()
            out = sess.run(None, {"input": x})[0]
            exps = np.exp(out - np.max(out, axis=1, keepdims=True))
            probs = (exps / np.sum(exps, axis=1, keepdims=True)).squeeze()
            idx = int(np.argmax(probs))
            st.success(f"Predicted Condition (ONNX): **{LABELS[idx]}**")
            st.write(f"Confidence: {probs[idx]*100:.2f}%")
        except FileNotFoundError as fe:
            st.error(f"Model missing: {fe}")
        except Exception as e:
            st.error(f"ONNX inference error: {e}")
