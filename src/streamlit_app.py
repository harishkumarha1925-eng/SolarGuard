# src/streamlit_app.py

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

# --- Streamlit Page Config ---
st.set_page_config(page_title="SolarGuard", layout="centered")

# --- Model Paths ---
MODEL_DIR = Path("models")
ONNX_MODEL_PATH = MODEL_DIR / "resnet50_best.onnx"

# --- Correct Label Order (alphabetical as ImageFolder uses) ---
LABELS = [
    'Bird-Drop',
    'Clean',
    'Dusty',
    'Electrical-Damage',
    'Physical-Damage',
    'Snow-Covered'
]

# --- Torch Conditional Import ---
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    # ONNXRuntime will be imported later inside load_onnx()

# --- Image Preprocessing Function ---
def preprocess(img, size=224):
    """Resize, normalize, and prepare image tensor for model inference."""
    img = img.resize((size, size)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW

    # Standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    arr = (arr - mean) / std

    return arr[np.newaxis, :, :, :].astype(np.float32)


# --- Load ONNX Model (cached) ---
@st.cache_resource
def load_onnx_model():
    """Load the ONNX model for inference."""
    import onnxruntime as ort
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")
    sess = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    return sess


# --- Streamlit UI ---
st.title("☀️ SolarGuard — Solar Panel Condition Detector")
st.write("Upload a solar panel image to detect its condition using Deep Learning.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if not uploaded_file:
    st.info("Please upload an image to start detection.")
else:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("⏳ Processing image...")

    x = preprocess(img)

    # --- Inference: Try Torch First (for local) ---
    if TORCH_AVAILABLE:
        try:
            from src.model import get_resnet50
            model = get_resnet50(num_classes=6, pretrained=False)
            model.load_state_dict(torch.load("models/resnet50_best.pth", map_location="cpu"))
            model.eval()

            with torch.no_grad():
                output = model(torch.from_numpy(x))
                probs = torch.nn.functional.softmax(output, dim=1).numpy().squeeze()

            idx = int(np.argmax(probs))
            st.success(f"**Predicted Condition (PyTorch): {LABELS[idx]}**")
            st.write(f"Confidence: {probs[idx]*100:.2f}%")

        except Exception as e:
            st.error(f"⚠️ PyTorch inference failed — falling back to ONNX. Error: {e}")

    # --- ONNX Inference (for Streamlit Cloud) ---
    try:
        sess = load_onnx_model()
        outputs = sess.run(None, {"input": x})[0]
        exps = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probs = (exps / np.sum(exps, axis=1, keepdims=True)).squeeze()

        idx = int(np.argmax(probs))
        st.success(f"**Predicted Condition (ONNX): {LABELS[idx]}**")
        st.write(f"Confidence: {probs[idx]*100:.2f}%")

    except FileNotFoundError as fe:
        st.error(f"❌ Model file missing: {fe}")
    except Exception as e:
        st.error(f"❌ ONNX Inference Error: {e}")


# --- Footer ---
st.markdown("""
---
**SolarGuard (v1.0)**  
Deep Learning–powered solar panel defect classification  
Built with 🧠 PyTorch • ☁️ Streamlit • 🔍 ONNX Runtime
""")
