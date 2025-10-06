# src/streamlit_app.py
import streamlit as st
from PIL import Image
import torch
from model import get_resnet50
import torchvision.transforms as transforms
import numpy as np
import io

st.set_page_config(page_title="SolarGuard", layout="centered")

@st.cache_resource
def load_classifier(path, num_classes=6):
    device = torch.device('cpu')
    model = get_resnet50(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_pil, size=224):
    tf = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tf(image_pil).unsqueeze(0)

st.title("SolarGuard — Solar Panel Condition Detector")

uploaded = st.file_uploader("Upload a solar-panel image", type=['jpg','jpeg','png'])
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded image", use_column_width=True)
    # load classifier (change path if needed)
    try:
        model = load_classifier("models/resnet50_best.pth", num_classes=6)
        input_tensor = preprocess_image(img)
        with st.spinner("Predicting..."):
            out = model(input_tensor)
            probs = torch.nn.functional.softmax(out, dim=1).squeeze().detach().numpy()
            labels = ['Clean','Dusty','Bird-Drop','Electrical-Damage','Physical-Damage','Snow-Covered']
            idx = int(probs.argmax())
            st.write("**Prediction:**", labels[idx])
            st.write("**Confidence:** {:.2f}%".format(probs[idx]*100))
            # show full probs table
            st.table({labels[i]: float(probs[i]) for i in range(len(labels))})
    except Exception as e:
        st.error(f"Classifier load/predict error: {e}")

    # Optional: YOLOv8 detection (if ultralytics installed and model available)
    if st.checkbox("Run object detection (YOLOv8)"):
        try:
            from ultralytics import YOLO
            det_model = YOLO("models/yolov8n_solar.pt")  # put your trained detection model here
            with st.spinner("Running YOLOv8..."):
                # ultralytics accepts numpy arrays
                img_np = np.array(img)
                results = det_model.predict(img_np, imgsz=640, conf=0.25)[0]
                # results.boxes.xyxy, results.boxes.conf, results.boxes.cls
                annotated = results.plot()
                st.image(annotated, caption="Detections", use_column_width=True)
        except Exception as e:
            st.error(f"YOLO not available or error: {e}")
