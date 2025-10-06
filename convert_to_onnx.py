# convert_to_onnx.py
import torch
from src.model import get_resnet50

model = get_resnet50(num_classes=6, pretrained=False)
model.load_state_dict(torch.load("models/resnet50_best.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 224, 224, device='cpu')
torch.onnx.export(
    model,
    dummy,
    "models/resnet50_best.onnx",
    export_params=True,
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print("Exported models/resnet50_best.onnx")
