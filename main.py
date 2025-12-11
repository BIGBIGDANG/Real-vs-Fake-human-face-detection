import os
from io import BytesIO

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# 配置
IMG_SIZE = 128
CLASS_NAMES = {0: "fake", 1: "real"}
WEIGHT_PATH = os.path.join(os.path.dirname(__file__), "best_resnet18_rvf10k.pth")


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def load_model(weight_path: str, num_classes: int = 2):
    """加载训练好的 ResNet18 模型权重。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


app = Flask(__name__)
CORS(app)
_transform = build_transform()
_model, _device = load_model(WEIGHT_PATH)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "invalid image file"}), 400

    tensor = _transform(image).unsqueeze(0).to(_device)
    outputs = _model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_label = CLASS_NAMES.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx].item())

    return jsonify(
        {
            "label": pred_label,
            "confidence": confidence,
            "probs": {
                CLASS_NAMES[i]: float(probs[i].item()) for i in range(probs.size(0))
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port, debug=False)
