import os
from io import BytesIO
import base64
import subprocess

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# 配置
IMG_SIZE = 128
CLASS_NAMES = {0: "fake", 1: "real"}
# 使用改进版训练脚本生成的权重文件
WEIGHT_PATH = os.path.join(os.path.dirname(
    __file__), "best_improved_resnet18.pth")


# 与 train.ipynb / train.py 中一致的模型结构
class FaceResNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(FaceResNet, self).__init__()
        # 使用 ResNet18 作为 backbone（训练时使用的是 pretrained=True）
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        # 去掉原始全连接层
        self.backbone.fc = nn.Identity()
        # 自定义分类头，与训练代码保持一致
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def load_model(weight_path: str, num_classes: int = 2):
    """加载使用 FaceResNet 训练好的模型权重。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceResNet(num_classes=num_classes)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


app = Flask(__name__)
CORS(app)
_transform = build_transform()

# 模型缓存：key 为模型文件名，value 为 (model, device) 元组
_model_cache = {}
_default_model, _default_device = load_model(WEIGHT_PATH)
_model_cache[os.path.basename(WEIGHT_PATH)] = (_default_model, _default_device)


def _encode_image_to_base64(path: str) -> str | None:
    """将图片文件编码为 base64，便于通过 JSON 返回。"""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/getModel", methods=["GET"])
def get_model():
    """返回项目根目录下所有 .pth 模型文件名列表"""
    project_root = os.path.dirname(__file__)
    model_files = []
    
    try:
        # 扫描项目根目录下的所有文件
        for filename in os.listdir(project_root):
            if filename.endswith(".pth") and os.path.isfile(os.path.join(project_root, filename)):
                model_files.append(filename)
        # 按文件名排序
        model_files.sort()
    except Exception as e:
        return jsonify({"error": f"扫描模型文件失败: {str(e)}"}), 500
    
    return jsonify({"models": model_files})


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    # 获取模型文件名参数（优先从 form data，其次从 query parameter）
    model_name = request.form.get("model") or request.args.get("model")
    
    # 确定要使用的模型和设备
    if model_name:
        # 如果指定了模型，检查缓存或加载
        if model_name not in _model_cache:
            # 构建模型文件路径（同级目录）
            model_path = os.path.join(os.path.dirname(__file__), model_name)
            if not os.path.exists(model_path):
                return jsonify({"error": f"模型文件不存在: {model_name}"}), 400
            try:
                model, device = load_model(model_path)
                _model_cache[model_name] = (model, device)
            except Exception as e:
                return jsonify({"error": f"加载模型失败: {str(e)}"}), 500
        model, device = _model_cache[model_name]
    else:
        # 使用默认模型
        model, device = _default_model, _default_device

    try:
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "invalid image file"}), 400

    tensor = _transform(image).unsqueeze(0).to(device)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_label = CLASS_NAMES.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx].item())

    return jsonify(
        {
            "label": pred_label,
            "confidence": confidence,
            "probs": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(probs.size(0))},
            "model": model_name or os.path.basename(WEIGHT_PATH),  # 返回使用的模型名称
        }
    )


@app.route("/train", methods=["POST"])
def train():
    """
    训练接口：
      - 调用 train.py 脚本进行一次完整训练
      - 训练脚本应在项目根目录下生成四张可视化图片，例如：
          1) train_loss_curve.png
          2) train_acc_curve.png
          3) train_feature_maps.png
          4) train_gradcam.png
      - 本接口将这些图片以 base64 的形式返回给前端
      - 可通过 form data 或 JSON 传入 epoch 参数来指定训练轮数
    """
    project_root = os.path.dirname(__file__)
    
    # 获取 epoch 参数（优先从 form data，其次从 JSON，最后使用默认值）
    if request.is_json:
        data = request.get_json()
        epochs = data.get("epoch")
    else:
        epochs = request.form.get("epoch", 2)
    
    # 转换为整数，如果转换失败则使用默认值
    try:
        epochs = int(epochs)
        if epochs < 1:
            epochs = 2
    except (ValueError, TypeError):
        epochs = 2

    # 调用独立的训练脚本，确保与命令行运行效果一致
    # 通过 --epochs 参数传递训练轮数
    proc = subprocess.run(
        ["python", "train.py", "--epochs", str(epochs)],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        return jsonify(
            {
                "status": "error",
                "message": "train.py 执行失败",
                "stdout": proc.stdout[-2000:],  # 截断返回，防止过长
                "stderr": proc.stderr[-2000:],
            }
        ), 500

    # 这里的文件名需要与 train.py 中保存图片的逻辑保持一致
    image_names = [
        "train_loss_curve.png",
        "train_acc_curve.png",
        "train_feature_maps.png",
        "train_gradcam.png",
    ]

    images = {}
    for name in image_names:
        full_path = os.path.join(project_root, name)
        b64 = _encode_image_to_base64(full_path)
        if b64 is not None:
            images[name] = b64

    return jsonify(
        {
            "status": "ok",
            "epochs": epochs,
            "images": images,
            "stdout": proc.stdout[-2000:],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port, debug=False)
