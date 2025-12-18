import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2  # 如果没有安装 opencv，请 pip install opencv-python
import os
from utils import get_dataset_paths


# -------------------------------------------------
# 1. 基础设置与模型加载
# -------------------------------------------------
def load_model(weight_path, device):
    print(f"Loading weights from {weight_path}...")
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    # 加载权重
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(img_path):
    """
    预处理单张图片，使其符合模型输入 (128x128, norm)
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0)  # [1, 3, 128, 128]
    return raw_img, input_tensor


# -------------------------------------------------
# 2. 中间层特征图可视化 (Feature Maps)
# -------------------------------------------------
class FeatureMapVisualizer:
    def __init__(self, model, target_layer):
        self.model = model
        self.feature_maps = []
        # 注册 Hook
        target_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature_maps.append(output.detach())

    def visualize(self, input_tensor, save_name="feature_maps.png"):
        self.feature_maps = []  # 清空之前的
        _ = self.model(input_tensor)  # 前向传播触发 Hook

        # 获取特征图 [1, Channels, H, W]
        fmap = self.feature_maps[0].cpu().squeeze(0)

        # 只画前 16 个通道
        num_channels = min(16, fmap.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(f"Top {num_channels} Feature Maps", fontsize=16)

        for i in range(num_channels):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            # 简单的归一化显示
            img = fmap[i].numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-9)
            ax.imshow(img, cmap='viridis')
            ax.axis('off')
            ax.set_title(f"Ch {i}")

        plt.tight_layout()
        plt.savefig(save_name)
        print(f"[Feature Map] Saved to {save_name}")
        plt.close()


# -------------------------------------------------
# 3. Grad-CAM 可视化 (热力图)
# -------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        # 注册 Hook
        # 1. 获取前向传播的特征图
        target_layer.register_forward_hook(self.save_activation)
        # 2. 获取反向传播的梯度
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        # 前向传播
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # 梯度 [1, C, H, W] -> Global Average Pooling over H,W -> [1, C, 1, 1]
        gradients = self.gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # 特征图 [1, C, H, W]
        activations = self.activations.detach()

        # 加权
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # 在 Channel 维度求平均 -> [1, H, W]
        heatmap = torch.mean(activations, dim=1).squeeze()

        # ReLU: 只需要正向影响
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)

        # 归一化
        heatmap /= (np.max(heatmap) + 1e-9)
        return heatmap, class_idx


def overlay_cam(raw_img, heatmap, save_name="gradcam.png", pred_label=""):
    # 将 heatmap 调整为原图大小
    raw_img = raw_img.resize((256, 256))  # 稍微放大一点方便看
    heatmap = cv2.resize(heatmap, (256, 256))

    # 转为伪彩色
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 转换原图为 numpy
    raw_img_np = np.array(raw_img)
    # OpenCV 使用 BGR，PIL 使用 RGB，转换一下
    raw_img_np = cv2.cvtColor(raw_img_np, cv2.COLOR_RGB2BGR)

    # 叠加
    superimposed_img = heatmap_color * 0.4 + raw_img_np * 0.6

    # 加文字
    cv2.putText(superimposed_img, f"Pred: {pred_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(save_name, superimposed_img)
    print(f"[Grad-CAM] Saved to {save_name}")


# -------------------------------------------------
# 4. 主执行逻辑
# -------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 通过 utils 统一获取数据集路径
    base_dataset_dir, root_dir, train_csv_path, valid_csv_path = get_dataset_paths()

    # 使用 valid 集中的一张 fake 样本作为测试图片
    test_img_path = os.path.join(root_dir, "valid", "fake", "0.jpg")

    # 如果找不到上面的路径，为了防止报错，请先手动指定一张图片：
    if not os.path.exists(test_img_path):
        print(f"警告: 找不到默认图片 {test_img_path}")
        print("请检查 rvf10k/valid/fake/ 目录下是否存在 0.jpg，或在代码中修改 test_img_path 为你本地存在的图片路径！")
        exit()

    weight_path = "best_resnet18_rvf10k.pth"
    if not os.path.exists(weight_path):
        print("警告: 找不到权重文件，请先运行 train.py")
        exit()

    # 2. 加载模型
    model = load_model(weight_path, device)

    # 3. 预处理图片
    raw_img, input_tensor = preprocess_image(test_img_path)
    input_tensor = input_tensor.to(device)

    # -----------------------
    # 任务 A: 特征图可视化
    # -----------------------
    # 观察 Layer1 (浅层特征：边缘、颜色)
    viz_feat = FeatureMapVisualizer(model, model.layer1)
    viz_feat.visualize(input_tensor, save_name="analysis_feature_map_layer1.png")

    # -----------------------
    # 任务 B: Grad-CAM 可视化
    # -----------------------
    # 观察 Layer4 (深层特征：语义信息)
    grad_cam = GradCAM(model, model.layer4)
    heatmap, pred_idx = grad_cam.generate(input_tensor)

    classes = {0: 'Fake', 1: 'Real'}
    overlay_cam(raw_img, heatmap, save_name="analysis_gradcam.png", pred_label=classes[pred_idx])