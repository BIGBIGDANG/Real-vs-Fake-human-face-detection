import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 如果 Kaggle 能下权重，用预训练：
# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# 如果没网/嫌麻烦，就用随机初始化：
model = models.resnet18(weights=None)

# 改成 2 类输出（fake / real）
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)

model = model.to(device)
print(model)  # 简单看看结构