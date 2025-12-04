import torch
import torch.nn as nn
from torchvision import models

from dataloader import *

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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
# 或者：torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()   # 确保是 long 类型

        outputs = model(imgs)              # [B, 2]
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

num_epochs = 20
best_val_acc = 0.0

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss,   val_acc   = evaluate(model, valid_loader, criterion, device)

    print(f"Epoch [{epoch}/{num_epochs}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # 保存最好的一版
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet18_rvf10k.pth")
        print(f"  >>> best model updated, val_acc={best_val_acc:.4f}")
