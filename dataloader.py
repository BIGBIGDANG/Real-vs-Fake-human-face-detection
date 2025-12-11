import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FaceRealFakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file:  'rvf10k/train.csv' 或 'rvf10k/valid.csv'
        root_dir:  'rvf10k'（包含 train/ 和 valid/ 的那一层）
        """
        df = pd.read_csv(csv_file)

        # 丢掉第一列无用索引
        first_col = df.columns[0]
        if first_col == "" or first_col.startswith("Unnamed"):
            df = df.drop(columns=[first_col])

        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rel_path = row["path"]  # 例如 'train/real/28609.jpg'
        label = int(row["label"])  # 0 / 1

        img_path = os.path.join(self.root_dir, rel_path)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import get_dataset_paths

img_size = 128  # 或 256

train_tf = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

valid_tf = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

base_dataset_dir = "/root/autodl-tmp/Real-vs-Fake-human-face-detection/dataset"
_, root_dir, train_csv_path, valid_csv_path = get_dataset_paths()

train_dataset = FaceRealFakeDataset(train_csv_path, root_dir, transform=train_tf)
valid_dataset = FaceRealFakeDataset(valid_csv_path, root_dir, transform=valid_tf)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)

# 简单检查一下
print(len(train_dataset), len(valid_dataset))
img, label = train_dataset[0]
print(img.shape, label)

row0 = train_dataset.df.iloc[6000]
print(row0["path"], row0["label"])

img, label = train_dataset[6000]
print(label)  # 应该和 row0['label'] 一样
