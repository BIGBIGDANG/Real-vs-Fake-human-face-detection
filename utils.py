import os


def get_dataset_paths():
    """
    返回数据集相关的路径：
      - base_dataset_dir: 数据集根目录，包含 CSV 和 rvf10k 文件夹
      - root_dir:         rvf10k 目录，包含 train/ 和 valid/ 子目录
      - train_csv_path:   训练集 CSV 路径
      - valid_csv_path:   验证集 CSV 路径
    """
    base_dataset_dir = "/root/autodl-tmp/Real-vs-Fake-human-face-detection/dataset"
    root_dir = os.path.join(base_dataset_dir, "rvf10k")
    train_csv_path = os.path.join(base_dataset_dir, "train.csv")
    valid_csv_path = os.path.join(base_dataset_dir, "valid.csv")
    return base_dataset_dir, root_dir, train_csv_path, valid_csv_path
