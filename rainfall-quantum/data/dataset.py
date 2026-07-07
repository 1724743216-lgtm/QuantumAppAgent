"""
降雨量预测 PyTorch Dataset

从 rainfall_data.npz 加载归一化后的 (input_seq, target_seq) 对。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os


class RainfallDataset(Dataset):
    """
    降雨量时序预测数据集。

    Parameters
    ----------
    npz_path : str
        rainfall_data.npz 的路径
    split : str
        'train', 'val', 或 'test'
    """

    SPLIT_KEYS = {
        "train": ("train_inputs", "train_targets"),
        "val": ("val_inputs", "val_targets"),
        "test": ("test_inputs", "test_targets"),
    }

    def __init__(self, npz_path, split="train"):
        assert split in self.SPLIT_KEYS, f"split 必须是 {list(self.SPLIT_KEYS.keys())}"
        inp_key, tgt_key = self.SPLIT_KEYS[split]

        data = np.load(npz_path)
        self.inputs = torch.from_numpy(data[inp_key])    # (N, seq_len, 7)
        self.targets = torch.from_numpy(data[tgt_key])    # (N, pred_len)
        self.split = split

        print(f"[RainfallDataset] split={split}, samples={len(self)}, "
              f"input_shape={tuple(self.inputs.shape)}, target_shape={tuple(self.targets.shape)}")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_scaler(npz_path):
    """从 npz 文件加载 scaler 参数，用于反归一化。"""
    data = np.load(npz_path)
    return {
        "feat_min": data["feat_min"],
        "feat_max": data["feat_max"],
        "feat_range": data["feat_range"],
        "tgt_min": float(data["tgt_min"]),
        "tgt_max": float(data["tgt_max"]),
        "tgt_range": float(data["tgt_range"]),
    }
