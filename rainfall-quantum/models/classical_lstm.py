"""
经典 LSTM 降雨量预测基线模型

架构:
  - 输入: (batch, seq_len=10, features=7)
  - LSTM 编码器: 2 层, hidden_size=64
  - 全连接预测头: -> (batch, pred_len=10)

包含训练函数 train_classical_lstm() 和评估函数 evaluate_model()。
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


class ClassicalLSTM(nn.Module):
    """
    多层 LSTM + 全连接预测头，用于短时降雨量预测。

    Parameters
    ----------
    input_size : int
        输入特征维度 (7)
    hidden_size : int
        LSTM 隐层大小 (64)
    num_layers : int
        LSTM 层数 (2)
    pred_len : int
        预测长度 (10)
    dropout : float
        LSTM 层间 dropout
    """

    def __init__(self, input_size=7, hidden_size=64, num_layers=2, pred_len=10, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor
            (batch, pred_len) 归一化空间的预测值
        """
        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)

        # 取最后时间步的隐状态
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # 全连接预测
        pred = self.fc(last_hidden)  # (batch, pred_len)
        return pred


def train_classical_lstm(
    model,
    train_dataset,
    val_dataset,
    n_epochs=50,
    batch_size=64,
    lr=0.001,
    device="cpu",
    patience=10,
):
    """
    训练 LSTM 模型。

    Parameters
    ----------
    model : ClassicalLSTM
    train_dataset : RainfallDataset
    val_dataset : RainfallDataset
    n_epochs : int
    batch_size : int
    lr : float
    device : str
    patience : int
        早停耐心值

    Returns
    -------
    dict : 包含模型、训练曲线、最佳 epoch 等信息
    """
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # ---- Train ----
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)

        # ---- Validate ----
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = model(inputs)
                loss = criterion(preds, targets)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        # ---- Early stopping check ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | best_val={best_val_loss:.6f}")

        if no_improve >= patience:
            print(f"  [早停] 连续 {patience} epoch 无改善，在 epoch {epoch} 停止")
            break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    print(f"  最佳 epoch: {best_epoch}, 最佳 val_loss: {best_val_loss:.6f}")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def evaluate_model(model, test_dataset, scaler, device="cpu", batch_size=64):
    """
    评估模型，返回原始 (非归一化) 空间的指标。

    Parameters
    ----------
    model : ClassicalLSTM
    test_dataset : RainfallDataset
    scaler : dict
        包含 tgt_min, tgt_max, tgt_range
    device : str
    batch_size : int

    Returns
    -------
    dict : {"RMSE": float, "MAE": float, "MAPE": float, "predictions": np.ndarray, "ground_truth": np.ndarray}
    """
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    preds_norm = np.concatenate(all_preds, axis=0)   # (N, pred_len) 归一化空间
    targets_norm = np.concatenate(all_targets, axis=0)  # (N, pred_len)

    # 反归一化到原始空间
    tgt_min = scaler["tgt_min"]
    tgt_range = scaler["tgt_range"]
    preds_raw = preds_norm * tgt_range + tgt_min
    targets_raw = targets_norm * tgt_range + tgt_min

    # 计算指标 (在原始空间)
    diff = preds_raw - targets_raw
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))

    # MAPE: 避免除以零，过滤掉 target < 0.01 的样本
    mask = np.abs(targets_raw) > 0.01
    if mask.sum() > 0:
        mape = np.mean(np.abs(diff[mask] / targets_raw[mask])) * 100.0
    else:
        mape = float("nan")

    print(f"[评估] RMSE={rmse:.4f} mm, MAE={mae:.4f} mm, MAPE={mape:.2f}%")

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "predictions": preds_raw,
        "ground_truth": targets_raw,
    }
