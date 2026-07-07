"""
短时降雨量预测 - 混合 VQC+LSTM 训练脚本

流程:
  1. 加载合成数据 (与经典基线相同)
  2. 构建 HybridVQCLSTM 模型
  3. 差异化学习率训练 (量子参数 lr=0.0005, 经典参数 lr=0.001)
  4. 评估并输出 quantum_report.json
  5. 保存模型
"""

import sys
import os
import json
import time

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import RainfallDataset, load_scaler
from models.hybrid_vqc_lstm import HybridVQCLSTM, get_hybrid_param_groups
from models.classical_lstm import evaluate_model


def set_seed(seed=42):
    """设置全局随机种子以确保可复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_hybrid_model(
    model,
    train_dataset,
    val_dataset,
    n_epochs=30,
    batch_size=32,
    classical_lr=0.001,
    quantum_lr=0.0005,
    device="cpu",
    patience=15,
):
    """
    训练混合 VQC+LSTM 模型。

    Parameters
    ----------
    model : HybridVQCLSTM
    train_dataset : RainfallDataset
    val_dataset : RainfallDataset
    n_epochs : int
    batch_size : int
    classical_lr : float
        经典参数学习率
    quantum_lr : float
        量子变分参数学习率
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

    # 差异化学习率
    param_groups = get_hybrid_param_groups(model, classical_lr=classical_lr, quantum_lr=quantum_lr)
    optimizer = torch.optim.Adam(param_groups)
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
        t_epoch_start = time.time()

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
        t_epoch = time.time() - t_epoch_start

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

        print(f"  Epoch {epoch:3d}/{n_epochs} | train_loss={avg_train_loss:.6f} | "
              f"val_loss={avg_val_loss:.6f} | best_val={best_val_loss:.6f} | "
              f"time={t_epoch:.1f}s")

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


def main():
    # ---- 配置 ----
    SEED = 42
    SEQ_LEN = 10
    PRED_LEN = 10
    FEATURES = 7
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    N_QUBITS = 6
    VQC_LAYERS = 2
    N_EPOCHS = 30
    BATCH_SIZE = 32
    CLASSICAL_LR = 0.001
    QUANTUM_LR = 0.0005

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
    REPORT_DIR = os.path.join(PROJECT_ROOT, "experiments", "stage2_quantum_method")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    DATA_PATH = os.path.join(DATA_DIR, "rainfall_data.npz")
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_vqc_lstm.pth")
    REPORT_PATH = os.path.join(REPORT_DIR, "quantum_report.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[配置] 设备: {device}, 种子: {SEED}")
    print(f"[配置] 量子比特: {N_QUBITS}, VQC层数: {VQC_LAYERS}")

    # ---- 1. 设置随机种子 ----
    set_seed(SEED)

    # ---- 2. 加载数据集 ----
    print("\n===== 加载数据集 =====")
    if not os.path.exists(DATA_PATH):
        print(f"  错误: 数据文件不存在 {DATA_PATH}")
        print("  请先运行 python train_baseline.py 生成数据")
        return

    train_ds = RainfallDataset(DATA_PATH, split="train")
    val_ds = RainfallDataset(DATA_PATH, split="val")
    test_ds = RainfallDataset(DATA_PATH, split="test")
    scaler = load_scaler(DATA_PATH)

    # ---- 3. 创建模型 ----
    print("\n===== 创建混合模型 =====")
    model = HybridVQCLSTM(
        input_size=FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        n_qubits=N_QUBITS,
        vqc_layers=VQC_LAYERS,
        pred_len=PRED_LEN,
    )

    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = model.quantum_params
    classical_params = total_params - quantum_params
    print(f"  总参数量: {total_params:,}")
    print(f"  量子参数量: {quantum_params:,}")
    print(f"  经典参数量: {classical_params:,}")
    print(f"  电路深度(理论): {model.vqc.circuit_depth}")

    # ---- 4. 训练 ----
    print("\n===== 训练 =====")
    t0 = time.time()
    train_result = train_hybrid_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        classical_lr=CLASSICAL_LR,
        quantum_lr=QUANTUM_LR,
        device=device,
        patience=15,
    )
    train_time = time.time() - t0
    print(f"  训练耗时: {train_time:.1f}s")

    # ---- 5. 保存模型 ----
    torch.save(train_result["model"].state_dict(), MODEL_PATH)
    print(f"\n  模型已保存: {MODEL_PATH}")

    # ---- 6. 评估 ----
    print("\n===== 评估 =====")
    metrics = evaluate_model(
        model=train_result["model"],
        test_dataset=test_ds,
        scaler=scaler,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # ---- 7. 保存训练曲线 ----
    curves_path = os.path.join(ARTIFACTS_DIR, "hybrid_train_curves.npz")
    np.savez(
        curves_path,
        train_losses=np.array(train_result["train_losses"]),
        val_losses=np.array(train_result["val_losses"]),
    )
    print(f"  训练曲线已保存: {curves_path}")

    # ---- 8. 生成 quantum_report.json ----
    baseline_rmse = 0.1753
    rmse_diff = metrics["RMSE"] - baseline_rmse
    vs_baseline_str = (
        f"量子RMSE={metrics['RMSE']:.4f}, 经典RMSE={baseline_rmse:.4f}, "
        f"差值={rmse_diff:+.4f} ({'优于' if rmse_diff < 0 else '劣于'}经典基线)"
    )

    report = {
        "task": "短时降雨量预测",
        "data": "synthetic_rainfall (5000/1000/1000 split, seed=42)",
        "primary_metric": "RMSE",
        "higher_is_better": False,
        "value": round(metrics["RMSE"], 4),
        "auxiliary_metrics": {
            "MAE": round(metrics["MAE"], 4),
            "MAPE": round(metrics["MAPE"], 2),
        },
        "command": "python train_hybrid.py",
        "artifact_paths": ["artifacts/hybrid_vqc_lstm.pth"],
        "seed": SEED,
        "backend": "cqlib StatevectorSimulator",
        "qubits": N_QUBITS,
        "circuit_depth": model.vqc.circuit_depth,
        "quantum_layers": VQC_LAYERS,
        "quantum_params": quantum_params,
        "model_config": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "features": FEATURES,
            "n_qubits": N_QUBITS,
            "vqc_layers": VQC_LAYERS,
        },
        "baseline_rmse": baseline_rmse,
        "training_config": {
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": CLASSICAL_LR,
            "quantum_lr": QUANTUM_LR,
            "train_time_s": round(train_time, 1),
            "best_epoch": train_result["best_epoch"],
        },
        "limitations": [
            "量子层梯度通过detach断开",
            "使用合成数据",
            "模拟器运行",
        ],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {REPORT_PATH}")

    # ---- 9. 汇总 ----
    print("\n" + "=" * 50)
    print("  混合模型训练完成!")
    print(f"  RMSE:  {metrics['RMSE']:.4f} mm (基线: {baseline_rmse:.4f})")
    print(f"  MAE:   {metrics['MAE']:.4f} mm")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")
    print(f"  最佳 epoch: {train_result['best_epoch']}")
    print(f"  训练耗时: {train_time:.1f}s")
    print(f"  量子参数: {quantum_params}, 总参数: {total_params}")
    print("=" * 50)

    return report


if __name__ == "__main__":
    main()
