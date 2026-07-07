"""
短时降雨量预测 - 经典 LSTM 基线训练脚本

流程:
  1. 生成/加载合成数据
  2. 训练 LSTM 基线
  3. 评估并输出 baseline_report.json
  4. 保存模型
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

from data.synthetic_rainfall import generate_synthetic_rainfall
from data.dataset import RainfallDataset, load_scaler
from models.classical_lstm import ClassicalLSTM, train_classical_lstm, evaluate_model


def set_seed(seed=42):
    """设置全局随机种子以确保可复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # ---- 配置 ----
    SEED = 42
    SEQ_LEN = 10
    PRED_LEN = 10
    FEATURES = 7
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    N_EPOCHS = 50
    BATCH_SIZE = 64
    LR = 0.001

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
    REPORT_DIR = os.path.join(PROJECT_ROOT, "experiments", "stage1_scope_baseline")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    DATA_PATH = os.path.join(DATA_DIR, "rainfall_data.npz")
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, "classical_lstm.pth")
    REPORT_PATH = os.path.join(REPORT_DIR, "baseline_report.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[配置] 设备: {device}, 种子: {SEED}")

    # ---- 1. 设置随机种子 ----
    set_seed(SEED)

    # ---- 2. 生成数据 ----
    print("\n===== 数据生成 =====")
    if not os.path.exists(DATA_PATH):
        generate_synthetic_rainfall(
            n_samples=7000,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            seed=SEED,
            save_path=DATA_PATH,
        )
    else:
        print(f"  数据文件已存在: {DATA_PATH}")

    # ---- 3. 加载数据集 ----
    print("\n===== 加载数据集 =====")
    train_ds = RainfallDataset(DATA_PATH, split="train")
    val_ds = RainfallDataset(DATA_PATH, split="val")
    test_ds = RainfallDataset(DATA_PATH, split="test")
    scaler = load_scaler(DATA_PATH)

    # ---- 4. 创建模型 ----
    print("\n===== 创建模型 =====")
    model = ClassicalLSTM(
        input_size=FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")

    # ---- 5. 训练 ----
    print("\n===== 训练 =====")
    t0 = time.time()
    train_result = train_classical_lstm(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=device,
        patience=15,
    )
    train_time = time.time() - t0
    print(f"  训练耗时: {train_time:.1f}s")

    # ---- 6. 保存模型 ----
    torch.save(train_result["model"].state_dict(), MODEL_PATH)
    print(f"\n  模型已保存: {MODEL_PATH}")

    # ---- 7. 评估 ----
    print("\n===== 评估 =====")
    metrics = evaluate_model(
        model=train_result["model"],
        test_dataset=test_ds,
        scaler=scaler,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # ---- 8. 保存训练曲线 ----
    curves_path = os.path.join(ARTIFACTS_DIR, "train_curves.npz")
    np.savez(
        curves_path,
        train_losses=np.array(train_result["train_losses"]),
        val_losses=np.array(train_result["val_losses"]),
    )
    print(f"  训练曲线已保存: {curves_path}")

    # ---- 9. 生成 baseline_report.json ----
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
        "command": "python train_baseline.py",
        "artifact_paths": ["artifacts/classical_lstm.pth"],
        "seed": SEED,
        "backend": "PyTorch LSTM",
        "model_config": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "features": FEATURES,
        },
        "training_config": {
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "best_epoch": train_result["best_epoch"],
            "best_val_loss": round(train_result["best_val_loss"], 6),
            "train_time_s": round(train_time, 1),
        },
        "limitations": ["使用合成数据，需真实数据验证"],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {REPORT_PATH}")

    # ---- 10. 汇总 ----
    print("\n" + "=" * 50)
    print("  训练完成!")
    print(f"  RMSE:  {metrics['RMSE']:.4f} mm")
    print(f"  MAE:   {metrics['MAE']:.4f} mm")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")
    print(f"  最佳 epoch: {train_result['best_epoch']}")
    print("=" * 50)

    return report


if __name__ == "__main__":
    main()
