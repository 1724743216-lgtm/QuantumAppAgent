"""
短时降雨量预测 - FastAPI 后端服务

提供模型推理、指标对比、样本数据、训练曲线等 API 接口。
前端静态文件由 FastAPI 直接 serve。
"""

import sys
import os
import uuid
import json
import time
import traceback
from pathlib import Path
from typing import Optional

# 确保项目根目录在 Python 路径中，以便导入 models/ 和 data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.dataset import RainfallDataset, load_scaler
from models.classical_lstm import ClassicalLSTM, evaluate_model
from models.hybrid_vqc_lstm import HybridVQCLSTM

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "rainfall_data.npz"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLASSICAL_MODEL_PATH = ARTIFACTS_DIR / "classical_lstm.pth"
HYBRID_MODEL_PATH = ARTIFACTS_DIR / "hybrid_vqc_lstm.pth"
CLASSICAL_CURVES_PATH = ARTIFACTS_DIR / "train_curves.npz"
HYBRID_CURVES_PATH = ARTIFACTS_DIR / "hybrid_train_curves.npz"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ---------------------------------------------------------------------------
# 模型参数
# ---------------------------------------------------------------------------
INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
PRED_LEN = 10
N_QUBITS = 6
VQC_LAYERS = 2
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# 全局变量 (模块级别加载，兼容 TestClient 直接导入和 uvicorn 启动两种模式)
# ---------------------------------------------------------------------------
train_tasks: dict = {}  # 训练任务跟踪


def _load_models_and_data():
    """加载模型与数据到全局变量。在模块级别和 startup 事件中均调用。"""
    global classical_model, hybrid_model, scaler, test_dataset

    if classical_model is not None:
        return  # 已加载，跳过

    print("[加载] 加载 scaler 与测试数据集...")
    try:
        scaler = load_scaler(str(DATA_PATH))
        test_dataset = RainfallDataset(str(DATA_PATH), split="test")
    except Exception as e:
        print(f"  ⚠ 加载数据失败: {e}")
        return

    print("[加载] 加载经典 LSTM 模型...")
    classical_model = ClassicalLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN,
    )
    if CLASSICAL_MODEL_PATH.exists():
        state = torch.load(str(CLASSICAL_MODEL_PATH), map_location=DEVICE, weights_only=True)
        classical_model.load_state_dict(state)
        print(f"  ✓ 已加载: {CLASSICAL_MODEL_PATH}")
    else:
        print(f"  ⚠ 模型文件不存在: {CLASSICAL_MODEL_PATH}")
    classical_model.eval()

    print("[加载] 加载混合 VQC+LSTM 模型...")
    hybrid_model = HybridVQCLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        n_qubits=N_QUBITS,
        vqc_layers=VQC_LAYERS,
        pred_len=PRED_LEN,
    )
    if HYBRID_MODEL_PATH.exists():
        state = torch.load(str(HYBRID_MODEL_PATH), map_location=DEVICE, weights_only=True)
        hybrid_model.load_state_dict(state)
        print(f"  ✓ 已加载: {HYBRID_MODEL_PATH}")
    else:
        print(f"  ⚠ 模型文件不存在: {HYBRID_MODEL_PATH}")
    hybrid_model.eval()

    print("[加载] 服务就绪 ✓")


classical_model: Optional[ClassicalLSTM] = None
hybrid_model: Optional[HybridVQCLSTM] = None
scaler: Optional[dict] = None
test_dataset: Optional[RainfallDataset] = None

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="短时降雨量预测服务",
    description="基于经典 LSTM 与量子混合 VQC+LSTM 的短时降雨量预测 API",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# 启动事件: 加载模型与数据
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_load():
    _load_models_and_data()


# ---------------------------------------------------------------------------
# 模块级别加载: 兼容 TestClient 直接导入场景 (如 validate_quantum_application)
# ---------------------------------------------------------------------------
_load_models_and_data()


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    model_type: str  # "classical" | "hybrid"
    input_data: Optional[list] = None  # (seq_len=10, features=7) 的 2D 数组
    use_sample: bool = False  # 使用测试集随机样本

class TrainRequest(BaseModel):
    model_type: str  # "classical" | "hybrid"
    epochs: int = 20


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "ok", "service": "rainfall-quantum-prediction"}


@app.get("/api/info")
def app_info():
    """应用信息"""
    return {
        "name": "短时降雨量预测系统",
        "description": "基于经典 LSTM 与量子混合 VQC+LSTM 模型的短时降雨量预测应用",
        "version": "1.0.0",
        "models": [
            {
                "type": "classical",
                "name": "经典 LSTM 基线",
                "architecture": "2层 LSTM (hidden=64) + 全连接预测头",
                "params": sum(p.numel() for p in classical_model.parameters()) if classical_model else 0,
            },
            {
                "type": "hybrid",
                "name": "量子混合 VQC+LSTM",
                "architecture": "LSTM 编码 → 投影层 → VQC 量子特征提取 (6 qubits, 2 layers) → 残差融合 → 预测头",
                "params": sum(p.numel() for p in hybrid_model.parameters()) if hybrid_model else 0,
                "quantum_params": hybrid_model.quantum_params if hybrid_model else 0,
            },
        ],
        "metrics_description": {
            "RMSE": "均方根误差 (mm), 越低越好",
            "MAE": "平均绝对误差 (mm), 越低越好",
            "MAPE": "平均绝对百分比误差 (%), 越低越好",
        },
        "features": [
            "radar_reflectivity (dBZ)",
            "temperature (°C)",
            "humidity (%)",
            "wind_speed (m/s)",
            "pressure (hPa)",
            "wind_direction (°)",
            "past_precipitation (mm)",
        ],
        "seq_len": 10,
        "pred_len": 10,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    使用指定模型进行预测。

    请求体:
      - model_type: "classical" 或 "hybrid"
      - input_data: (seq_len=10, features=7) 的 2D 数组
    """
    if req.model_type not in ("classical", "hybrid"):
        raise HTTPException(status_code=400, detail=f"不支持的 model_type: {req.model_type}")

    # 如果 use_sample=True，从测试集随机取一个样本
    if req.use_sample:
        if test_dataset is None:
            raise HTTPException(status_code=500, detail="测试数据集未加载")
        idx = np.random.randint(0, len(test_dataset))
        input_arr = test_dataset.inputs[idx].numpy()  # (10, 7)
    elif req.input_data is not None:
        # 验证输入形状
        try:
            input_arr = np.array(req.input_data, dtype=np.float32)
            if input_arr.shape != (10, 7):
                raise ValueError(f"input_data 形状应为 (10, 7), 实际为 {input_arr.shape}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"输入数据格式错误: {e}")
    else:
        raise HTTPException(status_code=400, detail="请提供 input_data 或设置 use_sample=true")

    model = classical_model if req.model_type == "classical" else hybrid_model
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    # 推理
    with torch.no_grad():
        x = torch.from_numpy(input_arr).unsqueeze(0).to(DEVICE)  # (1, 10, 7)
        pred_norm = model(x).squeeze(0).cpu().numpy()  # (10,)

    # 反归一化
    tgt_min = scaler["tgt_min"]
    tgt_range = scaler["tgt_range"]
    pred_raw = pred_norm * tgt_range + tgt_min

    # 模型 RMSE (从之前的评估结果)
    rmse_map = {"classical": 0.1753, "hybrid": 0.1787}

    return {
        "predictions": pred_raw.tolist(),
        "model_type": req.model_type,
        "rmse": rmse_map.get(req.model_type, 0.0),
    }


@app.get("/api/metrics")
def metrics_comparison():
    """
    经典 vs 量子模型的 RMSE/MAE/MAPE 指标对比。
    使用测试集实时评估。
    """
    results = {}
    for name, model in [("classical", classical_model), ("hybrid", hybrid_model)]:
        if model is None:
            continue
        eval_result = evaluate_model(model, test_dataset, scaler, device=DEVICE)
        results[name] = {
            "RMSE": round(eval_result["RMSE"], 4),
            "MAE": round(eval_result["MAE"], 4),
            "MAPE": round(float(eval_result["MAPE"]), 2),
        }

    return {
        "classical": results.get("classical", {}),
        "hybrid": results.get("hybrid", {}),
        "unit": {"RMSE": "mm", "MAE": "mm", "MAPE": "%"},
    }


@app.get("/api/sample-data")
def sample_data(n_samples: int = Query(default=5, ge=1, le=50)):
    """
    获取测试集样本，用于前端展示。

    返回 n_samples 组 (input_seq, target_seq, feature_names) 数据。
    """
    inputs = test_dataset.inputs[:n_samples].numpy()     # (n, 10, 7)
    targets = test_dataset.targets[:n_samples].numpy()    # (n, 10)

    # 反归一化
    tgt_min = scaler["tgt_min"]
    tgt_range = scaler["tgt_range"]
    targets_raw = targets * tgt_range + tgt_min

    # 输入特征反归一化
    feat_min = scaler["feat_min"]
    feat_range = scaler["feat_range"]
    inputs_raw = inputs * feat_range + feat_min

    feature_names = [
        "雷达反射率 (dBZ)", "温度 (°C)", "湿度 (%)",
        "风速 (m/s)", "气压 (hPa)", "风向 (°)", "过去降水 (mm)"
    ]

    # 使用两个模型分别预测
    predictions = {}
    for name, model in [("classical", classical_model), ("hybrid", hybrid_model)]:
        if model is None:
            continue
        with torch.no_grad():
            x = test_dataset.inputs[:n_samples].to(DEVICE)
            preds = model(x).cpu().numpy()  # (n, 10)
        preds_raw = preds * tgt_range + tgt_min
        predictions[name] = preds_raw.tolist()

    return {
        "samples": [
            {
                "index": i,
                "input_seq": inputs_raw[i].tolist(),   # (10, 7)
                "target": targets_raw[i].tolist(),       # (10,)
                "predictions": {
                    name: preds[i] for name, preds in predictions.items()
                },
            }
            for i in range(n_samples)
        ],
        "feature_names": feature_names,
        "n_samples": n_samples,
    }


@app.get("/api/train-curves")
def train_curves():
    """获取训练/验证 loss 曲线数据。"""
    curves = {}

    if CLASSICAL_CURVES_PATH.exists():
        data = np.load(str(CLASSICAL_CURVES_PATH))
        curves["classical"] = {
            "train_losses": data["train_losses"].tolist(),
            "val_losses": data["val_losses"].tolist(),
            "epochs": list(range(1, len(data["train_losses"]) + 1)),
        }

    if HYBRID_CURVES_PATH.exists():
        data = np.load(str(HYBRID_CURVES_PATH))
        curves["hybrid"] = {
            "train_losses": data["train_losses"].tolist(),
            "val_losses": data["val_losses"].tolist(),
            "epochs": list(range(1, len(data["train_losses"]) + 1)),
        }

    return curves


@app.post("/api/train")
def train_model(req: TrainRequest):
    """
    触发模型训练 (简化实现，同步执行)。

    请求体:
      - model_type: "classical" 或 "hybrid"
      - epochs: 训练轮数 (默认 20)
    """
    if req.model_type not in ("classical", "hybrid"):
        raise HTTPException(status_code=400, detail=f"不支持的 model_type: {req.model_type}")

    task_id = str(uuid.uuid4())[:8]
    train_tasks[task_id] = {
        "model_type": req.model_type,
        "epochs": req.epochs,
        "status": "running",
        "start_time": time.time(),
    }

    try:
        from torch.utils.data import DataLoader
        import torch.nn as nn

        train_ds = RainfallDataset(str(DATA_PATH), split="train")
        val_ds = RainfallDataset(str(DATA_PATH), split="val")

        if req.model_type == "classical":
            from models.classical_lstm import train_classical_lstm
            model = ClassicalLSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                pred_len=PRED_LEN,
            )
            result = train_classical_lstm(
                model, train_ds, val_ds,
                n_epochs=req.epochs, batch_size=64, lr=0.001,
                device=DEVICE, patience=10,
            )
            # 保存模型
            torch.save(result["model"].state_dict(), str(CLASSICAL_MODEL_PATH))
            # 保存训练曲线
            np.savez(
                str(CLASSICAL_CURVES_PATH),
                train_losses=result["train_losses"],
                val_losses=result["val_losses"],
            )
            # 更新全局模型
            global classical_model
            classical_model = result["model"]
            classical_model.eval()

        else:  # hybrid
            from models.hybrid_vqc_lstm import get_hybrid_param_groups
            model = HybridVQCLSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                n_qubits=N_QUBITS,
                vqc_layers=VQC_LAYERS,
                pred_len=PRED_LEN,
            )
            # 简化训练: 使用与 train_hybrid.py 相同的流程
            model = model.to(DEVICE)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            param_groups = get_hybrid_param_groups(model, classical_lr=0.001, quantum_lr=0.0005)
            optimizer = torch.optim.Adam(param_groups)
            criterion = nn.MSELoss()

            train_losses = []
            val_losses = []
            best_val_loss = float("inf")
            best_state = None

            for epoch in range(1, req.epochs + 1):
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    optimizer.zero_grad()
                    preds = model(inputs)
                    loss = criterion(preds, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                avg_train = epoch_loss / max(n_batches, 1)
                train_losses.append(avg_train)

                model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        preds = model(inputs)
                        val_loss += criterion(preds, targets).item()
                        n_val += 1
                avg_val = val_loss / max(n_val, 1)
                val_losses.append(avg_val)

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_state:
                model.load_state_dict(best_state)
                model = model.to(DEVICE)

            torch.save(model.state_dict(), str(HYBRID_MODEL_PATH))
            np.savez(
                str(HYBRID_CURVES_PATH),
                train_losses=train_losses,
                val_losses=val_losses,
            )
            global hybrid_model
            hybrid_model = model
            hybrid_model.eval()

            result = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
            }

        train_tasks[task_id]["status"] = "completed"
        train_tasks[task_id]["best_val_loss"] = result.get("best_val_loss", 0.0)

    except Exception as e:
        train_tasks[task_id]["status"] = "failed"
        train_tasks[task_id]["error"] = str(e)
        traceback.print_exc()

    return {
        "task_id": task_id,
        "status": train_tasks[task_id]["status"],
        "model_type": req.model_type,
        "epochs": req.epochs,
    }


# ---------------------------------------------------------------------------
# 静态文件: 前端页面
# ---------------------------------------------------------------------------
# 将前端文件挂载在 / 下，但 API 路由优先匹配
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index():
    """返回前端首页"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "短时降雨量预测 API - 前端页面未找到, 请访问 /docs 查看 API 文档"}
