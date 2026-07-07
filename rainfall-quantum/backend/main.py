"""
短时降雨量预测 - FastAPI 主应用

提供模型推理、指标对比、样本数据、训练触发等 API 接口。
前端静态文件由 FastAPI 直接 serve。

运行方式:
  cd /rainfall-quantum && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
  或: python backend/main.py
"""

import sys
import threading
from pathlib import Path
from typing import Optional

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.prediction_service import prediction_service
from backend.training_service import training_service

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="短时降雨量预测服务",
    description="基于经典 LSTM 与量子混合 VQC+LSTM 的短时降雨量预测 API",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS 中间件
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    model_type: str = Field(..., pattern=r"^(classical|hybrid)$", description="模型类型: classical 或 hybrid")
    input_data: Optional[list] = Field(None, description="输入数据, 形状 (10, 7)")
    use_sample: bool = Field(False, description="使用示例数据，为 true 时忽略 input_data")


class TrainRequest(BaseModel):
    model_type: str = Field(..., pattern=r"^(classical|hybrid)$", description="模型类型: classical 或 hybrid")
    epochs: int = Field(50, ge=1, le=500, description="训练轮数")


# ---------------------------------------------------------------------------
# 启动事件: 加载模型与数据
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_load():
    prediction_service.load()


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "ok", "service": "rainfall-quantum-prediction"}


@app.get("/api/info")
def app_info():
    """应用信息（模型列表、数据集信息）"""
    classical = prediction_service.classical_model
    hybrid = prediction_service.hybrid_model

    return {
        "name": "短时降雨量预测系统",
        "description": "基于经典 LSTM 与量子混合 VQC+LSTM 模型的短时降雨量预测应用",
        "version": "1.0.0",
        "models": [
            {
                "type": "classical",
                "name": "经典 LSTM 基线",
                "architecture": "2层 LSTM (hidden=64) + 全连接预测头",
                "params": sum(p.numel() for p in classical.parameters()) if classical else 0,
            },
            {
                "type": "hybrid",
                "name": "量子混合 VQC+LSTM",
                "architecture": "LSTM 编码 → 投影层 → VQC 量子特征提取 (6 qubits, 2 layers) → 残差融合 → 预测头",
                "params": sum(p.numel() for p in hybrid.parameters()) if hybrid else 0,
                "quantum_params": hybrid.quantum_params if hybrid else 0,
            },
        ],
        "dataset": {
            "name": "synthetic_rainfall",
            "split": "5000/1000/1000 (train/val/test)",
            "seq_len": 10,
            "pred_len": 10,
            "features": 7,
        },
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
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    使用指定模型进行预测。

    - model_type: "classical" 或 "hybrid"
    - input_data: (10, 7) 的 2D 数组，或设置 use_sample=true 使用示例数据
    """
    try:
        if req.use_sample:
            result = prediction_service.predict(model_type=req.model_type, use_sample=True)
        else:
            if req.input_data is None:
                raise HTTPException(
                    status_code=400,
                    detail="必须提供 input_data 或设置 use_sample=true",
                )
            result = prediction_service.predict(
                model_type=req.model_type,
                input_data=req.input_data,
            )
        return {"status": "success", **result}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
def train_model(req: TrainRequest):
    """
    触发模型训练（异步后台执行）。

    - model_type: "classical" 或 "hybrid"
    - epochs: 训练轮数 (默认 50)
    """
    job_id = training_service.start_training(req.model_type, req.epochs)

    # 在后台线程中执行训练
    def _run_background():
        try:
            result = training_service.run_training(job_id)
            # 训练完成后更新 prediction_service 中的模型引用
            if result and "model" in result:
                prediction_service.update_model(req.model_type, result["model"])
        except Exception:
            pass  # 错误已在 TrainingJob 中记录

    thread = threading.Thread(target=_run_background, daemon=True)
    thread.start()

    return {
        "status": "training_started",
        "job_id": job_id,
    }


@app.get("/api/train/{job_id}")
def train_status(job_id: str):
    """查询训练任务状态"""
    job = training_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"训练任务不存在: {job_id}")
    return job.to_dict()


@app.get("/api/metrics")
def metrics_comparison():
    """获取两个模型的对比指标"""
    try:
        return prediction_service.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {e}")


@app.get("/api/sample-data")
def sample_data(n_samples: int = Query(default=5, ge=1, le=50)):
    """获取示例输入数据"""
    try:
        return prediction_service.get_sample_data(n_samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取样本数据失败: {e}")


# ---------------------------------------------------------------------------
# 训练曲线
# ---------------------------------------------------------------------------
CLASSICAL_CURVES_PATH = PROJECT_ROOT / "artifacts" / "train_curves.npz"
HYBRID_CURVES_PATH = PROJECT_ROOT / "artifacts" / "hybrid_train_curves.npz"


@app.get("/api/train-curves")
def train_curves():
    """获取训练/验证 loss 曲线数据"""
    import numpy as np

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


# ---------------------------------------------------------------------------
# 静态文件: 前端页面
# ---------------------------------------------------------------------------
# 优先挂载 dist 目录，回退到 frontend 根目录
_static_dir = FRONTEND_DIST_DIR if FRONTEND_DIST_DIR.exists() else FRONTEND_DIR
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
def serve_index():
    """返回前端首页"""
    for candidate in [
        FRONTEND_DIST_DIR / "index.html",
        FRONTEND_DIR / "index.html",
    ]:
        if candidate.exists():
            return FileResponse(str(candidate))
    return {"message": "短时降雨量预测 API - 前端页面未找到, 请访问 /docs 查看 API 文档"}


# ---------------------------------------------------------------------------
# 直接运行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
