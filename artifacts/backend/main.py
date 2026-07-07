"""
FastAPI主应用 - 金融欺诈检测量子应用
提供健康检查、应用信息、单条/批量预测、经典vs量子对比端点
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.model_service import ModelService

# 全局模型服务实例
_model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化模型服务"""
    global _model_service
    _model_service = ModelService()
    yield
    _model_service = None


app = FastAPI(
    title="金融欺诈检测量子应用",
    description="使用VQC（变分量子电路）进行信用卡欺诈检测，与经典Logistic Regression基线对比",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- 静态前端托管 ----------
_dist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dist")
if os.path.isdir(_dist_path):
    app.mount("/static", StaticFiles(directory=_dist_path, html=True), name="static")


# ---------- 请求/响应模型 ----------
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=28, max_length=28, description="28个原始交易特征"
    )


class PredictResponse(BaseModel):
    status: str
    prediction: int = Field(description="0=正常, 1=欺诈")
    fraud_probability: float = Field(description="欺诈概率 [0, 1]")
    method: str


class PredictBatchRequest(BaseModel):
    features_list: list[list[float]] = Field(
        ..., description="多个交易的特征列表，每个28个特征"
    )


class PredictBatchResponse(BaseModel):
    status: str
    predictions: list[PredictResponse]


class ComparisonResponse(BaseModel):
    baseline: dict
    quantum: dict


# ---------- 端点 ----------


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}


@app.get("/api/info")
async def app_info():
    """应用信息端点"""
    return {
        "name": "金融欺诈检测量子应用",
        "description": "使用VQC进行信用卡欺诈检测，与经典Logistic Regression基线对比",
        "algorithm": "VQC (Variational Quantum Classifier)",
        "baseline_algorithm": "Logistic Regression (class_weight=balanced)",
        "n_qubits": 4,
        "n_layers": 2,
        "encoding": "angle_encoding (RY+RZ per qubit)",
        "ansatz": "RY+RZ+CNOT entanglement",
        "output": "z_expectation (qubit 0) → sigmoid",
        "preprocessing": "StandardScaler + PCA(n_components=4)",
        "metrics": {
            "primary": "f1_score",
            "baseline_f1": 0.4211,
            "quantum_f1": 0.2476,
        },
        "backend": "StatevectorSimulator (local)",
        "mode": "demo",
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """单条交易预测：返回欺诈概率和预测结果"""
    if _model_service is None:
        raise HTTPException(status_code=503, detail="模型服务未初始化")

    try:
        result = _model_service.predict_quantum(request.features)
        return PredictResponse(
            status="success",
            prediction=result["prediction"],
            fraud_probability=result["fraud_probability"],
            method=result["method"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/api/predict_batch", response_model=PredictBatchResponse)
async def predict_batch(request: PredictBatchRequest):
    """批量交易预测"""
    if _model_service is None:
        raise HTTPException(status_code=503, detail="模型服务未初始化")

    # 校验每条特征数量
    for i, features in enumerate(request.features_list):
        if len(features) != 28:
            raise HTTPException(
                status_code=400,
                detail=f"第{i + 1}条交易特征数不正确：期望28，收到{len(features)}",
            )

    try:
        results = _model_service.predict_batch_quantum(request.features_list)
        predictions = [
            PredictResponse(
                status="success",
                prediction=r["prediction"],
                fraud_probability=r["fraud_probability"],
                method=r["method"],
            )
            for r in results
        ]
        return PredictBatchResponse(status="success", predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


@app.get("/api/comparison", response_model=ComparisonResponse)
async def comparison():
    """经典vs量子的指标对比"""
    if _model_service is None:
        raise HTTPException(status_code=503, detail="模型服务未初始化")

    result = _model_service.get_comparison()
    return ComparisonResponse(baseline=result["baseline"], quantum=result["quantum"])
