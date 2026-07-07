"""
短时降雨量预测 - 预测服务

封装模型加载、推理逻辑、反归一化输出，支持经典 LSTM 和混合 VQC-LSTM 两种模型。
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import RainfallDataset, load_scaler
from models.classical_lstm import ClassicalLSTM, evaluate_model
from models.hybrid_vqc_lstm import HybridVQCLSTM

# ---------------------------------------------------------------------------
# 路径与模型参数
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "rainfall_data.npz"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLASSICAL_MODEL_PATH = ARTIFACTS_DIR / "classical_lstm.pth"
HYBRID_MODEL_PATH = ARTIFACTS_DIR / "hybrid_vqc_lstm.pth"

INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
PRED_LEN = 10
N_QUBITS = 6
VQC_LAYERS = 2
DEVICE = "cpu"

FEATURE_NAMES = [
    "radar_reflectivity",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "wind_direction",
    "past_precipitation",
]


class PredictionService:
    """预测服务：管理模型加载、缓存、推理和反归一化。"""

    def __init__(self):
        self.classical_model: Optional[ClassicalLSTM] = None
        self.hybrid_model: Optional[HybridVQCLSTM] = None
        self.scaler: Optional[Dict] = None
        self.test_dataset: Optional[RainfallDataset] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # 初始化 / 加载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载 scaler、测试数据集和两个模型的权重。"""
        if self._loaded:
            return

        print("[PredictionService] 加载 scaler 与测试数据集...")
        self.scaler = load_scaler(str(DATA_PATH))
        self.test_dataset = RainfallDataset(str(DATA_PATH), split="test")

        # 经典 LSTM
        print("[PredictionService] 加载经典 LSTM 模型...")
        self.classical_model = ClassicalLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            pred_len=PRED_LEN,
        )
        if CLASSICAL_MODEL_PATH.exists():
            state = torch.load(str(CLASSICAL_MODEL_PATH), map_location=DEVICE, weights_only=True)
            self.classical_model.load_state_dict(state)
            print(f"  ✓ 已加载: {CLASSICAL_MODEL_PATH}")
        else:
            print(f"  ⚠ 模型文件不存在: {CLASSICAL_MODEL_PATH}")
        self.classical_model.eval()

        # 混合 VQC+LSTM
        print("[PredictionService] 加载混合 VQC+LSTM 模型...")
        self.hybrid_model = HybridVQCLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            n_qubits=N_QUBITS,
            vqc_layers=VQC_LAYERS,
            pred_len=PRED_LEN,
        )
        if HYBRID_MODEL_PATH.exists():
            state = torch.load(str(HYBRID_MODEL_PATH), map_location=DEVICE, weights_only=True)
            self.hybrid_model.load_state_dict(state)
            print(f"  ✓ 已加载: {HYBRID_MODEL_PATH}")
        else:
            print(f"  ⚠ 模型文件不存在: {HYBRID_MODEL_PATH}")
        self.hybrid_model.eval()

        self._loaded = True
        print("[PredictionService] 加载完成 ✓")

    # ------------------------------------------------------------------
    # 模型获取
    # ------------------------------------------------------------------
    def get_model(self, model_type: str):
        """根据 model_type 返回对应模型实例。"""
        if model_type == "classical":
            return self.classical_model
        elif model_type == "hybrid":
            return self.hybrid_model
        raise ValueError(f"不支持的 model_type: {model_type}")

    def update_model(self, model_type: str, model) -> None:
        """训练完成后更新全局模型引用。"""
        model.eval()
        if model_type == "classical":
            self.classical_model = model
        elif model_type == "hybrid":
            self.hybrid_model = model

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def predict(
        self,
        model_type: str,
        input_data: Optional[List[List[float]]] = None,
        use_sample: bool = False,
    ) -> Dict:
        """
        执行预测并返回反归一化后的结果。

        Parameters
        ----------
        model_type : str
            "classical" 或 "hybrid"
        input_data : list[list[float]] | None
            形状 (seq_len=10, features=7) 的 2D 数组，use_sample=True 时忽略
        use_sample : bool
            是否使用测试集样本数据

        Returns
        -------
        dict : {
            "predictions": [...],
            "model_type": str,
            "input_summary": {...},
            "runtime_ms": float,
        }
        """
        model = self.get_model(model_type)
        if model is None:
            raise RuntimeError(f"模型未加载: {model_type}")

        # 获取输入
        if use_sample:
            # 使用测试集第一个样本
            input_arr = self.test_dataset.inputs[0].numpy()  # (10, 7) 归一化空间
        else:
            if input_data is None:
                raise ValueError("必须提供 input_data 或设置 use_sample=true")
            input_arr = np.array(input_data, dtype=np.float32)
            if input_arr.shape != (10, 7):
                raise ValueError(f"input_data 形状应为 (10, 7), 实际为 {input_arr.shape}")

        # 推理计时
        t_start = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(input_arr).unsqueeze(0).to(DEVICE)  # (1, 10, 7)
            pred_norm = model(x).squeeze(0).cpu().numpy()  # (10,)
        runtime_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # 反归一化预测值
        tgt_min = self.scaler["tgt_min"]
        tgt_range = self.scaler["tgt_range"]
        pred_raw = pred_norm * tgt_range + tgt_min

        # 反归一化输入用于摘要
        feat_min = self.scaler["feat_min"]
        feat_range = self.scaler["feat_range"]
        input_raw = input_arr * feat_range + feat_min

        input_summary = {
            "mean_radar": round(float(np.mean(input_raw[:, 0])), 2),
            "mean_humidity": round(float(np.mean(input_raw[:, 2])), 2),
        }

        return {
            "predictions": [round(float(v), 4) for v in pred_raw],
            "model_type": model_type,
            "input_summary": input_summary,
            "runtime_ms": runtime_ms,
        }

    # ------------------------------------------------------------------
    # 指标评估
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict:
        """获取两个模型在测试集上的对比指标。"""
        results = {}
        for name, model in [("classical", self.classical_model), ("hybrid", self.hybrid_model)]:
            if model is None:
                continue
            eval_result = evaluate_model(model, self.test_dataset, self.scaler, device=DEVICE)
            results[name] = {
                "RMSE": round(eval_result["RMSE"], 4),
                "MAE": round(eval_result["MAE"], 4),
                "MAPE": round(float(eval_result["MAPE"]), 2),
            }

        # 对比指标
        comparison = {}
        if "classical" in results and "hybrid" in results:
            comparison["rmse_diff"] = round(results["hybrid"]["RMSE"] - results["classical"]["RMSE"], 4)

        # 量子开销：用两个模型各推理一次来估算
        if self.classical_model is not None and self.hybrid_model is not None:
            sample_input = self.test_dataset.inputs[:1].to(DEVICE)
            # 经典推理时间
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.classical_model(sample_input)
            classical_ms = (time.perf_counter() - t0) * 1000
            # 混合推理时间
            t1 = time.perf_counter()
            with torch.no_grad():
                _ = self.hybrid_model(sample_input)
            hybrid_ms = (time.perf_counter() - t1) * 1000
            comparison["quantum_overhead_ms"] = round(hybrid_ms - classical_ms, 2)

        return {
            "classical": results.get("classical", {}),
            "hybrid": results.get("hybrid", {}),
            "comparison": comparison,
        }

    # ------------------------------------------------------------------
    # 示例数据
    # ------------------------------------------------------------------
    def get_sample_data(self, n_samples: int = 5) -> Dict:
        """获取测试集样本数据（反归一化）。"""
        n_samples = min(n_samples, len(self.test_dataset))
        inputs = self.test_dataset.inputs[:n_samples].numpy()
        targets = self.test_dataset.targets[:n_samples].numpy()

        tgt_min = self.scaler["tgt_min"]
        tgt_range = self.scaler["tgt_range"]
        targets_raw = targets * tgt_range + tgt_min

        feat_min = self.scaler["feat_min"]
        feat_range = self.scaler["feat_range"]
        inputs_raw = inputs * feat_range + feat_min

        # 两个模型的预测
        predictions = {}
        for name, model in [("classical", self.classical_model), ("hybrid", self.hybrid_model)]:
            if model is None:
                continue
            with torch.no_grad():
                x = self.test_dataset.inputs[:n_samples].to(DEVICE)
                preds = model(x).cpu().numpy()
            preds_raw = preds * tgt_range + tgt_min
            predictions[name] = preds_raw.tolist()

        return {
            "samples": [
                {
                    "index": i,
                    "input_seq": inputs_raw[i].tolist(),
                    "target": targets_raw[i].tolist(),
                    "predictions": {name: preds[i] for name, preds in predictions.items()},
                }
                for i in range(n_samples)
            ],
            "feature_names": FEATURE_NAMES,
            "n_samples": n_samples,
        }


# 全局单例
prediction_service = PredictionService()
