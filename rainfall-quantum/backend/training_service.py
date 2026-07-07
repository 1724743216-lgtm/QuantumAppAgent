"""
短时降雨量预测 - 训练服务

封装经典 LSTM 和混合 VQC+LSTM 的训练逻辑，支持后台异步训练。
"""

import sys
import time
import uuid
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import RainfallDataset
from models.classical_lstm import ClassicalLSTM, train_classical_lstm
from models.hybrid_vqc_lstm import HybridVQCLSTM, get_hybrid_param_groups

# ---------------------------------------------------------------------------
# 路径与模型参数
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "rainfall_data.npz"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLASSICAL_MODEL_PATH = ARTIFACTS_DIR / "classical_lstm.pth"
HYBRID_MODEL_PATH = ARTIFACTS_DIR / "hybrid_vqc_lstm.pth"
CLASSICAL_CURVES_PATH = ARTIFACTS_DIR / "train_curves.npz"
HYBRID_CURVES_PATH = ARTIFACTS_DIR / "hybrid_train_curves.npz"

INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
PRED_LEN = 10
N_QUBITS = 6
VQC_LAYERS = 2
DEVICE = "cpu"


class TrainingJob:
    """单个训练任务的跟踪记录。"""

    def __init__(self, job_id: str, model_type: str, epochs: int):
        self.job_id = job_id
        self.model_type = model_type
        self.epochs = epochs
        self.status: str = "pending"  # pending | running | completed | failed
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error: Optional[str] = None
        self.result: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "model_type": self.model_type,
            "epochs": self.epochs,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
        }


class TrainingService:
    """训练服务：管理训练任务、后台执行、模型持久化。"""

    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}

    def start_training(self, model_type: str, epochs: int = 50) -> str:
        """
        创建并启动一个训练任务。

        Parameters
        ----------
        model_type : str
            "classical" 或 "hybrid"
        epochs : int
            训练轮数

        Returns
        -------
        str : job_id
        """
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(job_id, model_type, epochs)
        self.jobs[job_id] = job
        return job_id

    def run_training(self, job_id: str) -> Dict:
        """
        同步执行训练任务。在请求线程或后台线程中调用。

        Parameters
        ----------
        job_id : str

        Returns
        -------
        dict : 训练结果摘要
        """
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"训练任务不存在: {job_id}")

        job.status = "running"
        job.start_time = time.time()

        try:
            train_ds = RainfallDataset(str(DATA_PATH), split="train")
            val_ds = RainfallDataset(str(DATA_PATH), split="val")

            if job.model_type == "classical":
                result = self._train_classical(train_ds, val_ds, job.epochs)
            else:
                result = self._train_hybrid(train_ds, val_ds, job.epochs)

            job.status = "completed"
            job.end_time = time.time()
            job.result = result
            return result

        except Exception as e:
            job.status = "failed"
            job.end_time = time.time()
            job.error = str(e)
            traceback.print_exc()
            raise

    def _train_classical(self, train_ds, val_ds, epochs: int) -> Dict:
        """训练经典 LSTM 模型。"""
        model = ClassicalLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            pred_len=PRED_LEN,
        )
        result = train_classical_lstm(
            model, train_ds, val_ds,
            n_epochs=epochs, batch_size=64, lr=0.001,
            device=DEVICE, patience=10,
        )
        # 保存模型权重
        torch.save(result["model"].state_dict(), str(CLASSICAL_MODEL_PATH))
        # 保存训练曲线
        np.savez(
            str(CLASSICAL_CURVES_PATH),
            train_losses=result["train_losses"],
            val_losses=result["val_losses"],
        )
        return {
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "model_type": "classical",
            "model": result["model"],
        }

    def _train_hybrid(self, train_ds, val_ds, epochs: int) -> Dict:
        """训练混合 VQC+LSTM 模型。"""
        model = HybridVQCLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            n_qubits=N_QUBITS,
            vqc_layers=VQC_LAYERS,
            pred_len=PRED_LEN,
        )
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
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            # ---- Train ----
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

            # ---- Validate ----
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
                best_epoch = epoch

        # 恢复最佳模型
        if best_state:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)

        # 保存模型权重
        torch.save(model.state_dict(), str(HYBRID_MODEL_PATH))
        # 保存训练曲线
        np.savez(
            str(HYBRID_CURVES_PATH),
            train_losses=train_losses,
            val_losses=val_losses,
        )

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "model_type": "hybrid",
            "model": model,
        }

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)


# 全局单例
training_service = TrainingService()
