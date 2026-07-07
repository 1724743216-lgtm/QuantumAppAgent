"""
混合 VQC+LSTM 降雨量预测模型

架构:
  Input(batch, 10, 7) → LSTM(2层, hidden=64) → 取最后hidden state(batch, 64)
    → 投影层 Linear(64, n_qubits) → VQC量子层(batch, n_qubits)
    → 拼接[LSTM hidden, VQC输出] → 预测头 Linear(64+n_qubits, 10)
    → Output(batch, 10) 未来1h逐6min降雨量

设计遵循 cqlib-hybrid skill 的 "Sequence" 模式:
  - LSTM 编码时序信息
  - 量子层提供非线性特征增强
  - 拼接经典+量子特征保证信息通路
"""

import torch
import torch.nn as nn

from models.quantum_layer import CqlibVQCLayer


class HybridVQCLSTM(nn.Module):
    """
    量子+经典混合降雨量预测模型。

    Parameters
    ----------
    input_size : int
        输入特征维度 (7)
    hidden_size : int
        LSTM 隐层大小 (64)
    num_layers : int
        LSTM 层数 (2)
    n_qubits : int
        量子比特数 (6)
    vqc_layers : int
        VQC 变分层深度 (2)
    pred_len : int
        预测长度 (10)
    dropout : float
        LSTM 层间 dropout
    """

    def __init__(
        self,
        input_size=7,
        hidden_size=64,
        num_layers=2,
        n_qubits=6,
        vqc_layers=2,
        pred_len=10,
        dropout=0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_qubits = n_qubits
        self.vqc_layers = vqc_layers
        self.pred_len = pred_len

        # 1. LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. 投影层: LSTM hidden → 量子层输入维度
        self.projection = nn.Linear(hidden_size, n_qubits)

        # 3. VQC 量子特征提取层
        self.vqc = CqlibVQCLayer(
            n_qubits=n_qubits,
            n_features=n_qubits,  # 投影后维度 = n_qubits
            layers=vqc_layers,
        )

        # 4. 预测头: 拼接 [LSTM hidden, VQC输出] → 预测
        self.head = nn.Linear(hidden_size + n_qubits, pred_len)

    @property
    def quantum_params(self):
        """量子变分参数数量"""
        return self.vqc.n_var_params + self.vqc.input_scale.numel()

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
        # 1. LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # 2. 投影到量子层维度
        projected = self.projection(last_hidden)  # (batch, n_qubits)

        # 3. VQC 量子特征提取
        quantum_features = self.vqc(projected)  # (batch, n_qubits)

        # 4. 拼接 [LSTM hidden, VQC输出]
        fused = torch.cat([last_hidden, quantum_features], dim=-1)  # (batch, hidden_size + n_qubits)

        # 5. 预测头
        pred = self.head(fused)  # (batch, pred_len)

        return pred


def get_hybrid_param_groups(model, classical_lr=0.001, quantum_lr=0.0005):
    """
    为混合模型创建差异化学习率的参数组。

    Parameters
    ----------
    model : HybridVQCLSTM
    classical_lr : float
        经典参数学习率
    quantum_lr : float
        量子变分参数学习率

    Returns
    -------
    list : 参数组列表，适用于 torch.optim
    """
    quantum_param_names = set()
    for name, _ in model.vqc.named_parameters():
        quantum_param_names.add(f"vqc.{name}")

    classical_params = []
    quantum_params = []

    for name, param in model.named_parameters():
        if name in quantum_param_names:
            quantum_params.append(param)
        else:
            classical_params.append(param)

    return [
        {"params": classical_params, "lr": classical_lr},
        {"params": quantum_params, "lr": quantum_lr},
    ]
