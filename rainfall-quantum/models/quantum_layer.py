"""
VQC 量子特征提取层

基于 cqlib SDK 实现的 CqlibVQCLayer，遵循 cqlib-hybrid skill 的骨架模式。

架构: 角度编码(Ry+Rz per qubit) → 可训练变分层(Ry+Rz+CX纠缠) → Z期望值测量

注意: 量子电路通过 StatevectorSimulator 逐样本执行，
      变分参数通过 .detach().cpu() 传入电路，梯度不穿透量子模拟器。
      训练时依赖经典部分的梯度回传以及量子输出对损失的贡献。
"""

import numpy as np
import torch
import torch.nn as nn
from cqlib import Circuit, Parameter
from cqlib.simulator import StatevectorSimulator


# 保持向后兼容的别名
CqlibExpectationLayer = None  # 延迟设置


class CqlibVQCLayer(nn.Module):
    """
    VQC 量子特征提取层:
    - 输入: (batch, n_features)
    - 输出: (batch, n_qubits) — Z 期望值
    - 编码: tanh 缩放后的 Ry + Rz 门 (per qubit)
    - 变分: layers × (Ry + Rz per qubit) + 线性 CX 纠缠
    - 测量: 各 qubit 的 <Z> 期望值

    Parameters
    ----------
    n_qubits : int
        量子比特数
    n_features : int
        输入特征维度 (编码参数数量 = min(n_qubits, n_features))
    layers : int
        变分层深度
    """

    def __init__(self, n_qubits, n_features, layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_enc = min(n_qubits, n_features)
        self._layers = layers

        # 构建参数化量子电路
        self.circuit, self.enc_names, self.var_names = self._build_circuit(layers)

        # 可训练变分参数 (PyTorch Parameter)
        self.var_values = nn.Parameter(torch.randn(len(self.var_names)) * 0.05)

        # 输入缩放参数 (可训练)
        self.input_scale = nn.Parameter(torch.ones(self.n_enc))

    def _build_circuit(self, layers):
        """
        构建参数化量子电路。

        编码层: 前 n_enc 个 qubit 各施加 Ry + Rz
        编码参数: enc_ry_0, enc_rz_0, enc_ry_1, enc_rz_1, ...
        变分参数: var_ry_{layer}_{q}, var_rz_{layer}_{q}
        纠缠: 线性 CX 链 (q0→q1→...→q_{n-1})
        """
        # 编码参数名: 每个 qubit 有 ry 和 rz 两个编码参数
        enc_names = []
        for i in range(self.n_enc):
            enc_names.append(f"enc_ry_{i}")
            enc_names.append(f"enc_rz_{i}")

        var_names = [
            f"var_ry_{layer}_{q}"
            for layer in range(layers)
            for q in range(self.n_qubits)
        ] + [
            f"var_rz_{layer}_{q}"
            for layer in range(layers)
            for q in range(self.n_qubits)
        ]

        circuit = Circuit(self.n_qubits, parameters=enc_names + var_names)

        # 编码层: 对前 n_enc 个 qubit 施加 Ry + Rz 门
        for i in range(self.n_enc):
            circuit.ry(i, Parameter(f"enc_ry_{i}"))
            circuit.rz(i, Parameter(f"enc_rz_{i}"))

        # 变分层
        for layer in range(layers):
            # 每个 qubit: Ry + Rz
            for q in range(self.n_qubits):
                ry_name = f"var_ry_{layer}_{q}"
                rz_name = f"var_rz_{layer}_{q}"
                circuit.ry(q, Parameter(ry_name))
                circuit.rz(q, Parameter(rz_name))

            # 线性纠缠: CX 链
            for q in range(self.n_qubits - 1):
                circuit.cx(q, q + 1)

        circuit.measure_all()
        return circuit, enc_names, var_names

    @property
    def n_var_params(self):
        """变分参数数量"""
        return len(self.var_names)

    @property
    def circuit_depth(self):
        """电路深度 (理论近似值)"""
        # 编码层(Ry+Rz并行=1层) + 每变分层(Ry+Rz并行=1 + CX链=1) + 测量(1)
        enc_depth = 1
        var_depth_per_layer = 2  # Ry/Rz并行算1层, CX链算1层
        total = enc_depth + self._layers * var_depth_per_layer + 1
        return total

    def _run_one(self, features):
        """
        单样本执行量子电路，返回各 qubit 的 Z 期望值。

        Parameters
        ----------
        features : torch.Tensor
            形状 (n_features,)

        Returns
        -------
        torch.Tensor
            形状 (n_qubits,)，各 qubit 的 <Z> 期望值
        """
        # 编码: tanh 缩放后乘以 π，映射到 [-π, π]
        scaled = torch.tanh(features[:self.n_enc] * self.input_scale) * np.pi

        # 构建参数字典: 每个 qubit 有 ry 和 rz 两个编码参数
        params = {}
        for i in range(self.n_enc):
            params[f"enc_ry_{i}"] = float(scaled[i].detach().cpu())
            params[f"enc_rz_{i}"] = float(scaled[i].detach().cpu())

        params.update({
            name: float(self.var_values[i].detach().cpu())
            for i, name in enumerate(self.var_names)
        })

        # 绑定参数并执行模拟
        bound = self.circuit.assign_parameters(params)
        probs = StatevectorSimulator(circuit=bound).measure()

        # 计算 Z 期望值: <Z_q> = Σ_prob (1 - 2*bit_q) * prob
        # bitstring 约定: bits[-1 - q] 对应 qubit q
        z_expectations = torch.tensor([
            sum(
                (1 - 2 * int(bits[-1 - q])) * float(prob)
                for bits, prob in probs.items()
            )
            for q in range(self.n_qubits)
        ], dtype=features.dtype, device=features.device)

        return z_expectations

    def forward(self, x):
        """
        批量前向传播。

        Parameters
        ----------
        x : torch.Tensor
            形状 (batch, n_features)

        Returns
        -------
        torch.Tensor
            形状 (batch, n_qubits)，各样本的 Z 期望值
        """
        return torch.stack([self._run_one(sample) for sample in x], dim=0)


# 向后兼容别名
CqlibExpectationLayer = CqlibVQCLayer
