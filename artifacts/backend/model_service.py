"""
模型服务层 - 金融欺诈检测量子应用
封装经典基线(Logistic Regression)和量子方法(VQC)的预测逻辑
演示模式：使用合成数据拟合Scaler/PCA/LR，VQC使用随机初始化参数
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from cqlib import Circuit, Parameter
from cqlib.simulator import StatevectorSimulator

SEED = 42
N_QUBITS = 4
N_LAYERS = 2
N_FEATURES_RAW = 28
N_COMPONENTS = 4


def _build_vqc_circuit(n_qubits: int, n_features: int, layers: int):
    """构建变分量子电路: angle encoding (RY+RZ) + 可训练ansatz"""
    enc_names = []
    var_names = []

    # 编码层: RY + RZ per qubit
    for i in range(min(n_qubits, n_features)):
        enc_names.extend([f"enc_ry_{i}", f"enc_rz_{i}"])

    # 可训练层: RY + RZ + CNOT entanglement
    for layer in range(layers):
        for q in range(n_qubits):
            var_names.extend([f"var_ry_{layer}_{q}", f"var_rz_{layer}_{q}"])

    circuit = Circuit(n_qubits, parameters=enc_names + var_names)

    # 编码门
    for i in range(min(n_qubits, n_features)):
        circuit.ry(i, Parameter(f"enc_ry_{i}"))
        circuit.rz(i, Parameter(f"enc_rz_{i}"))

    # 可训练ansatz
    for layer in range(layers):
        for q in range(n_qubits):
            circuit.ry(q, Parameter(f"var_ry_{layer}_{q}"))
            circuit.rz(q, Parameter(f"var_rz_{layer}_{q}"))
        for q in range(n_qubits - 1):
            circuit.cx(q, q + 1)

    circuit.measure_all()
    return circuit, enc_names, var_names


def _z_expectations(probs: dict, n_qubits: int) -> list:
    """计算每个量子比特的Z期望值"""
    values = []
    for q in range(n_qubits):
        exp_q = sum(
            (1 - 2 * int(bits[-1 - q])) * float(prob)
            for bits, prob in probs.items()
        )
        values.append(exp_q)
    return values


def _features_to_angles(X: np.ndarray, n_enc_params: int) -> np.ndarray:
    """将PCA特征映射到角度编码: tanh(x) * pi，重复填充匹配编码参数数"""
    X_angles = np.tanh(X) * np.pi
    n_features = X_angles.shape[1]
    if n_enc_params > n_features:
        X_angles = np.tile(X_angles, (1, n_enc_params // n_features + 1))[
            :, :n_enc_params
        ]
    return X_angles


class ModelService:
    """金融欺诈检测模型服务，封装经典和量子预测逻辑"""

    def __init__(self):
        np.random.seed(SEED)

        # ---------- 1. 使用合成数据拟合预处理和经典模型（演示模式）----------
        X, y = make_classification(
            n_samples=1000,
            n_features=N_FEATURES_RAW,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=2,
            weights=[0.8, 0.2],
            random_state=SEED,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )

        # StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # PCA
        self.pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
        self.pca.fit(X_train_scaled)

        # Logistic Regression
        self.lr = LogisticRegression(
            random_state=SEED, max_iter=1000, class_weight="balanced"
        )
        X_train_pca = self.pca.transform(X_train_scaled)
        self.lr.fit(X_train_pca, y_train)

        # ---------- 2. 构建VQC电路 ----------
        self.circuit, self.enc_names, self.var_names = _build_vqc_circuit(
            N_QUBITS, N_COMPONENTS, N_LAYERS
        )
        self.n_enc = len(self.enc_names)
        self.n_var = len(self.var_names)

        # VQC可训练参数 - 演示模式：随机初始化
        # 生产环境应从训练结果加载: np.load("vqc_params.npy")
        self.var_params = np.random.randn(self.n_var) * 0.1

        # ---------- 3. 硬编码的评估指标（来自stage1/stage2实验结果）----------
        self.baseline_metrics = {
            "method": "Logistic Regression",
            "f1_score": 0.4211,
            "auc_roc": 0.695,
            "precision": 0.3243,
            "recall": 0.6,
            "accuracy": 0.67,
        }
        self.quantum_metrics = {
            "method": "VQC",
            "f1_score": 0.2476,
            "auc_roc": 0.5723,
            "precision": 0.2,
            "recall": 0.325,
            "accuracy": 0.605,
        }

        self._initialized = True
        print("[ModelService] 初始化完成（演示模式：合成数据拟合，VQC随机参数）")

    def _preprocess(self, features: list) -> np.ndarray:
        """原始28维特征 → 标准化 → PCA降维"""
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        if X.shape[1] != N_FEATURES_RAW:
            raise ValueError(
                f"期望 {N_FEATURES_RAW} 个特征，收到 {X.shape[1]} 个"
            )
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def predict_classical(self, features: list) -> dict:
        """Logistic Regression 预测"""
        X_pca = self._preprocess(features)
        prediction = int(self.lr.predict(X_pca)[0])
        probability = float(self.lr.predict_proba(X_pca)[0, 1])
        return {
            "prediction": prediction,
            "fraud_probability": round(probability, 4),
            "method": "logistic_regression",
        }

    def predict_quantum(self, features: list) -> dict:
        """VQC 预测（使用cqlib StatevectorSimulator）"""
        X_pca = self._preprocess(features)
        angles = _features_to_angles(X_pca, self.n_enc)[0]

        # 绑定参数
        params = {}
        for i, name in enumerate(self.enc_names):
            params[name] = float(angles[i])
        for i, name in enumerate(self.var_names):
            params[name] = float(self.var_params[i])

        bound = self.circuit.assign_parameters(params)
        probs = StatevectorSimulator(circuit=bound).measure()
        z_exps = _z_expectations(probs, N_QUBITS)

        # qubit 0 的 Z期望 → sigmoid → 欺诈概率
        fraud_prob = 1.0 / (1.0 + np.exp(-z_exps[0]))
        prediction = 1 if fraud_prob >= 0.5 else 0

        return {
            "prediction": prediction,
            "fraud_probability": round(float(fraud_prob), 4),
            "method": "vqc",
        }

    def predict_batch_quantum(self, features_list: list) -> list:
        """VQC 批量预测"""
        results = []
        for features in features_list:
            results.append(self.predict_quantum(features))
        return results

    def get_comparison(self) -> dict:
        """返回经典vs量子的指标对比"""
        return {
            "baseline": self.baseline_metrics,
            "quantum": self.quantum_metrics,
        }
