"""
Stage 2: QML量子方法 - VQC金融欺诈检测
使用Cqlib SDK构建变分量子电路进行二分类
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from cqlib import Circuit, Parameter
from cqlib.simulator import StatevectorSimulator

SEED = 42
N_QUBITS = 4
N_LAYERS = 2
LEARNING_RATE = 0.01
EPOCHS = 30
BATCH_SIZE = 32

np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. 加载预处理数据
print("=== 加载预处理数据 ===")
import os
stage1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1_scope_baseline")
data = np.load(os.path.join(stage1_dir, "preprocessed_data.npz"))
X_train_pca = data['X_train_pca']
X_test_pca = data['X_test_pca']
y_train = data['y_train']
y_test = data['y_test']
print(f"训练集: {X_train_pca.shape[0]}, 测试集: {X_test_pca.shape[0]}, 特征: {X_train_pca.shape[1]}")


# 2. 构建VQC电路
def build_vqc(n_qubits, n_features, layers):
    """构建变分量子电路: angle encoding + 可训练ansatz"""
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


def z_expectations(probs, n_qubits):
    """计算每个量子比特的Z期望值"""
    values = []
    for q in range(n_qubits):
        exp_q = sum((1 - 2 * int(bits[-1 - q])) * float(prob) for bits, prob in probs.items())
        values.append(exp_q)
    return values


def run_vqc(circuit, enc_names, var_names, x_angles, var_values):
    """运行VQC并返回Z期望值"""
    params = {}
    for i, name in enumerate(enc_names):
        params[name] = float(x_angles[i])
    for i, name in enumerate(var_names):
        params[name] = float(var_values[i])
    bound = circuit.assign_parameters(params)
    probs = StatevectorSimulator(circuit=bound).measure()
    return z_expectations(probs, circuit.num_qubits)


# 3. 构建电路
print("\n=== 构建VQC电路 ===")
circuit, enc_names, var_names = build_vqc(N_QUBITS, X_train_pca.shape[1], N_LAYERS)
n_enc = len(enc_names)
n_var = len(var_names)
print(f"量子比特数: {N_QUBITS}")
print(f"编码参数: {n_enc}, 可训练参数: {n_var}")
print(f"线路深度: layers={N_LAYERS}")

# 4. 准备角度编码特征
# 将PCA特征映射到角度: tanh(x) * pi，然后重复填充以匹配编码参数数
def features_to_angles(X, n_enc_params):
    """将特征转换为角度编码"""
    X_angles = np.tanh(X) * np.pi
    # 重复特征以匹配编码参数数量 (每个qubit需要2个角度: RY, RZ)
    n_features = X_angles.shape[1]
    if n_enc_params > n_features:
        # 重复特征
        repeat = n_enc_params // n_features
        remainder = n_enc_params % n_features
        X_angles = np.tile(X_angles, (1, repeat + 1))[:, :n_enc_params]
    return X_angles


X_train_angles = features_to_angles(X_train_pca, n_enc)
X_test_angles = features_to_angles(X_test_pca, n_enc)

# 5. 训练VQC (参数移位法 + Adam优化器)
print("\n=== 训练VQC ===")
var_params = torch.randn(n_var, requires_grad=True)
optimizer = optim.Adam([var_params], lr=LEARNING_RATE)

# 使用第一个Z期望值的sigmoid作为分类概率
train_losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n_correct = 0
    n_total = 0

    # 随机排列训练数据
    indices = np.random.permutation(len(X_train_angles))

    for batch_start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[batch_start:batch_start + BATCH_SIZE]
        batch_angles = X_train_angles[batch_idx]
        batch_labels = torch.tensor(y_train[batch_idx], dtype=torch.float32)

        batch_preds = []
        for i in range(len(batch_idx)):
            exp_vals = run_vqc(circuit, enc_names, var_names, batch_angles[i], var_params.detach().numpy())
            pred = torch.tensor(exp_vals[0], dtype=torch.float32)  # 使用qubit 0的Z期望
            batch_preds.append(pred)

        batch_preds = torch.stack(batch_preds)
        # 使用sigmoid映射到[0,1]
        probs = torch.sigmoid(batch_preds)
        # 二分类交叉熵损失
        loss = nn.BCELoss()(probs, batch_labels)

        # 数值梯度（参数移分法）
        optimizer.zero_grad()

        # 有限差分梯度
        eps = 0.01
        grad = torch.zeros_like(var_params)
        for j in range(n_var):
            var_plus = var_params.clone()
            var_plus[j] += eps
            var_minus = var_params.clone()
            var_minus[j] -= eps

            preds_plus = []
            preds_minus = []
            for i in range(len(batch_idx)):
                exp_plus = run_vqc(circuit, enc_names, var_names, batch_angles[i], var_plus.detach().numpy())
                exp_minus = run_vqc(circuit, enc_names, var_names, batch_angles[i], var_minus.detach().numpy())
                preds_plus.append(torch.sigmoid(torch.tensor(exp_plus[0], dtype=torch.float32)))
                preds_minus.append(torch.sigmoid(torch.tensor(exp_minus[0], dtype=torch.float32)))

            preds_plus = torch.stack(preds_plus)
            preds_minus = torch.stack(preds_minus)
            loss_plus = nn.BCELoss()(preds_plus, batch_labels)
            loss_minus = nn.BCELoss()(preds_minus, batch_labels)
            grad[j] = (loss_plus - loss_minus) / (2 * eps)

        var_params.grad = grad
        optimizer.step()

        epoch_loss += loss.item() * len(batch_idx)
        predicted = (probs >= 0.5).float()
        n_correct += (predicted == batch_labels).sum().item()
        n_total += len(batch_idx)

    avg_loss = epoch_loss / len(X_train_angles)
    train_losses.append(avg_loss)
    acc = n_correct / n_total

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Train Acc={acc:.4f}")

# 6. 评估
print("\n=== VQC测试集评估 ===")
y_pred_list = []
y_prob_list = []

for i in range(len(X_test_angles)):
    exp_vals = run_vqc(circuit, enc_names, var_names, X_test_angles[i], var_params.detach().numpy())
    prob = 1.0 / (1.0 + np.exp(-exp_vals[0]))  # sigmoid
    y_prob_list.append(prob)
    y_pred_list.append(1 if prob >= 0.5 else 0)

y_pred = np.array(y_pred_list)
y_prob = np.array(y_prob_list)

f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

# 7. 生成量子报告
quantum_report = {
    "task": "金融欺诈检测二分类",
    "data": "合成信用卡欺诈检测数据集 (1000 samples, 28 features, PCA→4)",
    "primary_metric": "f1_score",
    "higher_is_better": True,
    "value": round(f1, 4),
    "secondary_metrics": {
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4)
    },
    "method": "VQC (Variational Quantum Classifier)",
    "encoding": "angle_encoding (RY+RZ per qubit)",
    "n_qubits": N_QUBITS,
    "ansatz_layers": N_LAYERS,
    "n_parameters": n_var,
    "output": "z_expectation (qubit 0) → sigmoid",
    "optimizer": f"Adam (lr={LEARNING_RATE})",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "preprocessing": "StandardScaler + PCA(n_components=4) + tanh*π angle encoding",
    "command": "python3 stage2_quantum.py",
    "artifact_paths": [
        "/experiments/stage2_quantum_method/quantum_report.json"
    ],
    "seed": SEED,
    "backend": "StatevectorSimulator (local)",
    "shots": "statevector (no sampling)",
    "circuit_depth": f"{N_LAYERS} layers",
    "limitations": [
        "合成数据集，非真实交易数据",
        "PCA降维可能丢失部分信息",
        "有限差分梯度近似，非精确参数移位",
        "仅使用1个量子比特的Z期望作为输出",
        "模拟器结果，非真实硬件"
    ]
}

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "quantum_report.json"), "w") as f:
    json.dump(quantum_report, f, indent=2, ensure_ascii=False)

print("\n量子报告已保存至 quantum_report.json")
