# 金融欺诈检测量子应用

基于变分量子电路（VQC）的信用卡交易欺诈检测二分类应用，与经典 Logistic Regression 基线对比。本应用为**技术演示 (PoC)**，展示量子机器学习在金融风控领域的可行性流程。

> ⚠️ **重要声明**：当前配置下，VQC 模型性能**低于**经典基线。所有量子计算结果基于本地模拟器（StatevectorSimulator），非真实量子硬件。详见[局限性声明](#局限性声明)。

---

## 快速开始

### 环境要求

- Python 3.10+
- pip 或 conda

### 安装依赖

```bash
cd /artifacts
pip install -r backend/requirements.txt
```

依赖包含：FastAPI, uvicorn, numpy, scikit-learn, cqlib, torch, pydantic

### 启动后端

```bash
cd /artifacts
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

启动成功后控制台输出：
```
[ModelService] 初始化完成（演示模式：合成数据拟合，VQC随机参数）
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 访问前端

- **Swagger UI**: http://127.0.0.1:8000/docs
- **健康检查**: http://127.0.0.1:8000/health
- **qccp 页面**: 需集成到 qccp-web 平台，详见 [INTEGRATE.md](./INTEGRATE.md)

### 验证服务

```bash
# 健康检查
curl http://127.0.0.1:8000/health
# 预期: {"status":"ok"}

# 应用信息
curl http://127.0.0.1:8000/api/info

# 单条预测（28维特征）
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.2, 0.5, 1.3, -0.7, 0.2, 0.8, -1.1, 0.3, -0.5, 0.9, 1.2, -0.3, 0.7, 0.1, -0.8, 0.4, 0.6, -1.0, 0.2, 0.5, -0.9, 0.3, 1.1, -0.6, 0.8, -0.4, 0.2]}'

# 经典vs量子对比
curl http://127.0.0.1:8000/api/comparison
```

---

## 项目结构

```
/
├── experiments/                          # 实验产物
│   ├── stage1_scope_baseline/            # Stage 1: 经典基线
│   │   ├── stage1_baseline.py            # 基线训练脚本
│   │   ├── baseline_report.json          # 基线评估报告
│   │   ├── preprocessed_data.npz         # 预处理后的数据
│   │   ├── requirements.json             # 任务需求定义
│   │   └── solution_plan.md              # 四阶段方案计划
│   ├── stage2_quantum_method/            # Stage 2: 量子方法
│   │   ├── stage2_quantum.py             # VQC 训练脚本
│   │   └── quantum_report.json           # 量子评估报告
│   └── stage4_verification_handoff/      # Stage 4: 验证与交付
│       └── verification_report.md        # 验证报告
├── artifacts/                            # 应用产物
│   ├── backend/                          # FastAPI 后端
│   │   ├── main.py                       # 主应用（端点定义）
│   │   ├── model_service.py              # 模型服务（预处理+预测）
│   │   ├── requirements.txt              # Python 依赖
│   │   └── README.md                     # 后端 API 文档
│   ├── qccp-page-output/                 # qccp 前端页面
│   │   └── quantumFraudDetection/
│   │       ├── INTEGRATE.md              # 前端集成指南
│   │       └── project-files/
│   │           └── src/views/solution/quantumFraudDetection/
│   │               ├── index.vue         # Vue SFC 页面
│   │               └── data.js           # 页面数据
│   ├── README.md                         # 本文件
│   └── INTEGRATE.md                      # 集成指南
```

---

## 算法说明

### 整体流程

```
原始交易数据 (28维)
    │
    ▼
StandardScaler 标准化
    │
    ▼
PCA 降维 (28维 → 4维)
    │
    ├──▶ [经典路径] Logistic Regression → 二分类
    │
    └──▶ [量子路径]
            │
            ▼
        角度编码: tanh(x) × π → RY+RZ 门
            │
            ▼
        变分量子电路: 4 qubits, 2 layers
        (RY+RZ 可训练旋转 + CNOT 纠缠)
            │
            ▼
        Z 期望测量 (qubit 0)
            │
            ▼
        Sigmoid → 欺诈概率 → 二分类
```

### 经典基线

- **算法**: Logistic Regression（`class_weight=balanced`，应对类别不平衡）
- **输入**: PCA 降维后 4 维特征
- **训练**: 800 训练样本，闭式解/迭代优化

### 量子方法 (VQC)

- **编码**: 角度编码（Angle Encoding），每量子比特 RY+RZ 双旋转，将 4 维 PCA 特征映射到 4 个量子比特
- **Ansatz**: 2 层参数化线路，每层包含 RY+RZ 旋转（可训练）+ 线性 CNOT 纠缠
- **输出**: 测量 qubit 0 的 Z 期望值 → sigmoid → [0, 1] 概率
- **优化**: Adam (lr=0.01)，有限差分梯度 (eps=0.01)，BCE 损失
- **参数量**: 16 个可训练参数（2 layers × 4 qubits × 2 params）
- **后端**: StatevectorSimulator（本地模拟器，无噪声、无采样误差）

---

## 实验结果对比

### 核心指标

| 指标 | 经典方法 (Logistic Regression) | 量子方法 (VQC) | 差异 |
|------|-------------------------------|----------------|------|
| **F1 Score** | **0.4211** | 0.2476 | -41.2% |
| AUC-ROC | **0.6950** | 0.5723 | -17.6% |
| Precision | **0.3243** | 0.2000 | -38.3% |
| Recall | **0.6000** | 0.3250 | -45.8% |
| Accuracy | **0.6700** | 0.6050 | -9.7% |

> **加粗**表示较优值。在所有指标上，经典 Logistic Regression 均优于 VQC。

### 实验配置

| 项目 | 值 |
|------|-----|
| 数据集 | 合成信用卡欺诈检测 (sklearn.make_classification) |
| 样本数 | 1000 (欺诈 200, 正常 800) |
| 原始特征 | 28 维 |
| PCA 降维 | 4 维 |
| 训练/测试划分 | 80% / 20% (stratified, seed=42) |
| 量子比特数 | 4 |
| Ansatz 层数 | 2 |
| 训练轮次 | 30 |
| 优化器 | Adam, lr=0.01 |
| 后端 | StatevectorSimulator (local) |

---

## API 文档

详细 API 文档请参阅 [backend/README.md](./backend/README.md) 或启动服务后访问 Swagger UI (`/docs`)。

### 端点概览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/info` | 应用信息（算法、指标等） |
| POST | `/api/predict` | 单条交易预测（28 维特征） |
| POST | `/api/predict_batch` | 批量交易预测 |
| GET | `/api/comparison` | 经典 vs 量子指标对比 |

---

## 局限性声明

### 核心局限

1. **VQC 性能低于经典基线**：在当前 4 量子比特、2 层 ansatz 配置下，VQC 的所有指标均低于 Logistic Regression 基线。本应用**不具备量子优势**，仅作为量子机器学习流程的技术演示。

2. **模拟器结果**：所有量子计算在 `StatevectorSimulator` 上执行。该模拟器无噪声、无去相干、无门误差、无读取噪声。真实量子硬件上预期性能将更低。

3. **合成数据集**：使用 `sklearn.make_classification` 生成，非真实信用卡交易数据。实验结论不可直接推广到生产环境。

4. **后端演示模式**：在线预测 API 使用**随机初始化**的 VQC 参数，而非训练结果。预测结果仅用于展示端到端流程可行性，不具备实际判别能力。

### 方法局限

5. **PCA 信息损失**：28 维降至 4 维必然丢失判别信息，同时影响经典和量子方法。
6. **单量子比特输出**：仅使用 qubit 0 的 Z 期望，浪费了 75% 的测量信息。
7. **有限差分梯度**：非解析参数移位，梯度精度有限。
8. **单次划分评估**：无交叉验证，指标方差未知。
9. **单一基线**：未与 Random Forest、XGBoost 等强基线对比。

### 改进方向

- 增加量子比特数至 8-16，减少 PCA 信息损失
- 采用混合量子-经典架构（HQNN）
- 使用多量子比特期望值或量子概率层
- 引入噪声模型模拟硬件行为
- 获取真实信用卡数据集
- 系统化超参数搜索

---

## 许可证

本项目仅供研究和演示目的使用。
