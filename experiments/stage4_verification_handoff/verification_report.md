# 验证报告 — 金融欺诈检测量子应用

**验证日期**: 2026-06-30  
**验证阶段**: Stage 4 — 验证与交付  
**验证人**: 自动验证流程

---

## 1. 对比验证：指标值与报告 JSON 一致性

### 1.1 经典基线指标一致性

| 指标 | 用户提供值 | baseline_report.json | model_service.py 硬编码 | /api/comparison 端点 | 一致？ |
|------|-----------|---------------------|------------------------|---------------------|--------|
| F1 Score | 0.4211 | 0.4211 | 0.4211 | 0.4211 | ✅ |
| AUC-ROC | 0.6950 | 0.695 | 0.695 | 0.695 | ✅ |
| Precision | 0.3243 | 0.3243 | 0.3243 | 0.3243 | ✅ |
| Recall | 0.6000 | 0.6 | 0.6 | 0.6 | ✅ |
| Accuracy | 0.6700 | 0.67 | 0.67 | 0.67 | ✅ |

> 注：JSON/代码中使用小数简写（如 0.6 vs 0.6000），数值完全一致。

### 1.2 量子方法（VQC）指标一致性

| 指标 | 用户提供值 | quantum_report.json | model_service.py 硬编码 | /api/comparison 端点 | 一致？ |
|------|-----------|--------------------|-----------------------|---------------------|--------|
| F1 Score | 0.2476 | 0.2476 | 0.2476 | 0.2476 | ✅ |
| AUC-ROC | 0.5723 | 0.5723 | 0.5723 | 0.5723 | ✅ |
| Precision | 0.2000 | 0.2 | 0.2 | 0.2 | ✅ |
| Recall | 0.3250 | 0.325 | 0.325 | 0.325 | ✅ |
| Accuracy | 0.6050 | 0.605 | 0.605 | 0.605 | ✅ |

**结论**: 所有指标在用户提供值、报告JSON、后端代码、API端点四处完全一致。

---

## 2. 数据/预处理一致性检查

### 2.1 数据集一致性

| 检查项 | Stage 1 (baseline) | Stage 2 (quantum) | 后端 model_service.py | 一致？ |
|--------|--------------------|--------------------|-----------------------|--------|
| 数据源 | `make_classification` | 加载 stage1 `preprocessed_data.npz` | `make_classification` (同参数) | ✅ |
| 样本数 | 1000 | 1000 | 1000 | ✅ |
| 原始特征数 | 28 | 28 | 28 | ✅ |
| 欺诈比例 | 0.2 | 0.2 | 0.2 | ✅ |
| 随机种子 | 42 | 42 | 42 | ✅ |
| 训练/测试划分 | 80/20, stratify | 加载 stage1 划分 | 80/20, stratify | ✅ |

### 2.2 预处理流水线一致性

| 检查项 | Stage 1 | Stage 2 | 后端 | 一致？ |
|--------|---------|---------|------|--------|
| StandardScaler | ✅ fit_transform | 使用 stage1 预处理结果 | ✅ fit_transform (同种子数据) | ✅ |
| PCA n_components | 4 | 4 | 4 | ✅ |
| PCA random_state | 42 | N/A (使用 stage1 结果) | 42 | ✅ |
| 角度编码 | N/A | `tanh(x) * π` + 重复填充 | `tanh(x) * π` + 重复填充 | ✅ |

**注意事项**: 
- 后端 `model_service.py` 在启动时使用 `make_classification(SEED=42)` 重新生成数据并拟合 Scaler/PCA/LR，确保与实验阶段参数一致。但 VQC 参数为**随机初始化**（`np.random.randn * 0.1`），非训练结果加载，因此后端在线预测结果与实验阶段 VQC 预测结果不同。
- Stage 2 使用 Stage 1 保存的 `preprocessed_data.npz`，确保训练/测试划分完全一致。

---

## 3. 量子电路参数确认

| 参数 | quantum_report.json | stage2_quantum.py | model_service.py | /api/info 端点 | 一致？ |
|------|--------------------|-------------------|-----------------|---------------|--------|
| 量子比特数 | 4 | N_QUBITS=4 | N_QUBITS=4 | n_qubits: 4 | ✅ |
| Ansatz 层数 | 2 | N_LAYERS=2 | N_LAYERS=2 | n_layers: 2 | ✅ |
| 编码方式 | angle_encoding (RY+RZ) | RY+RZ per qubit | RY+RZ per qubit | angle_encoding (RY+RZ) | ✅ |
| 纠缠方式 | CNOT entanglement | cx(q, q+1) | cx(q, q+1) | CNOT entanglement | ✅ |
| 输出方式 | z_expectation (qubit 0) → sigmoid | exp_vals[0] → sigmoid | z_exps[0] → sigmoid | z_expectation (qubit 0) → sigmoid | ✅ |
| 可训练参数数 | 16 | n_var=16 (2 layers × 4 qubits × 2 params) | 16 | N/A | ✅ |
| 优化器 | Adam (lr=0.01) | Adam lr=0.01 | N/A (随机参数) | Adam (lr=0.01) | ✅ |
| 训练轮次 | 30 | EPOCHS=30 | N/A | N/A | ✅ |
| 批次大小 | 32 | BATCH_SIZE=32 | N/A | N/A | ✅ |
| 后端 | StatevectorSimulator (local) | StatevectorSimulator | StatevectorSimulator | StatevectorSimulator (local) | ✅ |
| Shots | statevector (no sampling) | statevector | statevector | N/A | ✅ |

**结论**: 量子电路参数在报告、实验代码、后端服务、API信息端点四处完全一致。

---

## 4. 前后端合同一致性

### 4.1 API 端点合同

| 端点 | requirements.json 约定 | main.py 实现 | 一致？ |
|------|----------------------|-------------|--------|
| `GET /health` | ✅ | ✅ 返回 `{"status": "ok"}` | ✅ |
| `GET /api/info` | ✅ | ✅ 返回算法/指标信息 | ✅ |
| `POST /api/predict` | ✅ | ✅ 28 特征输入，返回预测+概率 | ✅ |
| `POST /api/predict_batch` | N/A (额外端点) | ✅ 批量预测 | ℹ️ 扩展 |
| `GET /api/comparison` | N/A (额外端点) | ✅ 经典vs量子对比 | ℹ️ 扩展 |

### 4.2 请求/响应模型一致性

- `PredictRequest.features`: 类型 `list[float]`，长度约束 `min_length=28, max_length=28` — 与 28 维特征约定一致 ✅
- `PredictResponse`: 包含 `prediction` (0/1)、`fraud_probability` (float)、`method` (str) — 与前端需求一致 ✅
- `ComparisonResponse`: 包含 `baseline` 和 `quantum` 指标字典 — 与前端对比表格需求一致 ✅

### 4.3 前端 qccp 页面合同

| 检查项 | 状态 |
|--------|------|
| Vue SFC 页面已生成 (`index.vue`, `data.js`) | ✅ |
| INTEGRATE.md 包含路由、i18n 配置 | ✅ |
| 路由 `/solution/quantumFraudDetection` | ✅ |
| 中英文 i18n 对象完整 | ✅ |
| 前端调用 `/api/predict` 和 `/api/comparison` | ✅ |

---

## 5. 缺失证据清单

| 编号 | 缺失项 | 影响 | 补充命令/操作 |
|------|--------|------|--------------|
| E1 | 训练损失曲线图 (loss vs epoch) | 无法直观判断 VQC 训练收敛情况 | `python -c "import matplotlib; ..."` 从 stage2 日志重绘 |
| E2 | VQC 训练后参数文件 (`vqc_params.npy`) | 后端使用随机参数而非训练结果，在线预测与实验评估不一致 | 在 stage2 代码末尾添加 `np.save("vqc_params.npy", var_params.detach().numpy())` |
| E3 | PCA 解释方差比数值 | 无法量化信息损失程度 | 在 stage1 输出中记录 `pca.explained_variance_ratio_` |
| E4 | 混淆矩阵 | 缺少详细分类错误分布 | `from sklearn.metrics import confusion_matrix` 生成 |
| E5 | ROC 曲线图 | 缺少 AUC-ROC 可视化 | 使用 `sklearn.metrics.roc_curve` 绘制 |
| E6 | 真实硬件执行证据 | 无法验证模拟器结果在硬件上的可靠性 | 在天衍/国盾平台提交作业 |
| E7 | 交叉验证结果 (k-fold) | 当前仅单次 train/test 划分，指标方差未知 | 运行 5-fold 或 10-fold CV |
| E8 | 后端 VQC 参数加载机制 | model_service.py 注释了 `np.load("vqc_params.npy")` 但未实现 | 将 E2 产出的参数文件集成到后端 |

---

## 6. 局限性说明

### 6.1 数据局限

- **合成数据集**：使用 `sklearn.make_classification` 生成，非真实信用卡交易数据。欺诈模式、特征分布与实际场景差异显著，实验结论**不可直接推广到生产环境**。
- **小样本量**：仅 1000 条样本，欺诈类 200 条。实际欺诈检测数据集通常百万级，欺诈率低于 0.2%。
- **无时间序列特征**：真实交易具有时序依赖性，当前数据集为 i.i.d. 假设。

### 6.2 预处理局限

- **PCA 信息损失**：从 28 维降至 4 维，必然丢失部分判别信息，可能同时影响经典和量子方法。
- **角度编码重复填充**：4 维 PCA 特征需填充至 8 个编码参数（4 qubits × RY+RZ），采用重复策略可能导致冗余编码。

### 6.3 量子方法局限

- **VQC 性能低于经典基线**：F1 Score 0.2476 vs 0.4211，差距显著（-41.2%）。当前配置下 VQC 不具备竞争力。
- **模拟器结果**：所有量子计算在 `StatevectorSimulator` 上执行，无噪声、无采样误差。真实硬件将引入去相干、门误差和读取噪声，预期性能更低。
- **有限差分梯度**：使用 `eps=0.01` 的有限差分近似梯度，非解析参数移位规则，梯度精度有限。
- **单量子比特输出**：仅使用 qubit 0 的 Z 期望值作为分类特征，浪费了 4 量子比特中的 75% 测量信息。
- **欠训练风险**：30 epochs、lr=0.01 的训练配置可能不足以收敛，未见训练曲线无法确认。
- **后端演示模式**：在线预测使用随机初始化参数，非训练结果，预测结果仅用于展示流程可行性。

### 6.4 实验设计局限

- **单次划分**：仅 1 次 80/20 train/test 划分，无交叉验证，指标方差未知。
- **单一基线**：仅对比 Logistic Regression，未与 Random Forest、XGBoost 等强基线对比。
- **超参数搜索缺失**：VQC 的层数、学习率、编码策略等均未做系统调优。

---

## 7. 交付决策

### 综合评估

| 评估维度 | 状态 | 说明 |
|----------|------|------|
| 指标一致性 | ✅ 通过 | 所有四处指标源完全一致 |
| 数据/预处理一致性 | ✅ 通过 | 流水线参数一致，Stage 2 复用 Stage 1 预处理 |
| 量子电路参数一致性 | ✅ 通过 | 报告/代码/后端/API 完全一致 |
| 前后端合同一致性 | ✅ 通过 | 端点、请求/响应模型、qccp 页面均匹配 |
| 性能达标 | ❌ 未通过 | VQC F1 (0.2476) 远低于方案预期 (F1 ≥ 0.80) 和基线 (0.4211) |
| 硬件证据 | ❌ 缺失 | 仅模拟器结果，无真实硬件验证 |
| 后端生产就绪 | ⚠️ 部分 | VQC 使用随机参数，非训练结果加载 |

### 交付判定：**有条件交付**

**理由**：
1. 作为**技术演示/概念验证 (PoC)**，应用可正常运行：前后端联调通过、API 响应正确、qccp 页面可展示。
2. 作为**量子优势验证**，当前不可交付：VQC 性能显著低于经典基线，无法声称量子方法在此任务上有优势。
3. 所有指标如实报告，未夸大量子方法表现，符合诚实报告原则。

**交付条件**：
- 在 README 和展示页面中**明确标注** VQC 性能低于经典基线
- 标注所有结果为**模拟器结果**，非真实硬件
- 标注后端 VQC 为**演示模式**（随机参数），非训练结果
- 完成缺失证据 E2（VQC 参数持久化）和 E8（后端参数加载）后再升级为生产就绪

**后续改进方向**：
- 增加量子比特数至 8-16 以容纳更多特征信息
- 尝试混合量子-经典架构（如 HQNN: 量子特征提取 + 经典分类头）
- 使用多量子比特期望值或量子概率层作为输出
- 引入噪声模型模拟真实硬件行为
- 获取真实信用卡数据集（如 Kaggle Credit Card Fraud Detection）
