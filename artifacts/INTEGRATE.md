# INTEGRATE.md — 金融欺诈检测量子应用集成指南

本文档描述如何将金融欺诈检测量子应用的后端服务和前端页面集成到目标环境中。

---

## 1. 后端部署步骤

### 1.1 后端技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | Python FastAPI 应用服务 |
| 量子计算 | cqlib + StatevectorSimulator | Cqlib SDK 本地模拟器 |
| 经典 ML | scikit-learn | Logistic Regression 基线 |
| 深度学习 | PyTorch | VQC 参数优化（训练阶段） |
| 数据处理 | numpy | 数值计算 |

> **后端路径**: Python FastAPI 应用服务（非 Java qccp-service 集成路径）

### 1.2 环境准备

```bash
# 1. 创建 Python 虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 2. 安装依赖
cd /artifacts
pip install -r backend/requirements.txt
```

依赖列表：
- `fastapi>=0.100.0`
- `uvicorn>=0.20.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `cqlib>=1.0.0`
- `torch>=2.0.0`
- `pydantic>=2.0.0`

### 1.3 启动后端服务

```bash
# 开发环境
cd /artifacts
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 生产环境
cd /artifacts
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 1.4 Docker 部署（可选）

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY artifacts/backend/ ./backend/
COPY artifacts/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t quantum-fraud-detection .
docker run -p 8000:8000 quantum-fraud-detection
```

### 1.5 健康检查

```bash
# 基础健康检查
curl http://127.0.0.1:8000/health
# 预期: {"status":"ok"}

# 详细信息检查
curl http://127.0.0.1:8000/api/info
# 预期: 包含 algorithm, metrics, backend 等字段
```

---

## 2. 前端集成步骤

前端页面已按 qccp-web 平台规范生成，集成步骤如下：

### 2.1 页面信息

| 字段 | 值 |
|------|-----|
| 页面名称 | 量子金融欺诈检测 / Quantum Financial Fraud Detection |
| pageKey | `quantumFraudDetection` |
| 模块 | solution |
| 路由 | `/solution/quantumFraudDetection` |
| 使用 Footer | 是 |
| 需要登录 | 否 |

### 2.2 文件复制

将以下文件复制到 qccp-web 项目对应目录：

| 源文件（已生成） | 目标位置（qccp-web 中） |
|-----------------|------------------------|
| `qccp-page-output/quantumFraudDetection/project-files/src/views/solution/quantumFraudDetection/index.vue` | `src/views/solution/quantumFraudDetection/index.vue` |
| `qccp-page-output/quantumFraudDetection/project-files/src/views/solution/quantumFraudDetection/data.js` | `src/views/solution/quantumFraudDetection/data.js` |

```bash
# 示例命令（根据实际 qccp-web 项目路径调整）
QCCP_WEB_ROOT=/path/to/qccp-web

cp /artifacts/qccp-page-output/quantumFraudDetection/project-files/src/views/solution/quantumFraudDetection/index.vue \
   ${QCCP_WEB_ROOT}/src/views/solution/quantumFraudDetection/index.vue

cp /artifacts/qccp-page-output/quantumFraudDetection/project-files/src/views/solution/quantumFraudDetection/data.js \
   ${QCCP_WEB_ROOT}/src/views/solution/quantumFraudDetection/data.js
```

### 2.3 路由注册

将以下路由对象追加到 qccp-web 的 solution 路由数组中：

```js
{
  path: '/solution/quantumFraudDetection',
  name: 'QuantumFraudDetection',
  component: () => import('@/views/solution/quantumFraudDetection/index.vue'),
  meta: {
    title: '量子金融欺诈检测',
    module: 'solution',
  },
}
```

### 2.4 国际化（i18n）配置

#### 中文（追加到 `src/utils/lang/zh.js`）

```js
quantumFraudDetection: {
  banner: {
    title: '量子金融欺诈检测',
    subtitle: '基于变分量子电路的智能风控解决方案',
  },
  architecture: {
    title: '算法架构',
    step1: {
      title: '数据预处理',
      desc: '原始交易特征经过标准化和PCA降维，匹配量子线路输入维度。',
    },
    step2: {
      title: '量子编码',
      desc: '经典特征向量通过角度编码映射到量子态，使用Ry旋转门实现。',
    },
    step3: {
      title: '变分线路',
      desc: '参数化ansatz通过纠缠CNOT门和可训练Ry/Rz旋转门处理量子态。',
    },
    step4: {
      title: '量子测量',
      desc: '在计算基下对量子比特进行测量，产生经典比特串作为输出特征。',
    },
    step5: {
      title: '分类输出',
      desc: '测量期望值输入经典后处理层，进行欺诈/合法二分类预测。',
    },
  },
  metrics: {
    title: '指标对比',
    metricCol: '评估指标',
    baselineCol: '经典方法 (Logistic Regression)',
    quantumCol: '量子方法 (VQC)',
    f1Score: 'F1 Score',
    aucRoc: 'AUC-ROC',
    precision: 'Precision',
    recall: 'Recall',
    accuracy: 'Accuracy',
    radarNote: '上方表格展示了经典逻辑回归与变分量子电路(VQC)在信用卡欺诈检测任务上的关键指标对比。在当前4量子比特、2层ansatz的配置下，VQC模型展现了量子计算的可行性，但指标仍有提升空间，后续可通过增加量子比特数、优化ansatz结构或引入混合量子-经典架构进一步提升性能。',
  },
  detection: {
    title: '在线检测',
  },
  form: {
    amountLabel: '交易金额 (归一化)',
    amountPlaceholder: '请输入归一化后的交易金额',
    timeGapLabel: '时间间隔 (归一化)',
    timeGapPlaceholder: '请输入与上次交易的时间间隔',
    freqRatioLabel: '频率比率 (归一化)',
    freqRatioPlaceholder: '请输入交易频率比率',
    riskScoreLabel: '风险评分 (归一化)',
    riskScorePlaceholder: '请输入综合风险评分',
    validation: {
      amountRequired: '请输入交易金额',
      timeGapRequired: '请输入时间间隔',
      freqRatioRequired: '请输入频率比率',
      riskScoreRequired: '请输入风险评分',
      numberInvalid: '请输入有效数值',
    },
  },
  result: {
    title: '预测结果',
    label: '检测结果',
    probability: '欺诈概率',
    fraud: '欺诈',
    legitimate: '合法',
  },
  params: {
    title: '算法参数',
    qubits: '量子比特数',
    layers: 'Ansatz层数',
    encoding: '编码方式',
    optimizer: '优化器',
    learningRate: '学习率',
    epochs: '训练轮次',
  },
  actions: {
    predict: '开始预测',
    reset: '重置',
    retry: '重试',
  },
  message: {
    loadFailed: '数据加载失败',
    networkError: '网络异常，请稍后重试',
    mockPrediction: '当前使用模拟数据返回预测结果',
  },
},
```

#### 英文（追加到 `src/utils/lang/en.js`）

```js
quantumFraudDetection: {
  banner: {
    title: 'Quantum Financial Fraud Detection',
    subtitle: 'Intelligent Risk Control Solution Based on Variational Quantum Circuits',
  },
  architecture: {
    title: 'Algorithm Architecture',
    step1: {
      title: 'Data Preprocessing',
      desc: 'Raw transaction features are normalized and PCA-transformed to match quantum circuit input dimensions.',
    },
    step2: {
      title: 'Quantum Encoding',
      desc: 'Classical feature vectors are mapped onto quantum states via angle encoding using Ry rotations.',
    },
    step3: {
      title: 'Variational Circuit',
      desc: 'A parameterized ansatz with entangling CNOT gates and trainable Ry/Rz rotations processes the quantum state.',
    },
    step4: {
      title: 'Measurement',
      desc: 'Qubits are measured in the computational basis to produce classical bitstrings as output features.',
    },
    step5: {
      title: 'Classification',
      desc: 'Measured expectation values are fed into a classical post-processing layer for binary fraud/legitimate prediction.',
    },
  },
  metrics: {
    title: 'Metrics Comparison',
    metricCol: 'Metric',
    baselineCol: 'Classical (Logistic Regression)',
    quantumCol: 'Quantum (VQC)',
    f1Score: 'F1 Score',
    aucRoc: 'AUC-ROC',
    precision: 'Precision',
    recall: 'Recall',
    accuracy: 'Accuracy',
    radarNote: 'The table above compares key metrics between classical logistic regression and variational quantum circuit (VQC) on the credit card fraud detection task. Under the current 4-qubit, 2-layer ansatz configuration, the VQC model demonstrates the feasibility of quantum computing, but metrics still have room for improvement. Performance can be further enhanced by increasing qubit count, optimizing ansatz structure, or introducing hybrid quantum-classical architectures.',
  },
  detection: {
    title: 'Online Detection',
  },
  form: {
    amountLabel: 'Transaction Amount (Normalized)',
    amountPlaceholder: 'Enter normalized transaction amount',
    timeGapLabel: 'Time Gap (Normalized)',
    timeGapPlaceholder: 'Enter time gap since last transaction',
    freqRatioLabel: 'Frequency Ratio (Normalized)',
    freqRatioPlaceholder: 'Enter transaction frequency ratio',
    riskScoreLabel: 'Risk Score (Normalized)',
    riskScorePlaceholder: 'Enter composite risk score',
    validation: {
      amountRequired: 'Please enter transaction amount',
      timeGapRequired: 'Please enter time gap',
      freqRatioRequired: 'Please enter frequency ratio',
      riskScoreRequired: 'Please enter risk score',
      numberInvalid: 'Please enter a valid number',
    },
  },
  result: {
    title: 'Prediction Result',
    label: 'Detection Result',
    probability: 'Fraud Probability',
    fraud: 'Fraud',
    legitimate: 'Legitimate',
  },
  params: {
    title: 'Algorithm Parameters',
    qubits: 'Qubits',
    layers: 'Ansatz Layers',
    encoding: 'Encoding',
    optimizer: 'Optimizer',
    learningRate: 'Learning Rate',
    epochs: 'Epochs',
  },
  actions: {
    predict: 'Predict',
    reset: 'Reset',
    retry: 'Retry',
  },
  message: {
    loadFailed: 'Data loading failed',
    networkError: 'Network error, please try again later',
    mockPrediction: 'Currently using simulated data for predictions',
  },
},
```

### 2.5 前端 API 代理配置

前端页面调用以下后端端点，需在 qccp-web 开发服务器中配置代理：

| 前端调用 | 后端端点 | 说明 |
|----------|---------|------|
| `GET /api/info` | FastAPI | 应用信息 |
| `POST /api/predict` | FastAPI | 单条预测 |
| `GET /api/comparison` | FastAPI | 指标对比 |

**Vue CLI / Vite 代理配置示例**：

```js
// vite.config.js 或 vue.config.js
proxy: {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
  },
}
```

---

## 3. 环境变量说明

当前后端无需额外环境变量即可启动。以下为可选配置：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `VQC_PARAMS_PATH` | 无 | VQC 训练参数文件路径（如 `vqc_params.npy`），设置后将从文件加载参数而非随机初始化 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `8000` | 服务监听端口 |
| `LOG_LEVEL` | `info` | 日志级别 |

> 注意：当前 `model_service.py` 未实现 `VQC_PARAMS_PATH` 环境变量读取逻辑。启用训练参数加载需修改代码，参见[已知限制](#5-已知限制)。

---

## 4. 健康检查命令

```bash
# 基础存活检查
curl -sf http://127.0.0.1:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'"

# 详细就绪检查（含模型初始化验证）
curl -sf http://127.0.0.1:8000/api/info | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'algorithm' in d
assert d['n_qubits'] == 4
print('Backend ready: algorithm=%s, qubits=%d' % (d['algorithm'], d['n_qubits']))
"

# 预测端点检查
curl -sf -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.2, 0.5, 1.3, -0.7, 0.2, 0.8, -1.1, 0.3, -0.5, 0.9, 1.2, -0.3, 0.7, 0.1, -0.8, 0.4, 0.6, -1.0, 0.2, 0.5, -0.9, 0.3, 1.1, -0.6, 0.8, -0.4, 0.2]}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='success'; print('Prediction: %d (prob=%.4f)' % (d['prediction'], d['fraud_probability']))"
```

---

## 5. 已知限制

| 编号 | 限制 | 影响 | 缓解措施 |
|------|------|------|----------|
| L1 | VQC 后端使用随机初始化参数 | 在线预测结果与实验评估不一致，无实际判别能力 | 将训练后参数持久化为 `vqc_params.npy` 并在 `model_service.py` 中加载 |
| L2 | 模拟器执行，无噪声模型 | 真实硬件预期性能更低 | 引入 cqlib 噪声模型或在天衍/国盾平台提交硬件作业 |
| L3 | 合成数据集 | 后端 Scaler/PCA/LR 在合成数据上拟合，真实数据分布不同 | 替换为真实数据集，重新训练所有模型 |
| L4 | 无认证/授权 | API 端点无访问控制 | 生产环境需添加 API Key / OAuth 认证 |
| L5 | 无请求限流 | 批量预测端点可被滥用 | 添加 rate limiting 中间件 |
| L6 | 单进程同步预测 | VQC 预测为 CPU 密集型，高并发下延迟高 | 使用异步任务队列或多 worker 部署 |
| L7 | VQC 性能低于经典基线 | 不适合生产风控场景 | 本应用仅作为技术演示，不可用于实际欺诈检测决策 |

---

## 6. 前后端联调验证清单

- [ ] 后端 `/health` 返回 `{"status":"ok"}`
- [ ] 后端 `/api/info` 返回正确算法参数
- [ ] 后端 `/api/predict` 接受 28 维特征并返回预测
- [ ] 后端 `/api/comparison` 返回经典和量子指标
- [ ] 前端页面路由 `/solution/quantumFraudDetection` 可访问
- [ ] 前端指标对比表格正确展示
- [ ] 前端在线检测表单提交后返回预测结果
- [ ] 前端中英文切换正常
