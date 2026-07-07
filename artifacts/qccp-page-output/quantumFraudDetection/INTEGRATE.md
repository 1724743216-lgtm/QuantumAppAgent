# INTEGRATE.md — quantumFraudDetection

## Page Info

| Field | Value |
| --- | --- |
| Page name | 量子金融欺诈检测 / Quantum Financial Fraud Detection |
| pageKey | `quantumFraudDetection` |
| Module | solution |
| Route | `/solution/quantumFraudDetection` |
| Uses Footer | Yes |
| Login required | No |

## File Copy Destinations

| Source (generated) | Destination (in qccp-web) |
| --- | --- |
| `project-files/src/views/solution/quantumFraudDetection/index.vue` | `src/views/solution/quantumFraudDetection/index.vue` |
| `project-files/src/views/solution/quantumFraudDetection/data.js` | `src/views/solution/quantumFraudDetection/data.js` |

## Route Object

Append to the solution routes array:

```js
{
  path: '/solution/quantumFraudDetection',
  name: 'QuantumFraudDetection',
  component: () => import('@/views/solution/quantumFraudDetection/index.vue'),
  meta: {
    title: '量子金融欺诈检测',
    module: 'solution',
  },
},
```

## Chinese i18n Object

Append to `src/utils/lang/zh.js`:

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

## English i18n Object

Append to `src/utils/lang/en.js`:

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
    epochs: 'Training Epochs',
  },
  actions: {
    predict: 'Predict',
    reset: 'Reset',
    retry: 'Retry',
  },
  message: {
    loadFailed: 'Failed to load data',
    networkError: 'Network error. Please try again later.',
    mockPrediction: 'Using simulated data for prediction result',
  },
},
```

## npm Dependencies

All dependencies are already present in qccp-web:

- `vue` (v3.x)
- `vue-router` (v4.x)
- `vue-i18n` (v9.x)
- `element-plus`

No new dependencies required.

## API Specification

### GET /api/comparison

| Field | Value |
| --- | --- |
| URL | `/api/comparison` |
| Method | GET |
| apiCode | (to be provided by backend) |
| Request params | none |
| Response (success) | `{ code: 200, data: { baseline: { method, f1_score, auc_roc, precision, recall, accuracy }, quantum: { method, f1_score, auc_roc, precision, recall, accuracy } } }` |

**Fallback**: When the API is unavailable, the page loads comparison data from `data.js` (`comparisonMetrics`).

### POST /api/predict

| Field | Value |
| --- | --- |
| URL | `/api/predict` |
| Method | POST |
| apiCode | (to be provided by backend) |
| Request body | `{ amount: number, timeGap: number, freqRatio: number, riskScore: number }` |
| Response (success) | `{ code: 200, data: { label: 0 \| 1, probability: string } }` |

**Fallback**: When the API is unavailable, the page simulates a prediction using the risk score threshold and returns a mock result with a notification.

## When Real API Is Available

1. Replace `fetch('/api/comparison')` in `loadMetrics()` with the project `axios` instance from `@/utils/axios.js`, using the correct `apiCode` header.
2. Replace `fetch('/api/predict', ...)` in `submitPrediction()` with the project `axios` instance.
3. Create `src/api/quantumFraudDetection/index.js` with `getComparisonData()` and `postPrediction()` functions following the API Rules pattern.
4. Remove the mock fallback logic (try/catch fallback blocks) once the API is confirmed working.
5. Keep `data.js` for development preview but it will no longer be needed in production.

## Verification

```bash
npm run build
```

The page must compile without errors and all i18n keys must resolve in both `zh` and `en` locales.
