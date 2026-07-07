# 短时降雨量量子混合预测系统

基于经典 LSTM 与量子混合 VQC+LSTM 的短时降雨量预测应用。预测未来 1 小时逐 6 分钟降雨量（共 10 步），提供 FastAPI 后端推理服务和 Vue3/ECharts 前端可视化界面。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       前端 (Vue3 + ECharts)                      │
│   预测曲线对比 │ 训练损失曲线 │ 指标对比 │ 样本数据浏览           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP API
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI 后端 (backend/app.py)                  │
│  /api/predict  /api/metrics  /api/sample-data  /api/train-curves│
│  /api/info     /api/train     /health                           │
└──────┬─────────────────────────────────────────┬────────────────┘
       │                                         │
┌──────▼──────────┐                    ┌────────▼────────┐
│  经典 LSTM 基线  │                    │ 量子混合 VQC+LSTM│
│  models/         │                    │ models/          │
│  classical_lstm  │                    │ hybrid_vqc_lstm  │
│  .py             │                    │ .py              │
│                  │                    │                  │
│  LSTM(2层,h=64)  │                    │ LSTM → 投影层 →  │
│  → FC(64→10)    │                    │ VQC(6q,2层) →    │
│                  │                    │ 残差融合 → FC     │
└──────┬──────────┘                    └────────┬────────┘
       │                                        │
       └──────────────┬─────────────────────────┘
                      │
              ┌───────▼───────┐
              │  数据层        │
              │  data/         │
              │  rainfall_data │
              │  .npz          │
              └───────────────┘
```

### 量子混合模型内部结构

```
输入 (batch, 10, 7)
    │
    ▼
LSTM 编码器 (2层, hidden=64)
    │
    ▼
投影层 Linear(64 → 6)
    │
    ├──────────────────────┐
    ▼                      │
VQC 量子特征提取            │
  · 角度编码 (Ry门)         │
  · 变分层 ×2               │
    (Ry+Rz+CX纠缠)         │
  · Z期望值测量             │
    │                      │
    ▼                      │
量子输出 (batch, 6)         │
    │                      │
    └──── concat ──────────┘
           │
           ▼
    融合层 Linear(12 → 6)
           │
           ▼
    预测头 Linear(6 → 10)
           │
           ▼
输出 (batch, 10)  — 未来1h逐6min降雨量
```

## 快速开始

### 环境要求

- Python 3.9+
- pip

### 安装依赖

```bash
cd /rainfall-quantum
pip install -r requirements.txt
```

### 启动服务

```bash
# 从项目根目录启动
python backend/start.py
```

服务启动后将监听 `http://0.0.0.0:8000`。打开浏览器访问即可看到前端页面。

### 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 查看应用信息
curl http://localhost:8000/api/info
```

## API 接口文档

### `GET /health`

健康检查。

**响应示例**：
```json
{"status": "ok", "service": "rainfall-quantum-prediction"}
```

### `GET /api/info`

获取应用信息、模型配置和特征说明。

### `POST /api/predict`

使用指定模型进行单样本预测。

**请求体**：
```json
{
  "model_type": "classical",
  "input_data": [[...], [...], ...]  // 10×7 二维数组
}
```

**响应示例**：
```json
{
  "predictions": [0.12, 0.15, ...],
  "model_type": "classical",
  "rmse": 0.1753
}
```

### `GET /api/metrics`

获取经典与量子模型在测试集上的实时评估指标。

**响应示例**：
```json
{
  "classical": {"RMSE": 0.1753, "MAE": 0.1213, "MAPE": 79.24},
  "hybrid":    {"RMSE": 0.1787, "MAE": 0.1239, "MAPE": 77.94},
  "unit": {"RMSE": "mm", "MAE": "mm", "MAPE": "%"}
}
```

### `GET /api/sample-data?n_samples=5`

获取测试集样本及两种模型的预测结果。

### `GET /api/train-curves`

获取训练/验证损失曲线数据。

### `POST /api/train`

触发模型训练（同步执行，注意可能阻塞）。

**请求体**：
```json
{
  "model_type": "hybrid",
  "epochs": 20
}
```

## 模型说明

### 经典 LSTM 基线

| 参数 | 值 |
|------|-----|
| 架构 | 2 层 LSTM + 全连接预测头 |
| hidden_size | 64 |
| pred_len | 10 |
| 训练轮数 | 50 |
| 批大小 | 64 |
| 学习率 | 0.001 |

### 量子混合 VQC+LSTM

| 参数 | 值 |
|------|-----|
| LSTM 编码器 | 与经典基线一致 |
| 量子比特数 | 6 |
| VQC 变分层 | 2 |
| 量子参数 | 30 |
| 总参数 | 52,536 |
| 编码方式 | 角度编码 (Ry门) |
| 变分结构 | Ry+Rz+CX 线性纠缠 |
| 测量 | Z 期望值 |
| 训练轮数 | 30 |
| 学习率 | 经典 0.001 / 量子 0.0005 |
| 执行后端 | cqlib StatevectorSimulator |

## 性能对比

| 指标 | 经典 LSTM | 量子混合 VQC+LSTM | 差值 |
|------|-----------|---------------------|------|
| **RMSE (mm)** | 0.1753 | 0.1787 | +0.0034 |
| **MAE (mm)** | 0.1213 | 0.1239 | +0.0026 |
| **MAPE (%)** | 79.24 | **77.94** | **-1.30** |
| 训练耗时 | 13.7s | 82.5s | +68.8s |

> **说明**：量子混合模型 RMSE 略高于经典基线（+1.9%），MAPE 略优（-1.6%）。当前结果基于 cqlib StatevectorSimulator 模拟器和合成数据，不代表真实量子硬件性能。详见 [验证报告](experiments/stage4_verification_handoff/verification_report.md)。

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 深度学习 | PyTorch |
| 量子计算 | cqlib-sdk (StatevectorSimulator) |
| 前端 | Vue 3 + ECharts 5 |
| 数据处理 | NumPy + SciPy |
| 语言 | Python 3.9+ |

## 目录结构

```
rainfall-quantum/
├── backend/
│   ├── app.py              # FastAPI 应用（路由、模型加载、推理）
│   └── start.py            # 启动脚本
├── frontend/
│   └── index.html          # Vue3 + ECharts 单页面前端
├── models/
│   ├── __init__.py
│   ├── classical_lstm.py   # 经典 LSTM 基线模型
│   ├── hybrid_vqc_lstm.py  # 量子混合 VQC+LSTM 模型
│   └── quantum_layer.py    # VQC 量子特征提取层 (cqlib)
├── data/
│   ├── __init__.py
│   ├── synthetic_rainfall.py  # 合成数据生成器
│   ├── dataset.py             # PyTorch Dataset
│   └── rainfall_data.npz      # 生成的数据文件
├── artifacts/
│   ├── classical_lstm.pth     # 经典模型权重
│   ├── hybrid_vqc_lstm.pth    # 混合模型权重
│   ├── train_curves.npz       # 经典模型训练曲线
│   └── hybrid_train_curves.npz # 混合模型训练曲线
├── experiments/
│   ├── stage1_scope_baseline/
│   │   └── baseline_report.json   # 经典基线评估报告
│   ├── stage2_quantum_method/
│   │   └── quantum_report.json    # 量子混合模型评估报告
│   ├── stage3_app_packaging/
│   └── stage4_verification_handoff/
│       └── verification_report.md # 验证报告
├── train_baseline.py           # 经典基线训练脚本
├── train_hybrid.py             # 量子混合模型训练脚本
├── requirements.txt            # Python 依赖
├── solution_plan.md            # 解决方案计划
├── README.md                   # 本文件
└── INTEGRATE.md                # 集成指南
```

## 许可

本项目仅供研究演示使用。
