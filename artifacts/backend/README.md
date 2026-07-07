# 金融欺诈检测量子应用 - FastAPI后端服务

使用VQC（变分量子电路）进行信用卡欺诈检测，与经典Logistic Regression基线对比。

## 快速启动

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

> 需要在项目根目录（`/artifacts`）下运行上述命令。

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### API文档

启动服务后访问 Swagger UI：

- http://127.0.0.1:8000/docs

## API端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/info` | 应用信息（算法、指标等） |
| POST | `/api/predict` | 单条交易预测 |
| POST | `/api/predict_batch` | 批量交易预测 |
| GET | `/api/comparison` | 经典vs量子指标对比 |

## API合约

### POST /api/predict

请求：
```json
{
  "features": [0.1, -0.2, 0.5, 1.3, ...]  // 28个原始特征
}
```

响应：
```json
{
  "status": "success",
  "prediction": 1,
  "fraud_probability": 0.72,
  "method": "vqc"
}
```

### POST /api/predict_batch

请求：
```json
{
  "features_list": [
    [0.1, -0.2, 0.5, ...],  // 28个特征
    [0.3, 0.1, -0.4, ...]   // 28个特征
  ]
}
```

响应：
```json
{
  "status": "success",
  "predictions": [
    {"status": "success", "prediction": 1, "fraud_probability": 0.72, "method": "vqc"},
    {"status": "success", "prediction": 0, "fraud_probability": 0.23, "method": "vqc"}
  ]
}
```

### GET /api/comparison

响应：
```json
{
  "baseline": {"method": "Logistic Regression", "f1_score": 0.4211, "auc_roc": 0.695, ...},
  "quantum": {"method": "VQC", "f1_score": 0.2476, "auc_roc": 0.5723, ...}
}
```

## 项目结构

```
artifacts/
├── backend/
│   ├── main.py            # FastAPI主应用（路由、请求校验）
│   ├── model_service.py   # 模型服务层（经典/量子预测逻辑）
│   ├── requirements.txt   # Python依赖
│   └── README.md          # 本文件
└── dist/                  # 前端构建产物（可选，自动托管）
```

## 注意事项

- 当前为**演示模式**：Scaler/PCA/LR 使用合成数据拟合，VQC 使用随机初始化参数
- 生产环境应加载 stage1/stage2 的预训练参数和模型权重
- VQC 电路使用 cqlib StatevectorSimulator 本地模拟，无需量子硬件
- 预处理流程：StandardScaler → PCA(4维) → tanh×π 角度编码
