# Stage 3: 应用打包验证记录

> 生成日期：2026-06-30
> 交付配置：local_fastapi_demo

---

## 1. 后端 (FastAPI)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 入口文件 | `backend/app.py` | FastAPI 应用，`app` 符号 |
| 启动命令 | `python backend/start.py` | uvicorn 启动，默认 `0.0.0.0:8000` |
| 健康检查 | `GET /health` | 返回 `{"status": "ok"}` |
| 应用信息 | `GET /api/info` | 模型配置、特征说明 |
| 预测接口 | `POST /api/predict` | 支持经典/量子混合模型推理 |
| 指标对比 | `GET /api/metrics` | 实时评估两种模型 RMSE/MAE/MAPE |
| 样本数据 | `GET /api/sample-data` | 测试集样本 + 双模型预测 |
| 训练曲线 | `GET /api/train-curves` | 训练/验证损失曲线 |
| 训练接口 | `POST /api/train` | 同步触发模型训练 |
| 静态文件 | `/static` 挂载 | frontend/ 目录 |
| 根路由 | `GET /` | 返回前端首页 |

## 2. 前端 (Vue3 + ECharts)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 入口文件 | `frontend/index.html` | 单页应用，CDN 加载 Vue3/Element Plus/ECharts |
| UI 框架 | Vue 3 + Element Plus | 深色科技风格 |
| 可视化 | ECharts 5 | 预测曲线对比、训练损失曲线、指标对比 |
| API 调用 | fetch → /api/* | 与后端完全对应 |
| 双模型展示 | 支持 | 经典 LSTM / 量子混合 VQC+LSTM |

## 3. 打包完整性

- 后端可独立启动：`python backend/start.py`
- 前端通过 FastAPI StaticFiles 挂载，无需单独部署
- 模型权重预置：`artifacts/classical_lstm.pth`、`artifacts/hybrid_vqc_lstm.pth`
- 数据文件预置：`data/rainfall_data.npz`
- 依赖声明：`requirements.txt`

## 4. 验证命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python backend/start.py

# 健康检查
curl http://localhost:8000/health

# 获取应用信息
curl http://localhost:8000/api/info

# 使用样本数据预测
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model_type": "classical", "use_sample": true}'

# 获取指标
curl http://localhost:8000/api/metrics
```
