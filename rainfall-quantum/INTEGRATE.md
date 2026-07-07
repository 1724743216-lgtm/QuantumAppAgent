# 集成指南：短时降雨量量子混合预测系统

本文档指导如何部署、配置和验证短时降雨量量子混合预测系统，以及如何接入真实数据替换合成数据。

---

## 1. 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Linux / macOS / Windows (WSL2) | Ubuntu 20.04+ |
| Python | 3.9 | 3.10+ |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 500 MB | 1 GB+ |
| GPU | 不需要（CPU 推理即可） | 可选，CUDA 加速训练 |

> 量子电路通过 cqlib StatevectorSimulator 在 CPU 上模拟执行，当前配置（6 qubits）无需 GPU。

---

## 2. 安装步骤

### 2.1 克隆项目

```bash
cd /your-workspace
# 假设项目已在 /rainfall-quantum
```

### 2.2 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 2.3 安装依赖

```bash
cd /rainfall-quantum
pip install -r requirements.txt
```

核心依赖：
- `fastapi>=0.104.0` — Web 框架
- `uvicorn[standard]>=0.24.0` — ASGI 服务器
- `pydantic>=2.0.0` — 数据校验
- `torch>=2.0.0` — 深度学习框架
- `numpy>=1.24.0` — 数值计算
- `scipy>=1.10.0` — 数据生成依赖
- `cqlib-sdk>=1.0.0` — 量子计算 SDK

### 2.4 生成数据与训练模型

项目 `artifacts/` 和 `data/` 目录下已包含预训练模型和数据文件。如需从头训练：

```bash
# 1. 生成合成数据（如 data/rainfall_data.npz 不存在，训练脚本会自动生成）
python train_baseline.py

# 2. 训练量子混合模型（依赖经典基线生成的数据文件）
python train_hybrid.py
```

训练完成后，权重文件保存在 `artifacts/` 目录：
- `classical_lstm.pth` — 经典 LSTM 模型
- `hybrid_vqc_lstm.pth` — 量子混合模型

---

## 3. 配置说明

### 3.1 模型参数

模型参数在 `backend/app.py` 顶部常量中定义：

```python
INPUT_SIZE = 7       # 输入特征维度
HIDDEN_SIZE = 64     # LSTM 隐层大小
NUM_LAYERS = 2       # LSTM 层数
PRED_LEN = 10        # 预测步数（10步 × 6分钟 = 60分钟）
N_QUBITS = 6         # 量子比特数
VQC_LAYERS = 2       # VQC 变分层深度
DEVICE = "cpu"       # 推理设备
```

如需修改，直接编辑 `backend/app.py` 中对应常量，然后重新启动服务。

### 3.2 服务端口

默认监听 `0.0.0.0:8000`。修改 `backend/start.py` 中的 `host` 和 `port` 参数：

```python
uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
```

### 3.3 前端静态文件

前端文件位于 `frontend/index.html`，由 FastAPI 直接 serve。修改前端文件后刷新浏览器即可生效（无需重启服务）。

---

## 4. 启动命令

```bash
# 确保在项目根目录
cd /rainfall-quantum

# 方式 1：使用启动脚本
python backend/start.py

# 方式 2：直接使用 uvicorn
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 方式 3：后台运行
nohup python backend/start.py > server.log 2>&1 &
```

---

## 5. 验证步骤

### 5.1 健康检查

```bash
curl http://localhost:8000/health
# 预期响应: {"status":"ok","service":"rainfall-quantum-prediction"}
```

### 5.2 模型加载检查

```bash
curl http://localhost:8000/api/info
# 预期响应: 包含 models 数组，两个模型参数量 > 0
```

检查返回 JSON 中 `models` 字段：
- `classical` 的 `params` 应为约 17,000
- `hybrid` 的 `params` 应为 52,536，`quantum_params` 应为 30

### 5.3 推理验证

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "classical",
    "input_data": [[0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                   [0.5,0.5,0.5,0.5,0.5,0.5,0.5]]
  }'
# 预期响应: 包含 predictions 数组（10个浮点数）和 rmse 值
```

### 5.4 指标对比验证

```bash
curl http://localhost:8000/api/metrics
# 预期响应: classical RMSE≈0.1753, hybrid RMSE≈0.1787
```

### 5.5 前端页面验证

1. 打开浏览器访问 `http://localhost:8000`
2. 确认页面正常加载，显示"短时降雨量预测系统"标题
3. 点击样本数据，确认预测曲线可正常展示

---

## 6. 真实数据接入指南

当前系统使用合成数据（基于 Z-R 关系生成）。以下说明如何替换为真实气象数据。

### 6.1 支持的数据集

| 数据集 | 来源 | 说明 |
|--------|------|------|
| **SEVIR** | NOAA, 2020 | 包含雷达反射率、卫星红外等气象观测的时空序列数据集，适合降雨量预测 |
| **CIKM 2017** | CIKM Cup 2017 | 降雨量预测竞赛数据，包含雷达反射率和降雨量标签 |
| **HKUST** | 自采数据 | 香港地区气象站时序数据 |

### 6.2 数据格式要求

系统期望的 `.npz` 文件格式：

```python
# rainfall_data.npz 须包含以下键:
{
    # 输入序列 (归一化后)
    "train_inputs": np.ndarray,   # (5000, 10, 7)  - 训练输入
    "val_inputs":   np.ndarray,   # (1000, 10, 7)  - 验证输入
    "test_inputs":  np.ndarray,   # (1000, 10, 7)  - 测试输入

    # 目标序列 (归一化后)
    "train_targets": np.ndarray,  # (5000, 10)     - 训练目标
    "val_targets":   np.ndarray,  # (1000, 10)     - 验证目标
    "test_targets":  np.ndarray,  # (1000, 10)     - 测试目标

    # 归一化参数
    "feat_min":   np.ndarray,     # (7,) 特征最小值
    "feat_max":   np.ndarray,     # (7,) 特征最大值
    "feat_range": np.ndarray,     # (7,) 特征范围 (max-min)
    "tgt_min":    float,          # 目标最小值
    "tgt_max":    float,          # 目标最大值
    "tgt_range":  float,          # 目标范围
}
```

### 6.3 接入 SEVIR 数据集的步骤

1. **下载 SEVIR 数据**：
   ```bash
   # 从 https://sevir.mit.edu/ 下载
   # 需要 radar (VIL) 和 precipitation 数据
   ```

2. **编写数据转换脚本**：

   ```python
   # scripts/convert_sevir.py
   import numpy as np

   def convert_sevir_to_npz(sevir_dir, output_path):
       """
       将 SEVIR 数据转换为 rainfall_data.npz 格式。

       步骤:
         1. 读取 SEVIR 的 HDF5 文件
         2. 提取 7 个特征: 雷达反射率、温度、湿度、风速、气压、风向、过去降水
         3. 滑动窗口构建 (input_seq, target_seq) 对
         4. Min-Max 归一化
         5. 按 5:1:1 分割 train/val/test
         6. 保存为 .npz
       """
       # TODO: 根据实际 SEVIR 数据格式实现
       pass
   ```

3. **替换数据文件**：
   ```bash
   # 备份合成数据
   cp data/rainfall_data.npz data/rainfall_data_synthetic.npz

   # 复制转换后的真实数据
   cp /path/to/converted_sevir.npz data/rainfall_data.npz
   ```

4. **重新训练模型**：
   ```bash
   python train_baseline.py
   python train_hybrid.py
   ```

5. **重启服务**：
   ```bash
   python backend/start.py
   ```

### 6.4 接入 CIKM 2017 数据集的步骤

1. 下载 CIKM 2017 数据（雷达反射率 + 降雨量标签）
2. 将雷达反射率作为特征 0（`radar_reflectivity`），填充其余 6 个特征（可设为常数或从其他气象源补充）
3. 按 6.2 节格式转换为 `.npz`
4. 替换 `data/rainfall_data.npz` 并重新训练

### 6.5 自定义数据接入注意事项

- **特征对齐**：7 个特征的顺序须与 `data/synthetic_rainfall.py` 一致：雷达反射率、温度、湿度、风速、气压、风向、过去降水。如特征不足，可填充零值但需注意影响预测精度。
- **归一化**：务必使用 Min-Max 归一化到 [0, 1] 区间，并保存 `feat_min/feat_max/feat_range` 和 `tgt_min/tgt_max/tgt_range` 以支持推理时反归一化。
- **序列长度**：输入序列 10 步、预测序列 10 步。如需修改，须同步更新 `backend/app.py` 中的 `PRED_LEN` 常量和前端展示逻辑。
- **数据规模**：建议训练集不少于 5000 样本以保证模型收敛。

---

## 7. 故障排除

### 7.1 服务启动失败

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'cqlib'` | 未安装 cqlib-sdk | `pip install cqlib-sdk` |
| `ModuleNotFoundError: No module named 'fastapi'` | 未安装依赖 | `pip install -r requirements.txt` |
| `FileNotFoundError: rainfall_data.npz` | 数据文件缺失 | 运行 `python train_baseline.py` 生成 |
| `Port 8000 already in use` | 端口占用 | 修改 `backend/start.py` 中的端口号，或 `kill` 占用进程 |

### 7.2 模型加载问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 启动日志 `⚠ 模型文件不存在` | 未训练模型或权重文件被删除 | 运行训练脚本重新生成 |
| 推理结果全为零 | 模型权重未正确加载 | 检查 `artifacts/*.pth` 文件完整性，重新训练 |
| `RuntimeError: size mismatch` | 模型参数与权重文件不匹配 | 确认 `backend/app.py` 中的参数配置与训练时一致 |

### 7.3 量子模拟器问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 推理/训练速度极慢 | 量子电路逐样本模拟 | 正常现象（6 qubits Statevector 约 6ms/样本），可减少批量大小 |
| `cqlib` 导入错误 | cqlib 版本不兼容 | `pip install cqlib-sdk>=1.0.0 --upgrade` |
| 量子层输出 NaN | 变分参数发散 | 降低量子学习率（如 0.0001），或增加训练轮数 |

### 7.4 前端问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 页面空白 | CDN 资源加载失败 | 检查网络，或本地部署 Vue/ECharts |
| 图表不显示 | API 返回异常 | 打开浏览器控制台查看错误，确认后端正常 |
| 预测按钮无响应 | 输入数据格式错误 | 确认输入为 10×7 数组 |

### 7.5 训练问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 训练 loss 不下降 | 学习率过大/过小 | 调整 `lr` 参数，经典基线建议 0.001 |
| 混合模型 val_loss 震荡 | 量子参数不稳定 | 降低 `quantum_lr`（如 0.0003） |
| OOM (内存不足) | 批量过大 | 减小 `batch_size`（经典 32，混合 16） |

---

## 8. 联系与支持

如遇到上述故障排除中未涵盖的问题，请检查以下资源：

- 项目 README：`/rainfall-quantum/README.md`
- 验证报告：`/rainfall-quantum/experiments/stage4_verification_handoff/verification_report.md`
- cqlib SDK 文档：https://cqlib.dev
