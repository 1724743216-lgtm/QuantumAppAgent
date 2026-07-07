# 金融欺诈检测量子应用方案计划

## Stage 1: 应用范围与经典基线

**目标**: 确定需求、数据、指标，建立可运行的经典基线

**成功信号**:
- requirements.json 完整
- 经典基线可运行，F1 >= 0.80
- baseline_report.json 生成

**执行命令**: `python stage1_baseline.py`

**产出物**: requirements.json, solution_plan.md, baseline_report.json

---

## Stage 2: 量子方法实现

**目标**: 用Cqlib实现VQC欺诈检测，与基线公平对比

**成功信号**:
- VQC模型可训练和预测
- quantum_report.json 生成，使用相同数据/指标
- 量子结果与基线可比

**执行命令**: `python stage2_quantum.py`

**产出物**: quantum_report.json, VQC模型代码

**算法细节**:
- 编码: Angle encoding（RY+RZ per qubit）
- 线路: 4 qubits, 2 layers, CNOT entanglement
- 输出: Z expectation → sigmoid → 二分类
- 优化: Adam, lr=0.01, 30 epochs
- 特征: PCA降维至4维匹配4个量子比特

---

## Stage 3: 应用打包

**目标**: 构建FastAPI后端 + qccp前端页面

**成功信号**:
- FastAPI服务启动并响应/health和/api/predict
- Vue SFC页面生成，遵循qccp设计规范
- 前后端可本地联调

**技能路由**:
- UI设计: qccp-ui
- 前端页面: qccp-frontend
- 后端服务: qccp-service (FastAPI)

**产出物**: backend/main.py, frontend Vue SFC, INTEGRATE.md

---

## Stage 4: 验证与交付

**目标**: 交叉验证报告一致性，生成交付文档

**成功信号**:
- verification_report.md 指标与报告一致
- README.md 和 INTEGRATE.md 完整
- 无模拟器结果伪装为硬件结果

**产出物**: verification_report.md, README.md, INTEGRATE.md
