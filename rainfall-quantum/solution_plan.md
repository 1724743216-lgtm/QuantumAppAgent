# 短时降雨量量子混合预测 - 解决方案计划

## 总体架构

```
历史气象序列 → [数据预处理] → [经典LSTM编码器] → [投影层] → [VQC量子特征提取] → [经典预测头] → 未来1h降雨量
                      ↘ [纯经典LSTM基线] ↗
```

## Stage 1: 需求范围与经典基线

**目标**: 准备数据集，建立经典LSTM基线

**成功信号**:
- 数据集可加载、可复现分割
- 经典LSTM基线运行成功，RMSE可记录
- `baseline_report.json` 产出

**执行步骤**:
1. 下载/准备CIKM 2017降雨量数据集（或合成数据验证流程）
2. 数据预处理：归一化、序列构建、训练/验证/测试分割
3. 实现经典LSTM基线（PyTorch）
4. 训练并评估，产出 `baseline_report.json`

**预期产出**:
- `/rainfall-quantum/data/` - 数据文件
- `/rainfall-quantum/models/classical_lstm.py` - 基线模型
- `/rainfall-quantum/experiments/stage1_scope_baseline/baseline_report.json`

## Stage 2: 量子混合模型实现

**目标**: VQC量子特征提取 + LSTM混合模型，对比经典基线

**成功信号**:
- VQC量子层可运行（cqlib模拟器）
- 混合模型训练收敛
- `quantum_report.json` 与 `baseline_report.json` 可公平对比

**执行步骤**:
1. 实现 `CqlibExpectationLayer` 量子特征提取层（基于cqlib-hybrid模式）
2. 构建混合模型：LSTM编码器 → 投影 → VQC → 经典预测头
3. 训练并评估，记录量子层参数、运行时间
4. 产出 `quantum_report.json`

**预期产出**:
- `/rainfall-quantum/models/hybrid_vqc_lstm.py` - 混合模型
- `/rainfall-quantum/experiments/stage2_quantum_method/quantum_report.json`

## Stage 3: Web应用打包

**目标**: FastAPI后端 + 前端展示页面

**成功信号**:
- FastAPI服务可启动，提供 `/api/predict`、`/api/train`、`/api/info`、`/health` 接口
- 前端页面可展示预测结果对比、训练曲线、指标

**执行步骤**:
1. FastAPI后端：加载模型、预测接口、训练接口、健康检查
2. 前端页面：数据展示、模型选择、预测可视化、指标对比
3. 集成测试

**预期产出**:
- `/rainfall-quantum/backend/` - FastAPI服务
- `/rainfall-quantum/frontend/` - 前端页面

## Stage 4: 验证与交付

**目标**: 验证一致性，编写交付文档

**成功信号**:
- 验证报告确认基线与量子结果一致
- README、INTEGRATE文档完整
- 应用可本地运行演示

**预期产出**:
- `/rainfall-quantum/experiments/stage4_verification_handoff/verification_report.md`
- `/rainfall-quantum/README.md`
- `/rainfall-quantum/INTEGRATE.md`
