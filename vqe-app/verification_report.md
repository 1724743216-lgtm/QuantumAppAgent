# Verification Report — VQE H2 分子基态能量求解

## 验证日期
2026-07-06

## 验证方法

### 1. VQE 算法验证
- 运行命令: `python vqe_solver.py`
- VQE 优化能量: -1.8570 Hartree
- 精确对角化能量: -1.8573 Hartree
- 绝对误差: 2.38×10⁻⁴ Hartree (< 0.01 Hartree 容差)
- 收敛: 156 次 COBYLA 迭代后收敛

### 2. 经典基线验证
- 精确对角化使用 numpy.linalg.eigvalsh 对 H2 哈密顿量矩阵求解
- 结果与文献值一致

### 3. 后端服务验证
- FastAPI 启动: `python backend/app.py`
- 端点 `/api/results`, `/api/run`, `/api/hamiltonian` 可正常返回 JSON
- 前端页面 `/` 可正常渲染

### 4. 报告一致性
- `baseline_report.json` 和 `quantum_report.json` 共享相同 task 和 primary_metric
- 两者可比 (comparable=True)

## 局限性说明

VQE 是近似变分方法，其能量值高于（更差于）精确对角化结果。这是量子算法的已知特性：
- VQE 提供了近似的基态能量估计
- 误差 2.38×10⁻⁴ Hartree 表明 ansatz 表达能力足够
- 本应用的价值在于展示 VQE 工作流程，而非超越经典方法

## 结论

- [x] VQE 算法运行正常，收敛到精确值附近
- [x] 基线结果正确
- [x] 报告格式符合要求
- [x] 后端 API 可用
- [ ] 量子方法未超越经典基线 (预期行为)
