# INTEGRATE.md — VQE H2 应用集成说明

## 后端集成

### FastAPI 服务

启动命令:
```bash
cd vqe-app && python backend/app.py
```

默认监听 `0.0.0.0:8000`。

### API 合约

#### GET /api/results
返回预计算的基线和 VQE 结果。

响应示例:
```json
{
  "baseline": {
    "task": "H2 ground-state energy estimation",
    "value": -1.857275,
    "primary_metric": "energy (Hartree)",
    "higher_is_better": false,
    "method": "exact_diagonalization"
  },
  "quantum": {
    "task": "H2 ground-state energy estimation",
    "value": -1.857037,
    "primary_metric": "energy (Hartree)",
    "higher_is_better": false,
    "method": "VQE",
    "iterations": 156,
    "convergence_trace": [...]
  }
}
```

#### GET /api/run?layers=2&max_iter=200&seed=42
运行新的 VQE 优化。

查询参数:
- `layers` (int): Ansatz 层数, 默认 2
- `max_iter` (int): 最大迭代, 默认 200
- `seed` (int): 随机种子, 默认 42

响应示例:
```json
{
  "status": "ok",
  "vqe_energy": -1.857037,
  "exact_energy": -1.857275,
  "error": 0.000238,
  "iterations": 156,
  "convergence_trace": [...]
}
```

#### GET /api/hamiltonian
返回 H2 哈密顿量项列表。

### 前端集成

前端为纯 HTML + ECharts 页面，位于 `frontend/index.html`。

集成到其他前端框架时:
1. 复用 `/api/results` 和 `/api/run` 端点
2. 收敛曲线和哈密顿量柱状图使用 ECharts 渲染
3. 颜色变量定义在 CSS `:root` 中

## cqlib 版本依赖

本应用使用以下 cqlib 接口:
- `cqlib.Circuit` / `cqlib.Parameter` — 参数化量子电路
- `cqlib.simulator.StatevectorSimulator` — 态矢量模拟

建议 cqlib >= 0.3.0。

## 扩展方向

- 增加 shot-based 模拟 (用 `sim.sample(shots=N)` 替代 `sim.measure()`)
- 支持更多分子 (增加 Jordan-Wigner 或 Bravyi-Kitaev 变换)
- 集成 TianYan 平台提交真实硬件任务
- 添加更多 ansatz 类型 (UCCSD, ADAPT-VQE)
