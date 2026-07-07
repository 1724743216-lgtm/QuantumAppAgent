# VQE H2 分子基态能量求解

基于 cqlib SDK 的变分量子本征求解器 (VQE) 应用，求解 H2 分子在键距 R=0.735 Å 时的基态能量。

## 概述

本应用使用 VQE 算法在量子电路上优化参数化波函数，逼近 H2 分子的基态能量。经典基线采用精确对角化方法计算相同的哈密顿量，用于对比验证。

## 方法

- **量子方法**: VQE (Variational Quantum Eigensolver)
  - 硬件高效 ansatz (Ry-Rz-CX, 2 层, 8 个参数)
  - COBYLA 优化器
  - cqlib StatevectorSimulator 模拟

- **经典基线**: 精确对角化 (numpy.linalg.eigvalsh)

## 哈密顿量

H2 分子最小基组哈密顿量 (Bravyi-Kitaev 编码, 2 量子比特):

| 项 | 系数 |
|---|---|
| I (恒等) | -1.0524 |
| Z₀ | +0.3979 |
| Z₁ | -0.3979 |
| Z₀Z₁ | -0.0113 |
| X₀X₁ | +0.1809 |

## 结果

| 指标 | 值 |
|---|---|
| 精确对角化能量 | -1.8573 Hartree |
| VQE 优化能量 | -1.8570 Hartree |
| 绝对误差 | 2.38×10⁻⁴ Hartree |
| 优化迭代次数 | 156 |
| 参数数量 | 8 |

## 快速开始

### 安装依赖

```bash
pip install cqlib scipy numpy fastapi uvicorn
```

### 运行 VQE 求解

```bash
cd vqe-app
python vqe_solver.py
```

### 启动 Web 演示

```bash
cd vqe-app
python backend/app.py
```

打开浏览器访问 `http://localhost:8000`

## 项目结构

```
vqe-app/
├── vqe_solver.py              # VQE 核心算法 + 经典基线
├── backend/
│   └── app.py                 # FastAPI 后端服务
├── frontend/
│   └── index.html             # Vue 风格前端页面 (ECharts 可视化)
├── baseline_report.json       # 经典基线报告
├── quantum_report.json        # VQE 量子方法报告
├── application_manifest.json  # 应用清单
├── requirements.json          # 验证需求
├── README.md                  # 本文档
└── INTEGRATE.md               # 集成说明
```

## API 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/results` | GET | 获取预计算的 VQE 和基线结果 |
| `/api/run` | GET | 运行新的 VQE 优化 (参数: layers, max_iter, seed) |
| `/api/hamiltonian` | GET | 获取 H2 哈密顿量项 |
| `/` | GET | 前端页面 |

## 局限性

- 使用态矢量模拟器，无采样噪声
- 仅包含 2 量子比特的小型哈密顿量
- 固定键距 R=0.735 Å
- 未在真实量子硬件上验证
