# 研究请求：金融欺诈检测量子算法应用

## 应用目标
构建一个基于量子机器学习（QML）的金融欺诈检测应用，使用变分量子电路（VQC）对信用卡交易数据进行二分类（欺诈/正常），并与经典基线方法进行对比。

## 用户选择
- 金融问题：欺诈检测（Fraud Detection）
- 量子算法路线：QML（推荐路线，VQC二分类）
- 云展示：是（完整前端页面 + FastAPI后端）
- 代码模式：Lite

## 关键约束
- 使用Cqlib SDK构建量子电路
- 使用信用卡欺诈检测数据集（合成数据，模拟Kaggle Credit Card Fraud Detection特征结构）
- 主指标：F1-Score（欺诈类别），辅助指标：AUC-ROC、Precision、Recall
- 本地模拟器执行，不涉及真实硬件
- 前端遵循qccp设计规范，后端使用FastAPI
