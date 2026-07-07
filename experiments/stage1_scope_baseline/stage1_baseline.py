"""
Stage 1: 经典基线 - 金融欺诈检测
使用 Logistic Regression 作为经典基线
"""
import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA

SEED = 42
N_SAMPLES = 1000
N_FEATURES = 28
N_INFORMATIVE = 10
FRAUD_RATIO = 0.2

# 1. 生成合成信用卡欺诈检测数据集
print("=== 生成合成信用卡欺诈检测数据集 ===")
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=5,
    n_clusters_per_class=2,
    weights=[1 - FRAUD_RATIO, FRAUD_RATIO],
    random_state=SEED,
)
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"欺诈比例: {y.mean():.3f}")

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# 3. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. PCA降维至4维（与量子方法一致）
pca = PCA(n_components=4, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA降维后: {X_train_pca.shape[1]}维, 解释方差比: {pca.explained_variance_ratio_.sum():.3f}")

# 5. 经典基线: Logistic Regression
print("\n=== 经典基线: Logistic Regression ===")
clf = LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
y_prob = clf.predict_proba(X_test_pca)[:, 1]

# 6. 评估
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

# 7. 保存基线报告
baseline_report = {
    "task": "金融欺诈检测二分类",
    "data": "合成信用卡欺诈检测数据集 (1000 samples, 28 features, PCA→4)",
    "primary_metric": "f1_score",
    "higher_is_better": True,
    "value": round(f1, 4),
    "secondary_metrics": {
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4)
    },
    "method": "Logistic Regression (class_weight=balanced)",
    "preprocessing": "StandardScaler + PCA(n_components=4)",
    "command": "python3 stage1_baseline.py",
    "artifact_paths": [
        "/experiments/stage1_scope_baseline/baseline_report.json"
    ],
    "seed": SEED,
    "backend": "classical",
    "limitations": [
        "合成数据集，非真实交易数据",
        "PCA降维可能丢失部分信息",
        "仅使用Logistic Regression作为基线"
    ]
}

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "baseline_report.json"), "w") as f:
    json.dump(baseline_report, f, indent=2, ensure_ascii=False)

# 保存预处理后的数据供量子方法使用
np.savez(os.path.join(script_dir, "preprocessed_data.npz"),
         X_train_pca=X_train_pca, X_test_pca=X_test_pca,
         y_train=y_train, y_test=y_test,
         scaler_mean=scaler.mean_.tolist(),
         scaler_scale=scaler.scale_.tolist(),
         pca_components=pca.components_.tolist(),
         pca_mean=pca.mean_.tolist())

print("\n基线报告已保存至 baseline_report.json")
print("预处理数据已保存至 preprocessed_data.npz")
