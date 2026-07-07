"""
合成降雨量数据生成器

基于 Z-R 关系 (Z = 200 * R^1.6) + 噪声 + 季节性模式
生成7个气象特征的时序数据，用于短时降雨量预测任务。

特征列表:
  0. radar_reflectivity (dBZ)
  1. temperature (°C)
  2. humidity (%)
  3. wind_speed (m/s)
  4. pressure (hPa)
  5. wind_direction (°)
  6. past_precipitation (mm)

输出:
  - input_seq: (N, seq_len=10, 7)  归一化后的输入特征
  - target_seq: (N, pred_len=10)    归一化后的未来降雨量
  - 保存到 rainfall_data.npz
"""

import numpy as np
import os


def z_to_r(Z_dBZ):
    """Z-R 关系: Z = 200 * R^1.6  =>  R = (Z/200)^(1/1.6)"""
    Z_linear = 10.0 ** (Z_dBZ / 10.0)
    R = (Z_linear / 200.0) ** (1.0 / 1.6)
    return R


def r_to_z(R):
    """R -> Z (dBZ): Z = 200 * R^1.6, 转换为 dBZ"""
    Z_linear = 200.0 * (np.maximum(R, 0.0) ** 1.6)
    Z_dBZ = 10.0 * np.log10(np.maximum(Z_linear, 1e-10))
    return Z_dBZ


def generate_synthetic_rainfall(
    n_samples=7000,
    seq_len=10,
    pred_len=10,
    seed=42,
    save_path=None,
):
    """
    生成合成降雨量时序数据。

    Parameters
    ----------
    n_samples : int
        总样本数 (train+val+test)
    seq_len : int
        输入序列长度 (步数, 每步6分钟)
    pred_len : int
        预测序列长度
    seed : int
        随机种子
    save_path : str or None
        保存路径, None 则使用默认路径

    Returns
    -------
    dict : 包含归一化后的数据和 scaler 参数
    """
    rng = np.random.RandomState(seed)
    total_len = seq_len + pred_len  # 每个样本需要的历史+未来总长度

    # ---- 1. 生成原始时序 ----
    # 我们生成一条长时序，然后滑动窗口切割
    # 为了保证 7000 个不重叠样本，需要 total_len * n_samples 步
    # 但为了多样性，生成更长序列并允许部分重叠
    long_len = total_len * n_samples + 1000  # 多生成一些保证足够

    t = np.arange(long_len, dtype=np.float64)

    # 季节性 (日周期 ≈ 240 步 = 24h, 每步6min)
    day_period = 240.0
    # 天气系统周期 (更长周期)
    synoptic_period = 2400.0

    # ---- 1a. 降雨量 R (mm/6min) ----
    # 基础: 多尺度正弦 + 降雨事件(脉冲)
    base_rain = 0.5 * (1 + np.sin(2 * np.pi * t / day_period))  # 日变化 0~1
    synoptic = 0.3 * (1 + np.sin(2 * np.pi * t / synoptic_period))  # 天气尺度 0~0.6
    # 随机脉冲 (降雨事件)
    pulse_mask = rng.rand(long_len) < 0.05  # 5% 的时间步有降雨事件
    pulse_intensity = rng.exponential(2.0, size=long_len) * pulse_mask
    # 平滑脉冲 (降雨不会瞬间停止)
    from scipy.ndimage import gaussian_filter1d
    pulse_smooth = gaussian_filter1d(pulse_intensity, sigma=3)

    R = np.clip(base_rain * synoptic + pulse_smooth + 0.1 * rng.randn(long_len), 0, None)

    # ---- 1b. 雷达反射率 (由 Z-R 关系 + 噪声) ----
    Z_dBZ = r_to_z(R) + rng.randn(long_len) * 2.0  # 加测量噪声
    Z_dBZ = np.clip(Z_dBZ, 5, 60)

    # ---- 1c. 温度 (°C): 与降雨负相关 + 日周期 ----
    temp_base = 25.0 + 8.0 * np.sin(2 * np.pi * t / day_period - np.pi / 2)
    temp = temp_base - 2.0 * R / (R.max() + 1e-8) * 5.0 + rng.randn(long_len) * 0.5

    # ---- 1d. 湿度 (%): 与降雨正相关 ----
    humid_base = 60.0 + 20.0 * np.sin(2 * np.pi * t / day_period)
    humid = humid_base + R / (R.max() + 1e-8) * 15.0 + rng.randn(long_len) * 2.0
    humid = np.clip(humid, 30, 100)

    # ---- 1e. 风速 (m/s): 降雨时略增 ----
    wind_base = 3.0 + 1.5 * np.sin(2 * np.pi * t / synoptic_period)
    wind_speed = wind_base + R / (R.max() + 1e-8) * 3.0 + rng.randn(long_len) * 0.3
    wind_speed = np.clip(wind_speed, 0, 20)

    # ---- 1f. 气压 (hPa): 降雨系统低压 ----
    pressure = 1013.0 - 5.0 * R / (R.max() + 1e-8) + 2.0 * np.sin(2 * np.pi * t / synoptic_period) + rng.randn(long_len) * 0.5

    # ---- 1g. 风向 (°): 周期性变化 ----
    wind_dir = (180 + 90 * np.sin(2 * np.pi * t / synoptic_period) + rng.randn(long_len) * 15) % 360

    # ---- 1h. 过去降雨量: 就是 R 的滞后 ----
    past_precip = R  # 当前步的 past_precipitation = 当前降雨

    # ---- 2. 组装特征矩阵 ----
    # shape: (long_len, 7)
    features = np.stack([Z_dBZ, temp, humid, wind_speed, pressure, wind_dir, past_precip], axis=1)

    # ---- 3. 滑动窗口切割样本 ----
    inputs = []
    targets = []
    step = total_len  # 不重叠步长
    for i in range(0, long_len - total_len + 1, step):
        if len(inputs) >= n_samples:
            break
        inp = features[i : i + seq_len]  # (seq_len, 7)
        tgt = R[i + seq_len : i + seq_len + pred_len]  # (pred_len,)
        inputs.append(inp)
        targets.append(tgt)

    inputs = np.array(inputs, dtype=np.float32)  # (N, seq_len, 7)
    targets = np.array(targets, dtype=np.float32)  # (N, pred_len)

    n_actual = inputs.shape[0]
    print(f"[数据生成] 成功生成 {n_actual} 个样本 (请求 {n_samples})")

    # ---- 4. 划分 train/val/test ----
    n_train = 5000
    n_val = 1000
    n_test = min(1000, n_actual - n_train - n_val)

    train_inputs = inputs[:n_train]
    train_targets = targets[:n_train]
    val_inputs = inputs[n_train : n_train + n_val]
    val_targets = targets[n_train : n_train + n_val]
    test_inputs = inputs[n_train + n_val : n_train + n_val + n_test]
    test_targets = targets[n_train + n_val : n_train + n_val + n_test]

    # ---- 5. 归一化 (用训练集统计量) ----
    # 输入特征: 每个特征独立 min-max 归一化
    feat_min = train_inputs.reshape(-1, 7).min(axis=0)  # (7,)
    feat_max = train_inputs.reshape(-1, 7).max(axis=0)  # (7,)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0  # 防止除零

    def normalize_features(x):
        return (x - feat_min) / feat_range

    # 目标: min-max 归一化
    tgt_min = train_targets.min()
    tgt_max = train_targets.max()
    tgt_range = tgt_max - tgt_min
    if tgt_range == 0:
        tgt_range = 1.0

    def normalize_targets(y):
        return (y - tgt_min) / tgt_range

    train_inputs_norm = normalize_features(train_inputs)
    val_inputs_norm = normalize_features(val_inputs)
    test_inputs_norm = normalize_features(test_inputs)

    train_targets_norm = normalize_targets(train_targets)
    val_targets_norm = normalize_targets(val_targets)
    test_targets_norm = normalize_targets(test_targets)

    scaler = {
        "feat_min": feat_min,
        "feat_max": feat_max,
        "feat_range": feat_range,
        "tgt_min": float(tgt_min),
        "tgt_max": float(tgt_max),
        "tgt_range": float(tgt_range),
    }

    # ---- 6. 保存 ----
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "rainfall_data.npz")

    np.savez(
        save_path,
        train_inputs=train_inputs_norm,
        train_targets=train_targets_norm,
        val_inputs=val_inputs_norm,
        val_targets=val_targets_norm,
        test_inputs=test_inputs_norm,
        test_targets=test_targets_norm,
        # 原始 (未归一化) 测试目标，用于还原评估
        test_targets_raw=test_targets,
        # scaler 参数
        feat_min=feat_min,
        feat_max=feat_max,
        feat_range=feat_range,
        tgt_min=float(tgt_min),
        tgt_max=float(tgt_max),
        tgt_range=float(tgt_range),
    )
    print(f"[数据保存] 保存到 {save_path}")
    print(f"  训练集: {train_inputs_norm.shape}, 验证集: {val_inputs_norm.shape}, 测试集: {test_inputs_norm.shape}")
    print(f"  目标范围 (原始): [{tgt_min:.3f}, {tgt_max:.3f}]")

    return {
        "train_inputs": train_inputs_norm,
        "train_targets": train_targets_norm,
        "val_inputs": val_inputs_norm,
        "val_targets": val_targets_norm,
        "test_inputs": test_inputs_norm,
        "test_targets": test_targets_norm,
        "test_targets_raw": test_targets,
        "scaler": scaler,
    }


if __name__ == "__main__":
    generate_synthetic_rainfall()
