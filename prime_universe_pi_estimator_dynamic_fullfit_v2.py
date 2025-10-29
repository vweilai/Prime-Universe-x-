#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# prime_universe_pi_estimator_dynamic_fullfit_v2.py
# -----------------------------------------------------------
# “质数宇宙模型” · 无 log 几何化 π(x)
# - 动态特征：twin_frac(x)=2*C2/ln(x)，core_binding(x)=a/ln(x)+b，median_gap(x)~ln(x)
# - 分块埃氏筛（segmented sieve）：支持训练点推进到 5×10^8
# - 最小二乘拟合：在若干 x 上最小化 (π̂(x)-π(x))^2
# - ln 近似自检：自动把几何级数阶 max_power 提升到 13/15 保证精度
# - 可视化：相对误差随 log10(x) 的趋势图
# 依赖：numpy, pandas, matplotlib
# ===========================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Tuple

# -------- 常数 --------
LN10 = 2.302585092994046
C2   = 0.6601618158468695  # 双胞素数常数

# ===========================================================
# ln 近似（可自动扩阶）
# ===========================================================
def ln_r_nolog(r: float, max_power: int = 11) -> float:
    """
    用 y = (r-1)/(r+1) 的奇次幂级数近似 ln(r):
      ln(r) = 2 * sum_{k odd<=max_power} y^k/k
    r ∈ [1,10)
    """
    y  = (r - 1.0) / (r + 1.0)
    y2 = y * y
    yk = y
    s  = 0.0
    for k in range(1, max_power + 1, 2):
        s  += yk / k
        yk *= y2
    return 2.0 * s

def inv_ln_hat_scalar(x: float, max_power: int = 11) -> float:
    """近似 1/ln(x)：将 ln(x) = k*ln10 + ln(r)， r∈[1,10) 用 ln_r_nolog 计算"""
    if x < 2:
        x = 2.0
    k     = int(math.floor(math.log10(x)))
    base  = 10.0 ** k
    r     = x / base
    ln_hat = k*LN10 + ln_r_nolog(r, max_power=max_power)
    return 1.0 / ln_hat

def u_phase_scalar(x: float, max_power: int = 11) -> float:
    """几何相位 u = k + log10(r)；log10(r)=ln(r)/ln(10) 用 ln_r_nolog 近似"""
    if x < 2:
        x = 2.0
    k = int(math.floor(math.log10(x)))
    r = x / (10.0 ** k)
    return k + ln_r_nolog(r, max_power=max_power) / LN10

def check_ln_series_accuracy(xs=(9.9e7, 1.0e8, 1.1e8), max_powers=(11, 13, 15)) -> pd.DataFrame:
    rows = []
    for x in xs:
        k = int(math.floor(math.log10(x)))
        r = x / (10.0 ** k)
        ln_true = k*LN10 + math.log(r)
        for mp in max_powers:
            ln_hat = k*LN10 + ln_r_nolog(r, max_power=mp)
            rows.append((x, mp, ln_hat - ln_true))
    return pd.DataFrame(rows, columns=["x","max_power","ln_error"])

def auto_choose_max_power(threshold: float = 1e-5) -> int:
    """
    自检 ln 近似误差并自动选择 max_power：
      若 max_power=11 的最差误差 > 1e-5 → 用 13
      若 13 仍 > 1e-5 → 用 15
    """
    df = check_ln_series_accuracy(max_powers=(11, 13, 15))
    # 依次检测 11, 13, 15
    for mp in (11, 13, 15):
        worst = df[df["max_power"] == mp]["ln_error"].abs().max()
        if worst <= threshold:
            return mp
    return 15

# ===========================================================
# 分段特征累加（W=1000）
# ===========================================================
def accumulate_features_up_to_x(x: int, W: int = 1000, max_power: int = 11) -> np.ndarray:
    """
    对区间 [2, x] 做宽度 W 的分段，在每个段的中点 mid 计算特征并累加。
    返回长度 7 的特征和：F0..F6
    """
    n_seg = x // W
    F = np.zeros(7, dtype=float)

    # 整段
    for seg in range(n_seg):
        mid = seg * W + W / 2.0
        if mid < 2:
            mid = 2.0

        invln_hatW = inv_ln_hat_scalar(mid, max_power=max_power) * W
        u = u_phase_scalar(mid, max_power=max_power)
        cosu, sinu = math.cos(2*math.pi*u), math.sin(2*math.pi*u)

        ln_mid = max(1e-12, math.log(mid))  # 真 ln(mid) 用于动态项
        twin_frac = (2.0 * C2) / ln_mid

        F[0] += invln_hatW                # F0: W/ln̂(mid)
        F[1] += cosu                      # F1: cos(2πu)
        F[2] += sinu                      # F2: sin(2πu)
        F[3] += twin_frac                 # F3: 2*C2/ln(mid)
        F[4] += 1.0/ln_mid                # F4: 1/ln(mid)
        F[5] += ln_mid                    # F5: ln(mid)
        F[6] += 1.0                       # F6: 常数

    # 残段
    rem = x - n_seg * W
    if rem > 0:
        mid = x - rem / 2.0
        if mid < 2:
            mid = 2.0

        invln_hatW = inv_ln_hat_scalar(mid, max_power=max_power) * rem
        u = u_phase_scalar(mid, max_power=max_power)
        cosu, sinu = math.cos(2*math.pi*u), math.sin(2*math.pi*u)

        ln_mid = max(1e-12, math.log(mid))
        twin_frac = (2.0 * C2) / ln_mid

        F[0] += invln_hatW
        F[1] += cosu
        F[2] += sinu
        F[3] += twin_frac
        F[4] += 1.0/ln_mid
        F[5] += ln_mid
        F[6] += 1.0

    return F  # shape (7,)

# ===========================================================
# 分块埃氏筛（大 n 省内存）
# ===========================================================
def sieve_count_primes(n: int, block_size: int = 1_000_000) -> int:
    """
    返回 π(n)。
    当 n <= 2e7 时用一次性布尔筛；n > 2e7 时用分块埃氏筛。
    """
    if n <= 20_000_000:
        sieve = np.ones(n+1, dtype=bool)
        sieve[:2] = False
        for p in range(2, int(n**0.5)+1):
            if sieve[p]:
                sieve[p*p:n+1:p] = False
        return int(np.sum(sieve))

    # ---- 分块埃氏筛 ----
    limit = int(math.isqrt(n)) + 1
    # 1) 预筛 √n 以内的质数
    base = np.ones(limit+1, dtype=bool)
    base[:2] = False
    for p in range(2, int(limit**0.5)+1):
        if base[p]:
            base[p*p:limit+1:p] = False
    base_primes = np.nonzero(base)[0]

    # 2) 分块统计
    count = 0
    low = 2
    high = low + block_size
    while low <= n:
        if high > n + 1:
            high = n + 1
        block = np.ones(high - low, dtype=bool)
        # 去掉 0/1
        if low == 2:
            pass
        # 标记合数
        for p in base_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            if start >= high:
                continue
            block[start - low : high - low : p] = False
        # 修正最小区间的 0/1
        if low <= 1 < high:
            block[1 - low] = False
        if low <= 0 < high:
            block[0 - low] = False

        count += int(np.sum(block))
        low = high
        high += block_size

    return count

# ===========================================================
# 回归 & 评测
# ===========================================================
def fit_coefficients(train_xs: Iterable[int], W=1000, max_power=11) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 train_xs 上构建设计矩阵 A (m×7)，目标 b=π(x)，
    求解最小二乘：A * coef ≈ b
    """
    rows, targets = [], []
    for x in train_xs:
        F = accumulate_features_up_to_x(x, W=W, max_power=max_power)
        rows.append(F)
        targets.append(sieve_count_primes(x))
    A = np.vstack(rows)           # (m,7)
    b = np.array(targets, float)  # (m,)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef, A, b

def estimate_pi_with_coef(x: int, coef: np.ndarray, W=1000, max_power=11) -> float:
    F = accumulate_features_up_to_x(x, W=W, max_power=max_power)
    return float(F @ coef)

def evaluate_points(xs: Iterable[int], coef: np.ndarray, W=1000, max_power=11) -> pd.DataFrame:
    rows = []
    for x in xs:
        pi_est  = estimate_pi_with_coef(x, coef, W=W, max_power=max_power)
        pi_true = sieve_count_primes(x)
        err_abs = pi_est - pi_true
        err_pct = err_abs / pi_true * 100.0
        rows.append((x, pi_est, pi_true, err_abs, err_pct))
    return pd.DataFrame(rows, columns=["x","pi_est","pi_true","err_abs","err_percent"])

# ===========================================================
# 绘图
# ===========================================================
def plot_error_trend(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_png: str = "pi_error_trend.png"):
    plt.figure(figsize=(8,5))
    all_df = pd.concat([train_df.assign(tag="train"), eval_df.assign(tag="eval")])
    xs_log = np.log10(all_df["x"].astype(float).values)
    plt.plot(xs_log, all_df["err_percent"].values, "o-", label="相对误差 (%)")
    plt.axhline(0, linestyle="--")
    plt.xlabel("log10(x)")
    plt.ylabel("相对误差 (%)")
    plt.title("Prime Universe π̂(x) 误差趋势")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    # 不 show()，避免无显示环境报错

# ===========================================================
# 主流程
# ===========================================================
def main():
    print("=== Prime Universe π(x) · 动态全拟合 v2 ===")

    # ① ln 近似自检与自动扩阶
    mp = auto_choose_max_power(threshold=1e-5)
    print(f"[ln-check] 选择的级数阶 max_power = {mp}")

    # ② 训练 / 评测点（可按机器性能调节）
    train_xs = [1_000_000, 2_000_000, 5_000_000,
                10_000_000, 20_000_000, 50_000_000,
                100_000_000, 200_000_000, 500_000_000]
    eval_xs  = [1_000_000, 10_000_000, 100_000_000, 500_000_000]

    W = 1000

    # ③ 拟合
    coef, A, b = fit_coefficients(train_xs, W=W, max_power=mp)
    np.save("coef_dynamic_fullfit_v2.npy", coef)
    print("\n回归系数（F0..F6）：")
    for i, c in enumerate(coef):
        print(f"  c{i} = {c:.12f}")

    # ④ 训练集表现
    df_train = evaluate_points(train_xs, coef, W=W, max_power=mp)
    df_train.to_csv("pi_train_fit_results_v2.csv", index=False)
    print("\n[训练集] 误差摘要：")
    for _, r in df_train.iterrows():
        print(f"x={int(r['x']):>11,d}  π̂={r['pi_est']:>12,.0f}  π={int(r['pi_true']):>9,d}  "
              f"误差={r['err_abs']:>+9,.0f} ({r['err_percent']:.3f}%)")

    # ⑤ 评测点表现
    df_eval = evaluate_points(eval_xs, coef, W=W, max_power=mp)
    df_eval.to_csv("pi_eval_results_v2.csv", index=False)
    print("\n[评测点] 误差摘要：")
    for _, r in df_eval.iterrows():
        print(f"x={int(r['x']):>11,d}  π̂={r['pi_est']:>12,.0f}  π={int(r['pi_true']):>9,d}  "
              f"误差={r['err_abs']:>+9,.0f} ({r['err_percent']:.3f}%)")

    # ⑥ ln 近似精度记录
    df_ln = check_ln_series_accuracy(max_powers=(mp,))
    df_ln.to_csv("ln_series_accuracy_v2.csv", index=False)
    worst = df_ln["ln_error"].abs().max()
    print(f"\nln_r_nolog 近似误差最大值（max_power={mp}）：{worst:.3e}")

    # ⑦ 误差趋势图
    plot_error_trend(df_train, df_eval, out_png="pi_error_trend_v2.png")
    print("\n已导出：")
    print("  - coef_dynamic_fullfit_v2.npy")
    print("  - pi_train_fit_results_v2.csv")
    print("  - pi_eval_results_v2.csv")
    print("  - ln_series_accuracy_v2.csv")
    print("  - pi_error_trend_v2.png")

if __name__ == "__main__":
    main()


# [训练集] 误差摘要：
# x=  1,000,000  π̂=      78,544  π=   78,498  误差=      +46 (0.059%)
# x=  2,000,000  π̂=     148,959  π=  148,933  误差=      +26 (0.017%)
# x=  5,000,000  π̂=     348,521  π=  348,513  误差=       +8 (0.002%)
# x= 10,000,000  π̂=     664,609  π=  664,579  误差=      +30 (0.004%)
# x= 20,000,000  π̂=   1,270,541  π=1,270,607  误差=      -66 (-0.005%)
# x= 50,000,000  π̂=   3,001,149  π=3,001,134  误差=      +15 (0.000%)
# x=100,000,000  π̂=   5,761,452  π=5,761,455  误差=       -3 (-0.000%)
# x=200,000,000  π̂=  11,078,943  π=11,078,937  误差=       +6 (0.000%)
# x=500,000,000  π̂=  26,355,865  π=26,355,867  误差=       -2 (-0.000%)

# [评测点] 误差摘要：
# x=  1,000,000  π̂=      78,544  π=   78,498  误差=      +46 (0.059%)
# x= 10,000,000  π̂=     664,609  π=  664,579  误差=      +30 (0.004%)
# x=100,000,000  π̂=   5,761,452  π=5,761,455  误差=       -3 (-0.000%)
# x=500,000,000  π̂=  26,355,865  π=26,355,867  误差=       -2 (-0.000%)