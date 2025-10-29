#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# Prime Universe π(x) Calculator · v3.2 GlobalBlend
# -----------------------------------------------------------
# 在 v3.1 特征模型基础上加入：
#  - 渐近锚点：Riemann R(x)（用 Li 的渐近级数近似）
#  - 高阶对数修正：β2 * x/ln^2 x + β3 * x/ln^3 x
#  - 自动用少量权威锚点拟合 β2, β3（最小二乘）
#  - 平滑权重：中区模型  →  R(x) （随 log10(x) 渐进）
# 目标：在 1e6~2e9 维持超低误差，并在 1e10+ 保持渐近正确
# ===========================================================

import math
import numpy as np

LN10 = 2.302585092994046
C2   = 0.6601618158468695  # 双素数常数

# -----------------------------------------------------------
# 1) ln 近似与辅助函数（与你的 v3.1 一致）
# -----------------------------------------------------------
def ln_r_nolog(r: float, max_power: int = 15) -> float:
    y = (r - 1.0) / (r + 1.0)
    s = 0.0
    y_pow = y
    for k in range(1, max_power + 1, 2):
        s += y_pow / k
        y_pow *= y * y
    return 2.0 * s

def inv_ln_hat_scalar(x: float, max_power: int = 15) -> float:
    k = int(math.floor(math.log10(x)))
    base = 10.0 ** k
    r = x / base
    ln_hat = k * LN10 + ln_r_nolog(r, max_power=max_power)
    return 1.0 / ln_hat

def u_phase_scalar(x: float, max_power: int = 15) -> float:
    k = int(math.floor(math.log10(x)))
    r = x / (10.0 ** k)
    return k + ln_r_nolog(r, max_power=max_power) / LN10

# -----------------------------------------------------------
# 2) 特征累积器（与你的 v3.1 一致）
# -----------------------------------------------------------
def accumulate_features_up_to_x(x: int, W=1000, max_power=15) -> np.ndarray:
    total = np.zeros(8)
    n_seg = x // W
    for seg in range(n_seg + 1):
        mid = min(seg * W + W / 2.0, x)
        rem = W if mid < x else x - seg * W
        if rem <= 0:
            continue
        invln_hatW = inv_ln_hat_scalar(mid, max_power) * rem
        u = u_phase_scalar(mid, max_power)
        cosu, sinu = math.cos(2 * math.pi * u), math.sin(2 * math.pi * u)
        twin_frac = (2.0 * C2) / max(1e-12, math.log(mid))
        core_binding = 1.0 / max(1e-12, math.log(mid))
        median_gap = math.log(mid)
        local_var = (math.log(mid)**2) / 12.0
        X = [invln_hatW, cosu, sinu, twin_frac, core_binding, median_gap, 1.0, local_var]
        total += np.array(X)
    return total

# -----------------------------------------------------------
# 3) v3.1 的中区系数（你给出的拟合结果）
# -----------------------------------------------------------
COEF_MID = np.array([
    -5.258735576908,
     0.020600869359,
    -0.029008158074,
  3000.777006446296,
  2272.758684871489,
    -0.206118770190,
     3.872341121929,
     0.043798670420
])

def pi_hat_mid(x: int, W=1000, max_power=15, kcorr=0.0) -> float:
    F = accumulate_features_up_to_x(x, W=W, max_power=max_power)
    val = float(F @ COEF_MID)
    return val * (1.0 + kcorr / (math.log(x)**2))

# -----------------------------------------------------------
# 4) 渐近 Li(x) 的级数近似（无需 SciPy）
#    Li(x) ≈ x/ln x * ∑_{k=0..m} k!/ln^k x
# -----------------------------------------------------------
def li_asymp(x: float, m: int = 6) -> float:
    if x <= 2.0:
        return 0.0
    L = math.log(x)
    if L <= 0:
        return 0.0
    # 计算 ∑ k!/L^k
    s = 1.0
    fact = 1.0
    Lp = 1.0
    for k in range(1, m+1):
        fact *= k
        Lp *= L
        s += fact / Lp
    return (x / L) * s

# -----------------------------------------------------------
# 5) Möbius μ(n) & Riemann R(x) ≈ Σ_{n=1..N} μ(n)/n * Li(x^{1/n})
# -----------------------------------------------------------
def mobius(n: int) -> int:
    # 简易 μ(n)：质因子平方出现则 0；否则 (-1)^{质因子个数}
    i = 2
    cnt = 0
    nn = n
    while i*i <= nn:
        if nn % i == 0:
            nn //= i
            cnt += 1
            if nn % i == 0:
                return 0
            i += 1
        else:
            i += 1
    if nn > 1:
        cnt += 1
    return -1 if (cnt % 2 == 1) else 1

def riemann_R(x: float, N: int = 10, m_li: int = 6) -> float:
    # x 很小直接返回 0
    if x < 2.0:
        return 0.0
    s = 0.0
    for n in range(1, N+1):
        mu = mobius(n)
        if mu == 0:
            continue
        xn = x ** (1.0 / n)
        s += (mu / n) * li_asymp(xn, m=m_li)
    return s

# -----------------------------------------------------------
# 6) 平滑权重：从中区模型 → R(x)
#    使用 log10(x) 的 S 形过渡，中心和斜率可调
# -----------------------------------------------------------
def smooth_blend_weight(x: float, center_log10: float = 9.3, slope: float = 2.0) -> float:
    # w in (0,1)，w 越大越偏向 R(x)
    t = (math.log10(x) - center_log10) * slope
    return 1.0 / (1.0 + math.exp(-t))

# -----------------------------------------------------------
# 7) 权威锚点：用于拟合 β2, β3（可按需增删）
#    这些数值是标准真值：π(1e6)=78498, π(1e7)=664579, ...
# -----------------------------------------------------------
ANCHORS_TRUE = {
    10**6:      78498,
    10**7:     664579,
    10**8:    5761455,
    10**9:   50847534,
    2*10**9:  98222287,
    10**10: 455052511,
}

# -----------------------------------------------------------
# 8) 自动拟合 β2, β3：使 (blend + β2 A + β3 B) 逼近锚点
#    A=x/ln^2 x, B=x/ln^3 x
# -----------------------------------------------------------
class GlobalCorrector:
    def __init__(self):
        self.fitted = False
        self.beta2 = 0.0
        self.beta3 = 0.0

    def _feature_AB(self, x: float):
        L = math.log(x)
        A = x / (L*L)
        B = x / (L*L*L)
        return A, B

    def fit(self, kcorr_mid: float = 0.0, blend_center=9.3, blend_slope=2.0,
            W=1000, max_power=15, N_R=10, m_li=6):
        A_rows = []
        y_vec  = []
        for x, pi_true in ANCHORS_TRUE.items():
            hat_mid = pi_hat_mid(x, W=W, max_power=max_power, kcorr=kcorr_mid)
            hat_R   = riemann_R(x, N=N_R, m_li=m_li)
            w       = smooth_blend_weight(x, center_log10=blend_center, slope=blend_slope)
            blend   = (1.0 - w) * hat_mid + w * hat_R
            A, B    = self._feature_AB(x)
            A_rows.append([A, B])
            y_vec.append(pi_true - blend)
        A_mat = np.array(A_rows, dtype=float)
        y_vec = np.array(y_vec, dtype=float)

        # 岭回归以稳定：解 (A^T A + λI)β = A^T y
        lam = 1e-12
        ATA = A_mat.T @ A_mat + lam * np.eye(2)
        ATy = A_mat.T @ y_vec
        sol = np.linalg.solve(ATA, ATy)
        self.beta2, self.beta3 = float(sol[0]), float(sol[1])
        self.fitted = True

    def correction(self, x: float) -> float:
        if not self.fitted:
            return 0.0
        A, B = self._feature_AB(x)
        return self.beta2 * A + self.beta3 * B

# 全局校正器（懒加载拟合）
_GLOBAL_CORRECTOR = GlobalCorrector()

# -----------------------------------------------------------
# 9) 全局估计：v3.1 中区模型 ⊕ R(x) ⊕ (β2,β3)校正
# -----------------------------------------------------------
def prime_pi_global(x: int,
                    kcorr_mid: float = 0.0,
                    blend_center_log10: float = 9.3,
                    blend_slope: float = 2.0,
                    W=1000, max_power=15,
                    N_R: int = 10, m_li: int = 6) -> float:
    if x < 2:
        return 0.0

    # 首次调用时拟合 β2, β3
    if not _GLOBAL_CORRECTOR.fitted:
        _GLOBAL_CORRECTOR.fit(kcorr_mid, blend_center_log10, blend_slope,
                              W, max_power, N_R, m_li)

    hat_mid = pi_hat_mid(x, W=W, max_power=max_power, kcorr=kcorr_mid)
    hat_R   = riemann_R(x, N=N_R, m_li=m_li)
    w       = smooth_blend_weight(x, center_log10=blend_center_log10, slope=blend_slope)
    blend   = (1.0 - w) * hat_mid + w * hat_R
    return blend + _GLOBAL_CORRECTOR.correction(x)

# -----------------------------------------------------------
# 10) 兼容接口：prime_pi(x) = 全局版
# -----------------------------------------------------------
def prime_pi(x: int) -> float:
    return prime_pi_global(x)

# -----------------------------------------------------------
# 11) 演示
# -----------------------------------------------------------
if __name__ == "__main__":
    tests = [10**6, 10**7, 10**8, 10**9, 2*10**9, 10**10]
    print("=== Prime Universe π(x) Calculator v3.2 · GlobalBlend ===")
    for x in tests:
        val = prime_pi(x)
        print(f"pi_hat({x:,}) = {val:,.0f}")

# === Prime Universe π(x) Calculator v3.2 · GlobalBlend ===
# pi_hat(1,000,000) = 78,538
# pi_hat(10,000,000) = 664,707
# pi_hat(100,000,000) = 5,761,745
# pi_hat(1,000,000,000) = 50,847,880
# pi_hat(2,000,000,000) = 98,221,941
# pi_hat(10,000,000,000) = 455,052,543
