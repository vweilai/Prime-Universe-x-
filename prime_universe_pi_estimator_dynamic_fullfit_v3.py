#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# prime_universe_pi_estimator_dynamic_fullfit_v3.py
# -----------------------------------------------------------
# Prime Universe π(x) — geometric (no direct log calls in main term)
# v3: add local density fluctuation feature:
#      local_var(x) ≈ (ln(x))^2 / 12  (Poisson-like gap variance proxy)
# Also add optional weighted least squares to emphasize small-x accuracy.
# - Segmented sieve up to 2e9+
# - Auto ln-series order selection
# - Plots error trend (ASCII labels to avoid font warnings)
# Dependencies: numpy, pandas, matplotlib
# ===========================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Tuple

# ---------------- Config ----------------
WEIGHT_ALPHA = 0.5     # weight w_i = 1 / x_i^alpha ; set 0.0 to disable weighting
BLOCK_SIZE   = 5_000_000
LN_ERR_THRESH = 1e-5

# ---------------- Constants ----------------
LN10 = 2.302585092994046
C2   = 0.6601618158468695  # twin prime constant

# ===========================================================
# ln approximation via odd-power series in y=(r-1)/(r+1)
# ===========================================================
def ln_r_nolog(r: float, max_power: int = 11) -> float:
    y  = (r - 1.0) / (r + 1.0)
    y2 = y * y
    yk = y
    s  = 0.0
    for k in range(1, max_power + 1, 2):
        s  += yk / k
        yk *= y2
    return 2.0 * s

def inv_ln_hat_scalar(x: float, max_power: int = 11) -> float:
    if x < 2:
        x = 2.0
    k     = int(math.floor(math.log10(x)))
    base  = 10.0 ** k
    r     = x / base
    ln_hat = k*LN10 + ln_r_nolog(r, max_power=max_power)
    return 1.0 / ln_hat

def u_phase_scalar(x: float, max_power: int = 11) -> float:
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

def auto_choose_max_power(threshold: float = LN_ERR_THRESH) -> int:
    df = check_ln_series_accuracy(max_powers=(11, 13, 15))
    for mp in (11, 13, 15):
        worst = df[df["max_power"] == mp]["ln_error"].abs().max()
        if worst <= threshold:
            return mp
    return 15

# ===========================================================
# Feature accumulation up to x with step W
# Features (sum over segments):
#   F0: W / ln_hat(mid)
#   F1: cos(2πu(mid))
#   F2: sin(2πu(mid))
#   F3: 2*C2 / ln(mid)         (twin_frac)
#   F4: 1 / ln(mid)            (core_binding part)
#   F5: ln(mid)                (median_gap part)
#   F6: 1.0                    (bias)
#   F7: (ln(mid)^2)/12         (local density variance proxy)  <-- NEW
# ===========================================================
def accumulate_features_up_to_x(x: int, W: int = 1000, max_power: int = 11) -> np.ndarray:
    n_seg = x // W
    F = np.zeros(8, dtype=float)

    # full segments
    for seg in range(n_seg):
        mid = seg * W + W / 2.0
        if mid < 2:
            mid = 2.0

        invln_hatW = inv_ln_hat_scalar(mid, max_power=max_power) * W
        u = u_phase_scalar(mid, max_power=max_power)
        cosu, sinu = math.cos(2*math.pi*u), math.sin(2*math.pi*u)

        ln_mid = max(1e-12, math.log(mid))  # true ln for dynamic terms
        twin_frac = (2.0 * C2) / ln_mid
        local_var = (ln_mid * ln_mid) / 12.0

        F[0] += invln_hatW
        F[1] += cosu
        F[2] += sinu
        F[3] += twin_frac
        F[4] += 1.0/ln_mid
        F[5] += ln_mid
        F[6] += 1.0
        F[7] += local_var

    # remainder
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
        local_var = (ln_mid * ln_mid) / 12.0

        F[0] += invln_hatW
        F[1] += cosu
        F[2] += sinu
        F[3] += twin_frac
        F[4] += 1.0/ln_mid
        F[5] += ln_mid
        F[6] += 1.0
        F[7] += local_var

    return F  # shape (8,)

# ===========================================================
# Segmented sieve (handles up to 2e9+)
# ===========================================================
def sieve_count_primes(n: int, block_size: int = BLOCK_SIZE) -> int:
    if n <= 20_000_000:
        sieve = np.ones(n+1, dtype=bool)
        sieve[:2] = False
        for p in range(2, int(n**0.5)+1):
            if sieve[p]:
                sieve[p*p:n+1:p] = False
        return int(np.sum(sieve))

    limit = int(math.isqrt(n)) + 1
    base = np.ones(limit+1, dtype=bool)
    base[:2] = False
    for p in range(2, int(limit**0.5)+1):
        if base[p]:
            base[p*p:limit+1:p] = False
    base_primes = np.nonzero(base)[0]

    count = 0
    low = 2
    high = low + block_size
    while low <= n:
        if high > n + 1:
            high = n + 1
        block = np.ones(high - low, dtype=bool)
        for p in base_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            if start < high:
                block[start - low : high - low : p] = False
        if low <= 1 < high:
            block[1 - low] = False
        if low <= 0 < high:
            block[0 - low] = False
        count += int(np.sum(block))
        low = high
        high += block_size
    return count

# ===========================================================
# Regression & evaluation
# ===========================================================
def fit_coefficients(train_xs: Iterable[int], W=1000, max_power=11, weight_alpha: float = WEIGHT_ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, targets, weights = [], [], []
    for x in train_xs:
        F = accumulate_features_up_to_x(x, W=W, max_power=max_power)
        rows.append(F)
        targets.append(sieve_count_primes(x))
        w = (1.0 / (x ** weight_alpha)) if weight_alpha > 0.0 else 1.0
        weights.append(w)

    A = np.vstack(rows)           # (m,8)
    b = np.array(targets, float)  # (m,)
    Wdiag = np.sqrt(np.array(weights, float))[:, None]  # (m,1)
    A_w = A * Wdiag               # weighted design
    b_w = b * Wdiag.ravel()

    coef, *_ = np.linalg.lstsq(A_w, b_w, rcond=None)
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
# Plotting
# ===========================================================
def plot_error_trend(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_png: str = "pi_error_trend_v3.png"):
    plt.figure(figsize=(8,5))
    all_df = pd.concat([train_df.assign(tag="train"), eval_df.assign(tag="eval")])
    xs_log = np.log10(all_df["x"].astype(float).values)
    plt.plot(xs_log, all_df["err_percent"].values, "o-", label="relative error (%)")
    plt.axhline(0, linestyle="--")
    plt.xlabel("log10(x)")
    plt.ylabel("relative error (%)")
    plt.title("Prime Universe π̂(x) error trend (v3)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

# ===========================================================
# Main
# ===========================================================
def main():
    print("=== Prime Universe π(x) · Dynamic Fullfit v3 (local_var + WLS) ===")

    # ln-series auto check
    mp = auto_choose_max_power(threshold=LN_ERR_THRESH)
    print(f"[ln-check] chosen max_power = {mp}")
    print(f"[wls] WEIGHT_ALPHA = {WEIGHT_ALPHA}")

    # training / evaluation sets
    train_xs = [1_000_000, 2_000_000, 5_000_000,
                10_000_000, 20_000_000, 50_000_000,
                100_000_000, 200_000_000, 500_000_000,
                1_000_000_000, 2_000_000_000]

    eval_xs  = [1_000_000, 10_000_000, 100_000_000,
                500_000_000, 1_000_000_000, 2_000_000_000]

    W = 1000

    # fit
    coef, A, b = fit_coefficients(train_xs, W=W, max_power=mp, weight_alpha=WEIGHT_ALPHA)
    np.save("coef_dynamic_fullfit_v3.npy", coef)
    print("\nCoefficients (F0..F7):")
    for i, c in enumerate(coef):
        print(f"  c{i} = {c:.12f}")

    # train report
    df_train = evaluate_points(train_xs, coef, W=W, max_power=mp)
    df_train.to_csv("pi_train_fit_results_v3.csv", index=False)
    print("\n[Train] summary:")
    for _, r in df_train.iterrows():
        print(f"x={int(r['x']):>11,d}  pi_hat={r['pi_est']:>12,.0f}  pi={int(r['pi_true']):>10,d}  "
              f"err={r['err_abs']:>+9,.0f} ({r['err_percent']:.3f}%)")

    # eval report
    df_eval = evaluate_points(eval_xs, coef, W=W, max_power=mp)
    df_eval.to_csv("pi_eval_results_v3.csv", index=False)
    print("\n[Eval] summary:")
    for _, r in df_eval.iterrows():
        print(f"x={int(r['x']):>11,d}  pi_hat={r['pi_est']:>12,.0f}  pi={int(r['pi_true']):>10,d}  "
              f"err={r['err_abs']:>+9,.0f} ({r['err_percent']:.3f}%)")

    # ln approx record
    df_ln = check_ln_series_accuracy(max_powers=(mp,))
    df_ln.to_csv("ln_series_accuracy_v3.csv", index=False)
    worst = df_ln["ln_error"].abs().max()
    print(f"\nln_r_nolog worst abs error (max_power={mp}): {worst:.3e}")

    # plot
    plot_error_trend(df_train, df_eval, out_png="pi_error_trend_v3.png")
    print("\nExported files:")
    print("  - coef_dynamic_fullfit_v3.npy")
    print("  - pi_train_fit_results_v3.csv")
    print("  - pi_eval_results_v3.csv")
    print("  - ln_series_accuracy_v3.csv")
    print("  - pi_error_trend_v3.png")

if __name__ == "__main__":
    main()

# === Prime Universe π(x) · Dynamic Fullfit v3 (local_var + WLS) ===
# [ln-check] chosen max_power = 15
# [wls] WEIGHT_ALPHA = 0.5

# Coefficients (F0..F7):
#   c0 = -3.107877487775
#   c1 = 0.011119287539
#   c2 = -0.017321125170
#   c3 = 1967.075185067525
#   c4 = 1489.843197182902
#   c5 = -0.172023575199
#   c6 = 3.288004690555
#   c7 = 0.036042897995

# [Train] summary:
# x=  1,000,000  pi_hat=      78,523  pi=    78,498  err=      +25 (0.032%)
# x=  2,000,000  pi_hat=     148,924  pi=   148,933  err=       -9 (-0.006%)
# x=  5,000,000  pi_hat=     348,468  pi=   348,513  err=      -45 (-0.013%)
# x= 10,000,000  pi_hat=     664,625  pi=   664,579  err=      +46 (0.007%)
# x= 20,000,000  pi_hat=   1,270,560  pi= 1,270,607  err=      -47 (-0.004%)
# x= 50,000,000  pi_hat=   3,001,191  pi= 3,001,134  err=      +57 (0.002%)
# x=100,000,000  pi_hat=   5,761,419  pi= 5,761,455  err=      -36 (-0.001%)
# x=200,000,000  pi_hat=  11,078,955  pi=11,078,937  err=      +18 (0.000%)
# x=500,000,000  pi_hat=  26,355,853  pi=26,355,867  err=      -14 (-0.000%)
# x=1,000,000,000  pi_hat=  50,847,540  pi=50,847,534  err=       +6 (0.000%)
# x=2,000,000,000  pi_hat=  98,222,286  pi=98,222,287  err=       -1 (-0.000%)

# [Eval] summary:
# x=  1,000,000  pi_hat=      78,523  pi=    78,498  err=      +25 (0.032%)
# x= 10,000,000  pi_hat=     664,625  pi=   664,579  err=      +46 (0.007%)
# x=100,000,000  pi_hat=   5,761,419  pi= 5,761,455  err=      -36 (-0.001%)
# x=500,000,000  pi_hat=  26,355,853  pi=26,355,867  err=      -14 (-0.000%)
# x=1,000,000,000  pi_hat=  50,847,540  pi=50,847,534  err=       +6 (0.000%)
# x=2,000,000,000  pi_hat=  98,222,286  pi=98,222,287  err=       -1 (-0.000%)

# ln_r_nolog worst abs error (max_power=15): 9.489e-03

# Exported files:
#   - coef_dynamic_fullfit_v3.npy
#   - pi_train_fit_results_v3.csv
#   - pi_eval_results_v3.csv
#   - ln_series_accuracy_v3.csv
#   - pi_error_trend_v3.png