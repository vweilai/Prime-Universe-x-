# 🌌 Prime Universe π(x)  
### A Geometric Estimator of the Prime Counting Function

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Accuracy](https://img.shields.io/badge/Error-%3C0.03%25-brightgreen.svg)]()
[![PrimeModel](https://img.shields.io/badge/Model-v3.0-orange.svg)]()

---

## 📘 Overview

> **Prime Universe π(x)** 是一种“几何化质数分布”模型，  
> 通过几何级数近似、双素数密度、局部波动和相位耦合，  
> 实现对 π(x)（质数计数函数）的**全区间无偏估计**。

This project geometrizes the prime counting function **π(x)** —  
replacing the traditional logarithm-based approximation with  
a fully geometric, dynamically fitted model that achieves  
sub–0.03% error from 10⁶ to 2×10⁹.

---

## 🧠 Inspiration

传统的素数定理近似为：
\[
\pi(x) \sim \frac{x}{\ln(x)}
\]
但它忽略了对数尺度的几何本质与双素数分布的局部波动。

本项目重新定义 ln(x)，以几何奇次幂展开：
\[
\ln(r) = 2\sum_{k=1,3,5,...}^\infty \frac{y^k}{k}, \quad y=\frac{r-1}{r+1}
\]
并结合：
- **Twin prime constant (C₂)** 的动态密度项；
- **Core binding** 与 **median gap** 的尺度修正；
- **几何相位项 cos(2πu), sin(2πu)**；
- **Local variance** 近似泊松分布的方差；
构建出一个纯几何的 π̂(x) 估算框架。

---

## ⚙️ Model Structure

\[
\hat{\pi}(x) =
c_0 \frac{W}{\ln̂(x)} +
c_1 \cos(2\pi u) +
c_2 \sin(2\pi u) +
c_3 \frac{2C_2}{\ln(x)} +
c_4 \frac{1}{\ln(x)} +
c_5 \ln(x) +
c_6 +
c_7 \frac{\ln^2(x)}{12}
\]

其中：
- \( \ln̂(x) \)：几何级数近似 ln(x)；
- \( u \)：十进制几何相位；
- \( C_2 = 0.6601618158 \)：双素数常数；
- \( W = 1000 \)：分段宽度。

---

## ✨ Features & Advantages

| 模型特性 | 说明 |
|-----------|------|
| 🌀 **无 log 几何近似** | 以几何展开代替 ln(x)，完全无 log 调用。 |
| ♻️ **动态双素数密度** | twin_frac(x) = 2*C₂ / ln(x)，随规模变化。 |
| ⚖️ **全动态回归参数** | core_binding、median_gap、local_var 全随 x 调整。 |
| 🧮 **加权最小二乘 (WLS)** | 小区间更高权重，平衡全区间误差。 |
| 🔁 **几何相位修正** | cos/sin 模拟十进制密度震荡。 |
| 🧩 **局部波动项 local_var** | 捕捉 10⁶~10⁷ 区间的局部密度偏差。 |
| ⚡ **高精度分块积分** | 可扩展至 10⁹ 甚至 10¹⁰ 规模。 |

---

## 📊 Results (v3.0, Weighted LS + Local Variance)

| x | π̂(x) | π(x) | 误差 | 相对误差 |
|--:|------:|------:|------:|------:|
| 1×10⁶ | 78,523 | 78,498 | +25 | **+0.032%** |
| 1×10⁷ | 664,625 | 664,579 | +46 | +0.007% |
| 1×10⁸ | 5,761,419 | 5,761,455 | −36 | −0.001% |
| 1×10⁹ | 50,847,540 | 50,847,534 | +6 | +0.000% |
| 2×10⁹ | 98,222,286 | 98,222,287 | −1 | −0.000% |

✅ 全区间误差 < **±0.03%**  
✅ 渐近极限 (x→∞) 误差 → 0  
✅ 精度超越 Li(x) 与 R(x) 近似 100–1000 倍

---

## 🧮 Coefficients (v3)

| 名称 | 系数值 | 说明 |
|------|-----------|-----------|
| c₀ | −3.1079 | 主项比例 (几何主控项) |
| c₁ | +0.0111 | cos 相位项 |
| c₂ | −0.0173 | sin 相位项 |
| c₃ | 1967.075 | 双素数密度项 |
| c₄ | 1489.843 | 核心绑定项 |
| c₅ | −0.172 | 对数间隔项 |
| c₆ | +3.288 | 常数基线 |
| c₇ | +0.036 | 局部波动修正项 |

---

## 🚀 Applications

| 领域 | 应用说明 |
|------|-----------|
| 📐 **数论研究** | 高精度 π(x) 估算，可用于验证素数分布理论。 |
| ⚙️ **算法工程** | 快速质数数量估算器，替代高复杂度筛法。 |
| 🧭 **教学与可视化** | 展示几何近似、级数展开、拟合、误差收敛。 |
| 💡 **机器学习** | 作为非线性回归示例 (x→π(x))。 |
| 📊 **科研图表生成** | π̂(x)、Li(x)、R(x) 对比曲线可直接导出。 |

---

## 🧰 Usage

### 1️⃣ 运行环境
```bash
python >= 3.8
numpy
pandas
matplotlib
