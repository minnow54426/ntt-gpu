# OpenFHE 的 CKKS 功能自举

**Fourier-extension functional bootstrapping（FE-FBT）把刷新密文和逐槽求值合在一次操作中。** OpenFHE 1.6.1 可以直接对实数 CKKS 槽位近似计算 sigmoid、指数和 GELU 等函数。也可以向相同的接口提供 ReLU 的近似系数。ReLU 在零点不可微，在下面的同区间、近似阶数相近的测试里，硬 ReLU 的误差明显高于 GELU。

本文检查 OpenFHE v1.6.1（提交 `b59e1f291965f14fbee5172cd83a20f5b856204f`），并按同一计时口径比较普通自举与功能自举。相关实现仍标为实验功能。见 [实现文档](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/examples/CKKS_FUNCTIONAL_BOOTSTRAPING.md) 和 [EUROCRYPT 2026 论文](https://eprint.iacr.org/2026/367)。

## 论文与首次发布标签

OpenFHE 把两条路线都称为 CKKS functional bootstrapping。这里的首次实现指首次包含该能力的 OpenFHE 发布标签。论文中的原型和 OpenFHE 发布分别记录。

| 路线 | 论文 | OpenFHE 首次发布 |
| --- | --- | --- |
| **通用可编程自举，整数 LUT** | Andreea Alexandru、Andrey Kim、Yuriy Polyakov，*General Functional Bootstrapping using CKKS*。ePrint 2024/1623 于 2024 年 10 月 10 日收录首稿，正式发表于 CRYPTO 2025。用三角 Hermite 插值对整数批量求任意 LUT。见 [论文](https://eprint.iacr.org/2024/1623)。 | **v1.4.0**，2025 年 8 月 18 日开发版，发布说明明确列出这个方法，跟踪项 #954。整数路径使用 RLWE 整数编码，借助 CKKS 批量计算。见 [v1.4.0 发布标签](https://github.com/openfheorg/openfhe-development/releases/tag/v1.4.0) 和 [实现跟踪项](https://github.com/openfheorg/openfhe-development/issues/954)。 |
| **Fourier-extension 功能自举，平滑实函数** | Song Bian、Yunhao Fu、Ruiyu Shen、Haowen Pan、Anyu Wang、Zhenyu Guan，*High-Precision Functional Bootstrapping for CKKS from Fourier Extension*，EUROCRYPT 2026。对有界区间内的实数槽近似计算函数。见 [论文](https://eprint.iacr.org/2026/367)。 | **v1.6.0**，2026 年 9 月开发版，跟踪项 #1207，合入 PR #1276。接口为 `EvalFEFuncBootstrap`，下文 ReLU 实验使用这条路线。见 [v1.6.0 发布标签](https://github.com/openfheorg/openfhe-development/releases/tag/v1.6.0) 和 [实现 PR](https://github.com/openfheorg/openfhe-development/pull/1276)。 |

## 函数是怎样嵌进去的

OpenFHE 的普通 SlotsToCoeffs-first CKKS 自举依次执行 SlotsToCoeffs、ModRaise、CoeffsToSlots 和 EvalMod。升模引入整数倍的旧模数，EvalMod 近似消去这部分，恢复原槽位值。FE-FBT 保留前三步，将最后的 EvalMod 换成复指数和 Fourier 多项式。见 [普通自举](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/examples/CKKS_BOOTSTRAPPING.md) 和 [FE-FBT 源码](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp#L1597)。

令原函数的输入 $x$ 在 $[-B,B)$，上游把它归一化为 $m=x/(2B)$。以归一化消息单位看，升模后的值带有形如 $2k$ 的溢出项，其中 $k$ 是整数。因此 $z=e^{\pi i(m+2k)}=e^{\pi i m}$。用 Fourier extension 在目标区间拟合 $f(2Bm)$，将另一半周期留给延拓，避免直接周期化造成的端点跳变。计算 $2\operatorname{Re}(\sum_{j=0}^{d}c_j z^j)$ 同时消去溢出和近似应用 $f$。

| 阶段 | OpenFHE 1.6.1 的实现 |
| --- | --- |
| 离线选择函数与精度 | 选区间 $[-B,B]$ 和最高 Fourier 阶数 $d$，生成一侧复数系数。$c_0$ 取实常数项的一半。官方生成器按固定阶数搜索 Hermite 延拓阶数，并报告基于平均重构误差的精度。 |
| 配置和密钥 | 用 `FHECKKSRNS::GetFEFBTDepth` 确定深度，加入输出所需余量。调用 `EvalFEFuncBootstrapSetup`、`EvalMultKeyGen`、`EvalBootstrapKeyGen`。普通和 FE 的 setup 会占用同槽数的同一个缓存条目，A/B 测试应各自设置。 |
| 运行自举 | `EvalFEFuncBootstrapExp` 处理前三步，用 Chebyshev 级数计算复指数，再以平方和 rescale 做倍角。最后 `EvalPoly` 在 $z$ 上求系数多项式，阶数至少 5 时用 Paterson–Stockmeyer，计算结果加其共轭得到实输出。 |
| 同一密文求多个函数 | `EvalFEFuncBootstrapPrecompute` 预先刷新并计算复指数幂，之后用 `EvalFEFuncBootstrapWithPrecomp` 求每个函数。共享预计算传入最长的系数向量，阶数至少为 5。单函数延迟应计入刷新和幂计算。 |

输入必须为 $m\in[-1/2,1/2)$，需要预留 SlotsToCoeffs 的层级。当前实现要求 HYBRID 密钥切换、`FirstModSize = ScalingModSize + 1`，输出为实数，COMPLEX 格式的输入也只保留实部。此接口没有迭代自举模式，不接受 128-bit native 构建。见 [官方例程](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/examples/FE-functional-bootstrapping-ckks.cpp)、[系数生成器](https://github.com/openfheorg/fefbt-python) 和 [实现文档](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/examples/CKKS_FUNCTIONAL_BOOTSTRAPING.md)。

## 能支持什么函数

- **有界区间上的单变量实函数逐槽计算。** 官方例程覆盖 `exp`（区间 $[-2,2]$，阶数 29）、sigmoid（$[-8,8]$，34）、tanh 形式的 GELU（$[-8,8]$，44）。tanh、sin、softplus、多项式也符合输入形式。对数、倒数等函数需要在定义域内选取远离奇点的区间，再做仿射重标定并生成相应系数。
- **精度取决于区间与光滑性。** sigmoid 斜率变陡、平滑 ReLU 更接近硬折角，或输入区间变宽时，为达到相同误差往往要提高阶数。检查最大误差，特别是折角和区间边缘。每槽相同的标量函数不会同时实现跨槽聚合或 max pooling。
- **硬 ReLU 可以近似求值。** C++ 接口直接接收系数，没有导数检查。ReLU 连续但在零点不可微，有限 Fourier 多项式不能精确重现折角。可直接生成分段线性延拓的 Fourier 系数并传入，也可用平滑替代 $r_\varepsilon(x)=(x+\sqrt{x^2+\varepsilon^2})/2$。后者相对真 ReLU 的模型误差至多 $\varepsilon/2$，还需加入 Fourier 截断和 CKKS 数值误差。
- **量化 ReLU 有单独的整数路径。** 在离散输入集合上构造 $i\mapsto\max(0,i)$ 的查找表可表达精确的整数语义。OpenFHE 的 `EvalFBT` 整数 LUT 需要相应整数编码和 RLWE schemelet，具体正确性仍依赖参数与解码。见 [整数 LUT 文档](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/examples/CKKS_FUNCTIONAL_BOOTSTRAPING.md)。

EUROCRYPT 论文摘要报告，针对其功能自举对照，数据精度提高 10–27 bit，摊销延迟获得 1.1–2 倍的加速。普通 `EvalBootstrap` 的额外开销应另做直接测量。见 [论文摘要](https://eprint.iacr.org/2026/367)。

## CPU 对照

比较 `EvalBootstrap` 和 `EvalFEFuncBootstrap` 的一次完整调用。普通基线启用 `BTSlotsEncoding=true`，使两个流程都先做 SlotsToCoeffs。硬件为 Apple M1 Pro，使用 OpenFHE Release 静态库、64 位 native 配置、关闭 OpenMP。固定 $N=65{,}536$、1,024 槽、HYBRID 三组 digits、FLEXIBLEAUTO、59/60 bit scale/首模数、`SPARSE_ENCAPSULATED`、COMPLEX 格式中的实数输入、level budget `{3,2}` 和自动 BSGS。每个模式重新设置上下文和密钥，按各自深度函数增加 6 层。手工设定 `HEStd_NotSet`，这些数据用于算法对比，未对参数独立估算安全性。

输入为 $m_i=-1/2+i/1024$，普通自举对照 $m_i$，功能自举对照其对应的 $f(2Bm_i)$。每个模式先跑一次 warm-up，再连续记录三个调用的墙钟时间，取中位数。计时包含运行时的 Fourier 幂计算，setup、密钥生成、加解密和误差计算在计时之外。误差报告这 1,024 个解密槽相对明文参考的最大绝对误差。对于这组每层一个模数的参数，输出有效余量按 `chainDepth - GetLevel() - (GetNoiseScaleDeg() - 1)` 计算。

| 操作 | Fourier 阶数 | 延迟中位数 | 相对开销 | 深度 API 预算 | 输出有效余量 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通自举 | — | 10.76 s | 基线 | 15 | 8 | 1.88e-07 |
| 指数 $[-2,2)$ | 29 | 13.46 s | +25% | 21 | 8 | 1.86e-09 |
| sigmoid $[-8,8)$ | 34 | 13.92 s | +29% | 21 | 8 | 4.79e-10 |
| tanh 形式的 GELU $[-8,8)$ | 44 | 17.59 s | +64% | 21 | 8 | 2.19e-09 |
| ReLU $[-8,8)$ | 45 | 14.24 s | +32% | 21 | 8 | 0.0571 |
| ReLU $[-8,8)$ | 129 | 20.54 s | +91% | 23 | 8 | 0.0127 |

这组对照保持相同输出有效余量，功能自举使用更长的模数链。测得的差异包含复指数、Fourier 求值及更长模数链对其它阶段的影响。三次样本之间有波动，这些数字描述单机配置。若比较融合的收益，需另外测普通自举后再执行相同函数，并对齐精度和最终剩余层级。

<details markdown="1">
<summary>墙钟时间的三个样本</summary>

- 普通自举（—）10.999、10.760、10.719 s
- 指数 $[-2,2)$（29）15.822、13.463、13.259 s
- sigmoid $[-8,8)$（34）13.625、13.921、14.585 s
- tanh 形式的 GELU $[-8,8)$（44）23.034、16.554、17.595 s
- ReLU $[-8,8)$（45）14.189、14.330、14.238 s
- ReLU $[-8,8)$（129）19.396、20.541、20.934 s

</details>

### ReLU 系数的复现

取 $B=8$，把 ReLU 的区间扩为周期 $4B$ 的分段线性函数。一个周期 $[-2B,2B]$ 中，负区间输出 0，$[0,B]$ 上输出 $x$，$[B,2B]$ 上输出 $2B-x$。这个简单延拓用于观察硬折角，未优化 Fourier 最大误差。内积积分给出下面的单边系数，可以用它替换官方 GELU 示例里的系数向量。

```cpp
std::vector<std::complex<double>> coefficients(degree + 1);
coefficients[0] = B / 8.0;
for (uint32_t j = 1; j <= degree; ++j) {
    if (j % 4 == 0)
        continue;
    const double angle = std::acos(-1.0) * j / 2.0;
    const auto phase = std::polar(1.0, -angle);
    coefficients[j] = B * (2.0 * phase - 1.0 - phase * phase)
                      / (4.0 * angle * angle);
}
```

对同一组 1,024 槽直接明文求 Fourier 多项式，45 阶的最大绝对误差为 0.05712，零点输出为 0.03682。129 阶的最大绝对误差为 0.01266，落在零点。解密结果的最大误差和零点输出均与明文 Fourier 对照在所列精度上吻合。这两例的主要误差来自有限阶函数逼近。

## ReLU 阶数扫描

延续上面的 $[-8,8)$、周期分段线性延拓和 OpenFHE 1.6.1 参数。45、129 阶复用原始测量。新增阶数先各热身一次，再取两次完整调用的中位数；表中输出层数按与上文相同的口径计算。槽位最大误差来自 1,024 个解密槽，输入间距为 0.015625，最后一个输入为 7.984375。全区间误差是**明文 Fourier 截断的误差上确界**，其计算见下文。

| Fourier 阶数 | 延迟中位数 | 深度预算 | 槽位最大绝对误差 | 全区间明文误差上确界 | 零点输出 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 11.03 s | 19 | 0.34483 | 0.36032 | 0.19873 |
| 19 | 12.11 s | 20 | 0.14646 | 0.16178 | 0.08079 |
| 45 | 14.24 s | 21 | 0.05712 | 0.07206 | 0.03682 |
| 63 | 15.94 s | 22 | 0.03600 | 0.05065 | 0.02532 |
| 95 | 17.29 s | 22 | 0.01961 | 0.03377 | 0.01688 |
| 129 | 20.54 s | 23 | 0.01266 | 0.02513 | 0.01266 |
| 257 | 25.59 s | 24 | 0.00633 | 0.01262 | 0.00633 |
| 325 | 29.11 s | 24 | 0.00500 | 0.00998 | 0.00500 |

所有阶数解密前均保留 8 个有效层级。测量里，同一深度档内提高阶数仍会改变延迟。固定深度与固定误差上限是两个独立的约束，使用时应给两者分别留预算。

<details markdown="1">
<summary>各阶数的墙钟原始样本</summary>

- 9 阶 10.807、11.256 s
- 19 阶 12.049、12.172 s
- 45 阶 14.189、14.330、14.238 s
- 63 阶 15.872、16.007 s
- 95 阶 17.260、17.311 s
- 129 阶 19.396、20.541、20.934 s
- 257 阶 25.437、25.747 s
- 325 阶 29.363、28.853 s

</details>

### 深度为什么按台阶增长

相同参数下，`GetFEFBTDepth` 先加上 15 层固定预算，其中两次线性变换用 5 层，复指数的 Chebyshev 级数用 6 层，倍角平方用 4 层。再加 Fourier 多项式预算。OpenFHE 用系数向量长度查询 Paterson–Stockmeyer 深度档位，对归一化输入减 1。因此总预算随阶数分档，档位内仍可能因为乘法数量和调度不同而改变运行时间。

| Fourier 阶数 | 6–13 | 14–27 | 28–59 | 60–119 | 120–247 | 248–495 | 496–1007 | 1008–2031 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 本配置深度预算 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |

257 阶和 325 阶的总预算相同，Paterson–Stockmeyer 的分组参数 $(k,m)$ 分别为 $(18,4)$ 和 $(11,5)$。深度预算描述依赖路径的层数，评估中的总乘法量也会影响耗时。见 [GetFEFBTDepth](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp#L4380) 和 [GetDepthByDegree](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/lib/scheme/ckksrns/ckksrns-utils.cpp#L88) 和 [ComputeDegreesPS](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/lib/scheme/ckksrns/ckksrns-utils.cpp#L303)。

### 为什么区间误差高于槽位实测

本文的周期延拓在 $x=0$ 保留 ReLU 的折角，还在 $x=B$ 引入一个额外折角。对这些系数，乘上 $x=B$ 处的相位后，所有非零项都是正实数。截断尾项在这里同相叠加，在其它点可用三角不等式得到同一个上界。因此在半开区间 $[-B,B)$ 上取极限，得到精确的明文误差上确界。

$$
E_d=B-P_d(B)=B\sum_{j>d}\frac{w_{j\bmod 4}}{(\pi j/2)^2},
\qquad (w_0,w_1,w_2,w_3)=(0,1,2,1).
$$

另以 32,769 点的明文网格交叉验证，数值最大值与尾项公式吻合。129 阶的全区间误差为 0.02513，约为 1,024 槽采样最大值 0.01266 的两倍。这里最坏误差近似按 $4B/(\pi^2 d)$ 下降，阶数翻倍通常只增加约一位逼近精度。保持阶数时，误差也随 $B$ 线性缩放。

按这组延拓和 $B=8$，可用尾项公式求出达到各误差阈值的最低阶数。深度来自 1.6.1 的档位表，各行是逼近分析，CKKS 数值误差仍需单独计入。

| 全区间明文误差要求 | 最低阶数 | 对应深度预算 |
| ---: | ---: | ---: |
| < 0.1 | 33 | 21 |
| < 0.05 | 65 | 22 |
| < 0.02 | 162 | 23 |
| < 0.01 | 325 | 24 |
| < 0.005 | 649 | 25 |
| < 0.001 | 3242 | 27 |

例如 323 和 324 阶的误差均为 0.01000695，325 阶为 0.00997625。
