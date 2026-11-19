# FHE 的三个进展

从可编程自举、可验证求值，到公链上的机密竞拍

**2026 年 11 月 18 日**

一次同态计算可以拆成三个问题。密文上能高效算什么？谁能检查执行是否正确？真实应用如何决定哪些值可以解密？本文沿着这三个问题，比较论文基准和 OpenFHE 实现，再追踪一笔主网密封报价。

## 01 · 可编程自举

普通自举刷新密文并近似恢复原值。功能自举在刷新时对输入求函数。OpenFHE 先后加入了整数查表和近似实函数两条路线。

| 路线 | 代表论文 | 首次 OpenFHE 发布 |
| --- | --- | --- |
| 整数 LUT | Alexandru、Kim、Polyakov，[General Functional Bootstrapping using CKKS](https://eprint.iacr.org/2024/1623)，CRYPTO 2025。用三角 Hermite 插值支持批量整数查表。 | [v1.4.0](https://github.com/openfheorg/openfhe-development/releases/tag/v1.4.0)，2025 年 8 月。 |
| 实数 Fourier extension | Bian 等，[High-Precision Functional Bootstrapping for CKKS from Fourier Extension](https://eprint.iacr.org/2026/367)，EUROCRYPT 2026。对有界区间的实数槽逐槽近似求值。 | [v1.6.0](https://github.com/openfheorg/openfhe-development/releases/tag/v1.6.0)，2026 年 9 月。 |

整数路线可以精确表达量化 ReLU 的查表规则。实数路线由 `EvalFEFuncBootstrap` 接收离线生成的 Fourier 系数。它沿用 SlotsToCoeffs、ModRaise 和 CoeffsToSlots，再用复指数的周期性消除升模溢出，并在复指数上求 Fourier 多项式。官方例程包含指数、sigmoid 和 tanh 形式的 GELU。见 [OpenFHE 实现说明](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/examples/CKKS_FUNCTIONAL_BOOTSTRAPING.md)。

### 硬 ReLU 的阶数、误差和开销

以下是 OpenFHE 1.6.1、Apple M1 Pro 单线程、环维 65,536、1,024 槽的本地结果。所有测试模式自举后都保留 8 个有效层级。延迟是自举 API 的完整调用中位数，密钥生成、加解密与初始化在计时之外。新增阶数各测两次，45 和 129 阶各测三次。

| Fourier 阶数 | 延迟中位数 | 深度预算 | 解密槽最大绝对误差 | 全区间明文逼近误差上确界 |
| ---: | ---: | ---: | ---: | ---: |
| 普通自举 | 10.76 s | 15 | — | — |
| 9 | 11.03 s | 19 | 0.34483 | 0.36032 |
| 45 | 14.24 s | 21 | 0.05712 | 0.07206 |
| 63 | 15.94 s | 22 | 0.03600 | 0.05065 |
| 95 | 17.29 s | 22 | 0.01961 | 0.03377 |
| 129 | 20.54 s | 23 | 0.01266 | 0.02513 |
| 257 | 25.59 s | 24 | 0.00633 | 0.01262 |
| 325 | 29.11 s | 24 | 0.00500 | 0.00998 |

硬 ReLU 在零点有折角，这组周期延拓还在右边界引入折角，令全区间误差高于槽位采样最大值。对这组系数和区间 [−8, 8)，**325 阶首次使 Fourier 截断误差低于 0.01**。CKKS 的数值误差要另外计算。深度则按 Paterson–Stockmeyer 档位增长，因而 257 和 325 阶的预算相同，延迟仍不同。见 [OpenFHE 深度实现](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/lib/scheme/ckksrns/ckksrns-utils.cpp#L88)。

作为数量级参考，Cerium 报告普通 CKKS 自举在单张 B200 上耗时 14.5 ms，8 张 B200 上为 7.5 ms。8 卡的单图 ResNet-20 与 128-token BERT-Base 分别是 298 ms 和 8.8 s。Cinnamon-8 对这三项的 ASIC 仿真值分别是 1.71 ms、73.85 ms 和 2.07 s。两项工作的电路、参数和输出层各有设定。见 [Cerium 表 1](https://www.pdl.cmu.edu/PDL-FTP/HECStorage/3830418.3843864.pdf) 和 [Cinnamon 表 2](https://www.cs.cmu.edu/~dskarlat/publications/cinnamon_asplos25.pdf)。

## 02 · 可验证 FHE

加密保证求值过程中可以保密。可验证求值增加另一项保证，检查输出是否确实由约定的密文计算产生。Zama 的 CCS 2025 原型用 Plonky2 证明经过调整的 TFHE 自举。USENIX Security 2026 的 HasteBoots 则直接证明多项式与商环关系，减少把 NTT 等运算逐步展开成通用证明电路的成本。

| 论文配置 | 生成证明 | 验证证明 | 证明大小 |
| --- | ---: | ---: | ---: |
| Zama / Plonky2，HasteBoots 论文复跑 | 25 分钟 | 4 ms | 0.20 MB |
| HasteBoots / Dory，论文的 Zama 参数对照 | 12.36 s | 118.12 ms | 0.22 MB |
| HasteBoots / BabyBear + Dory，单次 | 5.47 s | 109.35 ms | 0.24 MB |
| HasteBoots / BabyBear + Dory，16 次一批 | 60.90 s | 126 ms | 0.28 MB |

前两行的证明生成时间相差约 120 倍，验证时间相差约 30 倍，方向相反。16 次一批的数字使用 BabyBear 参数。论文测量的是 TFHE 求值证明，链上部署还需要测 verifier gas、提交数据成本和结算延迟。见 [HasteBoots 表 2 和表 4](https://www.usenix.org/system/files/usenixsecurity26-liu-fengrun.pdf)、[USENIX 实验材料](https://secartifacts.github.io/usenixsec2026/results) 和 [Plonky2 原型代码](https://github.com/zama-ai/verifiable-fhe-paper)。

## 03 · 机密合约与链上金融

Zama 在 2026 年 1 月于 Ethereum 主网运行密封竞拍。报价价格公开，申报数量保密，清算价在加密数据上计算。GSR 在同年 3 月另行公布了一笔采用该协议的机密 OTC 交易。见 [竞拍复盘](https://www.zama.org/post/118m-committed-for-the-first-encrypted-ico-on-ethereum) 和 [GSR 案例](https://www.zama.org/post/the-confidentiality-layer-bringing-institutional-trade-execution-to-ethereum-with-gsr)。

| 一笔报价的阶段 | 机制 | 验证的内容 |
| --- | --- | --- |
| 提交 | 客户端附输入零知识证明，协处理器验证。Solidity 侧可从 `FHE.fromExternal` 看起。 | 输入密文是否按要求构造。 |
| 求值 | 合约发出指令，链下协处理器用 TFHE-rs 计算，Gateway 接受达到多数的一致签名。 | 依据多数诚实假设接受计算结果。 |
| 解密 | Gateway 检查权限，由门限 KMS 共同解密并签名。 | 谁有权看到哪些明文，以及结果的来源。 |

HasteBoots 研究的求值证明对应中间一行。主网文档描述的是多数协处理器签名路径，公开竞拍材料没有显示使用 HasteBoots 逐次证明求值。另一个主网实现 Fhenix CoFHE 的初始部署披露，密钥份额由 Fhenix 运营的基础设施托管并受 TEE 保护，独立参与方属于逐步推进的计划。见 [Zama 协处理器](https://docs.zama.org/protocol/protocol/overview/coprocessor)、[Gateway](https://docs.zama.org/protocol/protocol/overview/gateway)、[KMS](https://docs.zama.org/protocol/protocol/overview/kms) 和 [Fhenix 技术说明](https://www.fhenix.io/litepaper)。

阅读实现可从 [Zama FHEVM](https://github.com/zama-ai/fhevm) 的 `host-contracts/`、`coprocessor/` 和 `kms-connector/` 开始。[ERC-7984 接口草案](https://eips.ethereum.org/EIPS/eip-7984) 用 `bytes32` 句柄表示保密金额，计算与解密的机制由具体实现提供。

## 三个问题可以一起验收

- **能不能算** · 给出函数语义、误差预算、剩余层级和一次完整求值的耗时。
- **能不能验** · 写清验证关系，分别测生成证明、验证证明和数据大小。
- **能不能上线** · 明确公开字段、解密权限、节点信任和结算时间，并标明实际部署采用的机制。
