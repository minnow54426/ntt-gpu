# FHE 近两年进展

调研范围为 2024 年 11 月 18 日至 2026 年 11 月 18 日（Asia/Shanghai）。本文选取公开论文、正式软件发布和可核对的产品案例，重点关注 CKKS 密文推理以及 GPU 实现。下列性能数据均来自所引来源。

**这两年最大的变化，是优化对象从单个密码算子扩展到了整条计算路径。** 算法把非线性函数求值并入自举，编译器决定如何打包、管理精度和安排自举，GPU 系统则通过融合、显存管理和多卡调度减少搬运。小模型已有单卡亚秒级研究结果，大模型已能完成受限预填充；密封竞拍等任务则已上线。

## 算法、编译器和软件库

| 方向 | 代表进展 | 对实际工作的含义 |
| --- | --- | --- |
| CKKS 功能自举 | Alexandru、Kim、Polyakov 的 [CRYPTO 2025 论文](https://eprint.iacr.org/2024/1623) 用 CKKS 支持批量求整数 LUT，首次进入 OpenFHE [v1.4.0](https://github.com/openfheorg/openfhe-development/releases/tag/v1.4.0)。[EUROCRYPT 2026 的 Fourier Extension](https://eprint.iacr.org/2026/367) 面向有界区间上的平滑实函数，首次进入 [v1.6.0](https://github.com/openfheorg/openfhe-development/releases/tag/v1.6.0)。 | 整数 LUT 可以精确表达量化 ReLU 的规则。实数槽上的 ReLU 要衡量区间、逼近误差和开销。 |
| 推理编译 | Orion（ASPLOS 2025）将 PyTorch 模型编译为 CKKS 程序，处理槽位打包、自举插入和精度管理。NeuJeans 将卷积与自举的一部分变换融合，减少重复的变换。 | 网络结构和槽位布局会影响旋转、自举次数与执行深度。比较两种算子实现时，也要检查它们是否运行了同一个推理电路。见 [Orion](https://ckks.org/blog/2026/orion/) 和 [NeuJeans](https://ckks.org/blog/2025/NueJeans/)。 |
| 通用库 | OpenFHE 1.6.0 公告给出部分自举场景最高约 2× 的延迟改善、更新的稀疏密钥封装以及更低的 CKKS 自举失败概率。1.6.1 于 2026 年 10 月 9 日发布。 | CPU 基线应固定版本、线程数、参数和自举模式。上游报告的最高倍数依赖场景。见 [1.6.0 公告](https://openfhe.discourse.group/t/v1-6-0-development-is-released/2389) 和 [1.6.1 源码](https://github.com/openfheorg/openfhe-development/tree/v1.6.1)。 |
| TFHE 与整数逻辑 | Zama 在 2025 年为 TFHE-rs 发布开源 FPGA HPU 后端，同时维护 CUDA 后端和公开的整数、可编程自举基准。 | TFHE 的整数比较和查表适合离散逻辑。与 CKKS 的批量实数近似运算比较时，应先明确任务、批量与精度，再比较延迟。见 [HPU 发布](https://www.zama.org/post/announcing-hpu-on-fpga-the-first-open-source-hardware-accelerator-for-fhe) 和 [GPU 基准](https://docs.zama.org/tfhe-rs/get-started/benchmarks/gpu)。 |

实现细节、ReLU 与普通自举对照见 [OpenFHE 的 CKKS 功能自举](ckks-fourier-functional-bootstrap.md)。

## GPU 与专用芯片

单密文自举、ResNet-20 与 BERT-Base 的详细对照见 [GPU 与 ASIC 的密态推理基准](fhe-gpu-asic-inference-benchmarks.md)。

| 工作 | 公开数据 | 测量范围 |
| --- | --- | --- |
| Cheddar / Theodosian（2026） | 单张 RTX 5090 的 CKKS 自举 **22.1 ms**，ResNet-20 推理 **0.72 s**。 | Cheddar 的 32 位 RNS 实现。Theodosian 在其优化配置上发现 NTT 和基转换可受 L2 带宽限制，给出的 **8.8 ms** 是基于 L2 流量的估算下界。见 [作者测量与分析](https://ckks.org/blog/2026/CKKS-on-GPU/)。 |
| Cerium（SOSP 2026） | 1 张 B200 的自举 **14.5 ms**，8 张 B200 的自举 **7.5 ms**、BERT-Base **8.8 s**、Llama3-8B 预填充 **43 s**。 | 表 1 的 Llama 实验使用 **128 token** 提示词，43 秒是首 token 时间。模型权重与评估密钥离线准备，服务端持有明文权重。连续解码吞吐量未由该实验测量。见 [论文第 6 节](https://www.pdl.cmu.edu/PDL-FTP/HECStorage/3830418.3843864.pdf) 和 [源码](https://github.com/CMU-CAOS/Cerium)。 |
| Intel HERACLES（JSSC 2026） | Intel 3 制造的可编程芯片支持 BGV、BFV 和 CKKS。论文报告 **4.17 Tb/s** 的 NTT 比特吞吐，以及 **3.27 次/秒** 的自举加缩放吞吐。 | 芯片结合 HBM3、片上暂存和专用蝶形数据通路。两项数字覆盖不同操作，跨平台应按同一任务和参数重测。见 [JSSC 论文](https://doi.org/10.1109/JSSC.2026.3737700)。 |

Cerium 使用编译器融合内核、压缩编码后的模型权重，并调度主机与显存、多 GPU 之间的数据搬运。论文把自举、模型推理、编译和 CUDA Graph 构建分别计时。复现服务成本时还应记录密钥生成、编码、请求往返，以及解码场景的逐 token 时间。Cerium 与 CraterLake 等设计的比较使用论文中的 ASIC 仿真结果。见 [Cerium 表 1、表 2 与图 10](https://www.pdl.cmu.edu/PDL-FTP/HECStorage/3830418.3843864.pdf)。

## 应用与标准化

- **链上机密计算。** Zama 报告其协议于 2025 年 12 月 31 日上线主网，2026 年 1 月用它运行密封竞拍。主链合约记录密文句柄并触发任务，协处理器执行 FHE，门限 KMS 管理密钥和授权解密。评估这类系统需要考虑协处理器与解密权限。见 [上线与竞拍公告](https://www.zama.org/post/zama-developer-program-mainnet-season1-building-for-the-long-game)、[协处理器架构](https://www.zama.org/post/announcing-the-zama-confidential-blockchain-protocol) 和 [KMS 文档](https://docs.zama.org/protocol/protocol/overview/kms)。
- **私密查询。** Apple 的 Swift Homomorphic Encryption 库使用 BFV，为 Live Caller ID Lookup 和 Network Extension URL Filter 提供私密信息检索组件。这是固定查询型同态加密的落地案例，性能证据对应私密查表。见 [Apple 开源实现](https://github.com/apple/swift-homomorphic-encryption) 和 [URL Filter 文档](https://developer.apple.com/documentation/networkextension/url-filters)。
- **门限和可验证计算。** NIST 于 2026 年 1 月发布 IR 8214C，在 S5 征集门限 FHE 方案、实现和评估。该文件属于公开征集。USENIX Security 2026 的 HasteBoots 报告对含 TFHE 可编程自举的求值在数秒内生成证明，展示了如何验证特定同态计算的完整性。见 [NIST 征集文件](https://doi.org/10.6028/NIST.IR.8214C) 和 [HasteBoots](https://www.usenix.org/conference/usenixsecurity26/presentation/liu-fengrun)。

## 另外值得关注的进展

| 方向 | 进展及其意义 |
| --- | --- |
| **更大的整数 LUT** | *Evaluating Larger Lookup Tables using CKKS*（TCHES 2026）提出两种面向较大 LUT 的方法，论文目标覆盖至 20 位输入。16 位实验中，作者的单线程 OpenFHE 实现相较 CRYPTO 2025 方法，单次延迟加速 95.1 倍，每槽摊销时间加速 47.5 倍。这为量化激活、分段函数和比较提供另一条路线。论文实现的这组数据对应特定基线、参数和槽位数。见 [期刊论文](https://doi.org/10.46586/tches.v2026.i1.559-591)。 |
| **普通 CKKS 自举的精度和层级** | CCS 2025 的 *Leveraging Discrete CKKS to Bootstrap in High Precision* 用整数清理提高自举精度，论文将其与门限 FHE 和需加噪的解密场景联系起来。PKC 2026 的 *Faster Bootstrapping for CKKS with Less Modulus Consumption* 在作者基准中省去一个模数层、提升 20%–35% 的吞吐，同时保持原基线的精度和失败概率。比较这些方法时，应同时记录输出层数和评估密钥大小。见 [高精度自举](https://eprint.iacr.org/2025/1786) 和 [节省模数层的自举](https://eprint.iacr.org/2025/1403)。 |
| **原生矩阵同态加密** | Gentry–Lee（GL）围绕矩阵算术设计密文表示，支持批量矩阵的乘法及转置。随后的 GL 自举工作把 CtS、StC 表达为方案原生支持的明密矩阵乘法，原型中的主要瓶颈转向 EvalMod。它给矩阵密集工作负载提供了一条减少旋转与格式转换的方案级路线。见 [GL 论文](https://eprint.iacr.org/2025/1935) 和 [GL 自举论文](https://eprint.iacr.org/2026/956)。 |
| **多密钥 FHE** | EUROCRYPT 2026 的 HERDS 面向分别用各自密钥加密的多方输入。论文将其自举复杂度从已有方案的 $O(dkn)$ 降至 $O(kn)$，并给出并行复杂度及 16 方的基准。它为跨机构联合计算提供了独立于共享门限密钥的技术路线。见 [HERDS 论文](https://link.springer.com/chapter/10.1007/978-3-032-25327-9_8)。 |
| **通用 FHE 编译器** | Google HEIR 用 MLIR 表达同态计算与优化，公开了 OpenFHE 和 Lattigo 的生成路径，并在 2026 年公开四个由 HEIR 编译的私密推理应用。它使数据布局、方案参数和底层库有机会在同一工具链内比较。可用操作随方案和后端而异。见 [HEIR 仓库](https://github.com/google/heir) 和 [应用示例说明](https://blog.google/security/how-google-is-making-private-ai-practical-with-homomorphic-encryption/)。 |
| **FHE 国际标准进入最终草案** | 截至 2026 年 11 月 18 日，ISO/IEC 28033 的第 1 部分（通则）和第 3 部分（近似数算术机制）均列为 FDIS，即最终国际标准草案。第 1 部分在 11 月 11 日进入 50.00 阶段。见 [ISO 第 1 部分状态](https://www.iso.org/standard/87638.html) 和 [ISO 第 3 部分状态](https://www.iso.org/standard/87640.html)。 |

## 从论文基准到应用还要验证什么

- **协议安全与模型隐私。** 加密输入、隐藏服务端模型、限制结果泄露和证明服务器按约定计算，对应不同的安全目标。ASIACRYPT 2025 的 CKKS 协议论文提出差分隐私意义下的服务端隐私，并用零知识知识论证约束恶意接收方提交的公钥及密文。OpenFHE 安全文档也单独处理 CKKS 共享解密结果的场景。公开推理结果时应写清威胁模型，以及输出释放、噪声泛洪和验证的成本。见 [CKKS 协议安全论文](https://eprint.iacr.org/2025/382) 和 [OpenFHE 安全说明](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/security.html)。
- **整条电路的失败概率。** 安全位数、自举有效精度和算错概率要分别报告。若一条电路有 $m$ 个可能失败的调用，且第 $i$ 个调用的失败概率至多 $p_i$，并集界给出这部分的请求失败率上界 $\sum_i p_i$，不要求独立。2026 年的稀有事件仿真工作用 importance splitting 检验噪声尾部模型，实证对象为 TFHE 及其自举变体。CKKS 应针对自己的数值误差阈值、输入范围与失败机制验证。见 [稀有事件仿真论文](https://eprint.iacr.org/2026/610)。
- **密文上传和转密文（transciphering）。** 客户端可先上传紧凑的对称密文，服务端用 FHE 加密的对称密钥同态求解密电路，得到可继续计算的 FHE 密文。2026 年的 AES-on-CKKS 预印本在两种配置下分别报告单个 AES block **26 ms**，以及大批量 **238 KB/s**。应用的完整比较应纳入密钥材料、上传字节、分块或批量、转换时间和完整性处理。上述数字来自 AES 电路实验。见 [AES-on-CKKS 预印本](https://eprint.iacr.org/2026/1209)。
- **LLM 逐 token 解码与方案切换。** 自回归 decode 的并行形态与 prefill 不同。ROSETTA 为不同非线性算子选择 CKKS 或 TFHE，并按模块测量估算完整的一次解码迭代。表 7 在 A100 80 GB 上给出 LLaMA-3-8B、512 token 上下文的 **5.18 分钟/token**，8k 上下文为 **6.79 分钟/token**，计时排除 prefill。这组结果适合定位算子、切换和长上下文的成本。用不同论文的首 token 时间、模块估计和连续生成吞吐比较系统时，应分别标注。见 [ROSETTA 第 6 节及表 7](https://arxiv.org/html/2609.16915v1)。
- **可复现的端到端评估。** 单次自举、每请求墙钟和批量摊销分别记录，再加上任务准确率、冷启动、上传下载和评估密钥内存。HomomorphicEncryption.org 的公开 BERT harness 固定 BERT-Base 和 GLUE MRPC，可作为统一任务边界的起点。见 [FHE Benchmarking Suite](https://fhe-benchmarking.github.io/) 和 [BERT harness](https://github.com/fhe-benchmarking/BERT)。

## 对 GPU 密文推理项目的影响

当前优先级仍是获得 **RTX 5090 上与 OpenFHE 1.6.1 精确对齐的原生基线**。使用现有 Hybrid 参数、模数顺序、根和表示，按项目队列检查设备常驻链，再推进基转换和完整密钥切换。早期 Ada 数据只代表该次运行的配置。见 [当前路线图](openfhe-integration-plan.md) 和 [GPU 验证队列](next-gpu-rental.md)。

1. **先量驻留链和总搬运。** 在相同输入下比较设备事件时间和包含上传、下载的公共调用时间。记录 DRAM 与 L2 流量、批量、显存峰值，以及 NTT、基转换和评估密钥读取的分项耗时。L2 限制是待验证的假设，Cheddar 使用的窄模数与本项目冻结的 OpenFHE 模数配置不同。
2. **把完整 Hybrid key switch 作为第一个有代表性的后续基准。** 固定环维（4096–65536）、活跃 Q/P towers、Hybrid digit 划分和评估密钥，分段计时 ModUp、NTT/逆 NTT、评估密钥乘加和 ModDown。先逐 tower 精确对比 CPU，再报告整个操作的延迟、吞吐和显存峰值。
3. **用小型 CKKS 推理验证组合收益。** 后续选固定输入、打包方式和精度的浅层模型，再逐步加入自举。通过同参数 CPU 与 GPU 的全路径比较，检验 NTT、基转换和密钥切换优化在应用里保留多少收益。跨论文引用性能时，应注明硬件、模数链与安全参数、槽位、批量、输出误差和计时边界。

下一步可交付的实验及验收顺序见 [GPU 验证队列](next-gpu-rental.md) 和 [基准与调优计划](benchmark-tuning-plan.md)。
