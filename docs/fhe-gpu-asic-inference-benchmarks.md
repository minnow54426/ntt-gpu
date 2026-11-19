# GPU 与 ASIC 的密态推理基准

**截至 2026 年 11 月 18 日。** 本文关注单个 CKKS 密文的自举墙钟延迟，以及以加密输入运行单次模型推理的延迟。挑选既公开数字又能核对实验口径的工作。GPU 表的结果来自物理 GPU，ASIC 表注明仿真或综合后建模，另列已流片芯片的吞吐。推理数字来自各自的论文，测试程序、参数和计时范围随工作而异。

## benchmark 用到的 GPU 规格

Cerium 的三个数据中心型号按论文里的 DGX SXM 配置填写。GeForce 按 NVIDIA 公布的桌面卡规格填写。表中为单张完整 GPU 的标称规格及硬件驻留上限。

| GPU | 架构 / CUDA CC | SM | 每 SM 的 FP32 CUDA 核心 | CUDA 核心总数 | 每 SM 最大驻留线程 | 单 GPU 最大驻留线程 | 显存 | 理论显存带宽 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A100 80 GB SXM4 | Ampere / 8.0 | 108 | 64 | 6,912 | 2,048 | 221,184 | 80 GB HBM2e | 2.039 TB/s |
| H100 SXM5 | Hopper / 9.0 | 132 | 128 | 16,896 | 2,048 | 270,336 | 80 GB HBM3 | 3.35 TB/s |
| B200 SXM | Blackwell / 10.0 | 148 | 128 | 18,944 | 2,048 | 303,104 | 180 GB HBM3e | 最高 8 TB/s |
| GeForce RTX 4090 | Ada / 8.9 | 128 | 128 | 16,384 | 1,536 | 196,608 | 24 GB GDDR6X | 1.008 TB/s |
| GeForce RTX 5090 | Blackwell / 12.0 | 170 | 128 | 21,760 | 1,536 | 261,120 | 32 GB GDDR7 | 1.792 TB/s |

CUDA 核心数在此指 FP32 运算通道数。驻留线程上限按 SM 数乘以每 SM 的最大驻留 warp 数，再乘以每 warp 的 32 个线程计算。实际 kernel 的驻留线程取决于每块线程数、寄存器、共享内存和块数限制。一个 CUDA block 最多有 1,024 线程，大于单卡驻留上限的 grid 可以分批调度。

8 张 B200 共 1,440 GB 显存、151,552 个 CUDA 核心、2,424,832 个理论驻留线程。这些是逐卡相加的资源。单卡仍有 180 GB 本地显存，跨卡操作需要 GPU 间通信。CKKS 还要看模运算、缓存层级及数据搬运，CUDA 核心数本身无法预测延迟。

规格见 [A100](https://www.nvidia.com/en-us/data-center/a100/)、[H100 架构](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)、[HGX 显存与带宽](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)、[B200 SM 数](https://developer.nvidia.com/blog/boost-gpu-memory-performance-with-no-code-changes-using-nvidia-cuda-mps/)、[RTX 4090 架构规格](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)、[RTX 5090 架构规格](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) 和 [CUDA 硬件限制](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)。

## 单次 CKKS 自举

“单次”是一整个密文刷新完的时间。一个密文可以包含上万槽。按槽或按批内密文切片摊销的时间可用于计算吞吐。

| 实现 | 平台 | 单密文刷新 | 自举配置与证据 |
| --- | --- | ---: | --- |
| Cheddar | 1 × RTX 5090 | 22.1 ms | 作者表 2 的 scale 为 $2^{40}$。同配置 Theodosian 为 15.2 ms。 |
| Theodosian | 1 × RTX 5090 | 15.2 ms | 同上。另一组 scale $2^{35}$ 的配置为 16.21 ms。加入算法改进后为 **12.75 ms**，$N=65536$、输出模数预算 575 bit、报告精度 18.70 bit。 |
| Cerium | 1 × B200 / 8 × B200 | 14.5 / 7.5 ms | 同一密文可跨卡执行。目标 128-bit 安全性，$N=65536$，28-bit RNS，输出到 level 18，自举约 20 bit 精度。 |
| Cerium | 1 × H100 / 8 × H100 | 19.5 / 9.7 ms | 同一论文、同一自举配置。 |
| Cerium | 1 × A100 80 GB / 8 × A100 | 34.2 / 16.5 ms | 同一论文、同一自举配置。 |
| HEonGPU regular | 1 × RTX 5090 / 1 × RTX 4090 | 63.51 / 109.28 ms | 官方项目表，$N=65536$、32768 槽、输出还有 9 层，精度约 18 bit。 |

来源见 [Cheddar 与 Theodosian 表 2–3](https://ckks.org/blog/2026/CKKS-on-GPU/)、[Cerium 第 6.1 节与表 1](https://www.pdl.cmu.edu/PDL-FTP/HECStorage/3830418.3843864.pdf) 和 [HEonGPU 基准](https://github.com/enCRYPTON-labs/HEonGPU)。上表均为普通刷新。包含 ReLU 的功能自举是另一个电路，相关比较见 [CKKS 功能自举笔记](ckks-fourier-functional-bootstrap.md)。

## GPU 的单请求密态网络

| 实现 | 设备 | ResNet-20 单图 | BERT-Base 单输入 | Llama3-8B 预填充 |
| --- | --- | ---: | ---: | ---: |
| Cheddar | 1 × RTX 5090 | 720 ms | — | — |
| Theodosian | 1 × RTX 5090 | 467 ms | — | — |
| Cerium | 1 × A100 80 GB | 1,050 ms | 66.1 s | 555 s |
| Cerium | 1 × H100 | 620 ms | 36.1 s | 370 s |
| Cerium | 1 × B200 | 456 ms | 28.3 s | 241 s |
| Cerium | 8 × H100 | 386 ms | 10.6 s | 62 s |
| Cerium | 8 × B200 | **298 ms** | **8.8 s** | **43 s** |

Cerium 的 ResNet-20 对一张 32 × 32 CIFAR-10 图像分类，准确率 91.4%。BERT-Base 输入 128 token，GLUE RTE 准确率 69.3%，电路有 997 次自举。两者报告的明文和密文准确率相同。Llama3-8B 用 128 token 提示词，43 秒为论文定义的首 token 时间。评估密钥、明文权重编码及图的准备在计时范围外。Cerium 同代 B200 从 1 卡扩至 8 卡时，自举、ResNet-20、BERT 加速分别约 **1.93×、1.53×、3.22×**。不同网络可并行的工作量不同，多卡通信会影响延迟。见 [Cerium 表 1 与表 2](https://www.pdl.cmu.edu/PDL-FTP/HECStorage/3830418.3843864.pdf) 和 [Theodosian 表 2](https://ckks.org/blog/2026/CKKS-on-GPU/)。

BERT 可参考 Cerium 同表列出的 THOR 约 602 秒（单 A100），并保留论文间的模型和计时边界。图像侧也要核对模型与数据集。WPC 的单 A100 ImageNet 表给出 ResNet-18 的 3.68 秒（Hermite 激活）或 10.23 秒（PolyReLU），ResNet-50 的 24.47 秒（PolyReLU）。WPC 的数据集和网络配置与 CIFAR-10 的 32 × 32 ResNet-20 不同。见 [Cerium 表 1](https://www.pdl.cmu.edu/PDL-FTP/HECStorage/3830418.3843864.pdf) 和 [WPC 表 7](https://www.thu-haslab.org/publication/2025-wpc/3719027.3765022.pdf)。

### 批量吞吐的例子

Terrazzo 使用 Gentry–Lee（GL）矩阵同态方案。它在 1 张 RTX 5090 上刷新整个 GL 密文耗时 **1.458 s**，论文按 128 个 RLWE 切片折算为 **11.39 ms/切片**。128 token 的 BERT-Base 一批承载 256 个独立输入，满批墙钟约 **1,802 s**，摊销 **7.04 s/输入**，约 **0.142 输入/s**。论文指出，同一个 batch 仅填 1 个输入也要执行整个电路。因此这些数字衡量批处理效率，交互延迟应看整批墙钟。见 [Terrazzo 预印本，表 1–3](https://eprint.iacr.org/2026/1870)。

## ASIC 设计与流片结果

| 设计 | 设备规模 | 单密文自举 | ResNet-20 单图 | BERT-Base / 128 token | 证据 |
| --- | --- | ---: | ---: | ---: | --- |
| Cinnamon-M | 1 个大单片设计 | 1.87 ms | 105.94 ms | 3.83 s | 综合 RTL 后的周期级仿真 |
| Cinnamon-4 | 4 个芯片 | 1.98 ms | 94.52 ms | 3.83 s | 同上 |
| Cinnamon-8 | 8 个芯片 | 1.71 ms | 73.85 ms | 2.07 s | 同上 |
| Cinnamon-12 | 12 个芯片 | 1.63 ms | 70.57 ms | 1.67 s | 同上，使用交换互连 |
| ARK | ASIC 设计 | 3.50 ms | 125 ms | — | Cinnamon 表 2 引用的仿真 |
| CraterLake | ASIC 设计 | 6.33 ms | 321.26 ms | — | Cinnamon 表 2 引用的仿真 |
| Intel HERACLES | Intel 3 实测 SoC | — | — | — | 公开指标包括 **3.27 次/s** 的自举加缩放吞吐和 **4.17 Tb/s** 的 NTT 比特吞吐 |

Cinnamon 自举基准报告 128-bit 安全性，$N=65536$，输出留 13 个有效应用层。它的 BERT 是单个 128 token 输入的完整网络，论文描述的自举调用数约 1,400 次。自举算法、输出层和应用电路均会影响比较。HERACLES 是已流片证据，3.27 次/s 是复合操作吞吐；其倒数是该吞吐下每完成一次操作的平均时间间隔。这里没有可用于和上表对齐的 HERACLES 独立单次 CKKS 延迟或 ResNet/BERT 结果。见 [Cinnamon 第 6 节及表 2](https://www.cs.cmu.edu/~dskarlat/publications/cinnamon_asplos25.pdf) 和 [HERACLES JSSC](https://doi.org/10.1109/JSSC.2026.3737700)。

## 如何读 GPU 和 ASIC 的差距

- **同为八个设备的配置下，Cinnamon 的仿真延迟更低。** 8 张 B200 的 Cerium 对比仿真的 Cinnamon-8，自举为 7.5 对 1.71 ms，ResNet-20 为 298 对 73.85 ms，BERT-Base 为 8.8 对 2.07 s。三个延迟比分别约 **4.4×、4.0×、4.3×**。这里是公开配置的数值之比。Cerium 自举输出到 level 18，Cinnamon 保留 13 个有效应用层，两个网络电路也分别实现。
- **与较早的 CraterLake 仿真数值重叠。** 同一张对照表中，8 B200 的 7.5 ms 自举略慢于 CraterLake 的 6.33 ms，ResNet-20 的 298 ms 则略快于 321.26 ms。竞争关系取决于 ASIC 代际、工作负载和设备规模。
- **复现实验时锁定语义。** 自举至少固定 $N$、槽数、输入与输出模数链、精度、安全参数和失败概率，注明单密文或批量。网络至少固定数据集、准确率、模型版本、token 数或图像大小、批量、在线计时边界、密钥与编码权重的驻留方式。HomomorphicEncryption.org 的 BERT 工作负载使用 GLUE MRPC，可用作公共比较基准。Cerium 表中的准确率来自 GLUE RTE。见 [标准化 benchmark](https://fhe-benchmarking.github.io/) 和 [BERT 基准代码](https://github.com/fhe-benchmarking/BERT)。

来源说明见 [Cerium 图 10](https://www.pdl.cmu.edu/PDL-FTP/HECStorage/3830418.3843864.pdf) 和 [Cinnamon 表 2](https://www.cs.cmu.edu/~dskarlat/publications/cinnamon_asplos25.pdf)。本次调研引用公开结果。
