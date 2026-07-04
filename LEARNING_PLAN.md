# Tensor Core / HGEMM 学习计划

以本仓库为教材，从零开始学习 Tensor Core 编程，最终目标：**能独立读懂并解释 [src/mma/mma_async_stage4.cu](src/mma/mma_async_stage4.cu)（644 行，本仓库最复杂的 kernel），并理解它为什么能接近甚至超过 cuBLAS 的性能。**

> 节奏假设：每周投入 3~5 个晚上，每晚 1~2 小时。基础好可压缩到 2~3 周。
> 硬件：租用 Vast.ai / RunPod 的 RTX 3090（sm_86），按秒计费，每次实验后记得关实例。

---

## 阶段 0：CUDA 基础自检（0.5~2 天）

如果下面的问题你都能不查资料回答，直接进入阶段 1；否则先补基础（推荐 CUDA C++ Programming Guide 第 1~5 章，或《Professional CUDA C Programming》前几章）。

- [ ] thread / block / grid / warp 分别是什么？warp 为什么固定是 32 个线程？
- [ ] global memory / shared memory / register / L1 / L2 的容量和延迟量级差异？
- [ ] 什么是内存合并访问（coalescing）？warp 内线程访问连续地址为什么快？
- [ ] `__syncthreads()` 同步的是什么范围？warp 内需要同步吗？
- [ ] shared memory 的 bank 是什么？什么情况下发生 bank conflict？
- [ ] occupancy 是什么？寄存器用量和 shared memory 用量如何影响它？

**产出**：把答不上来的问题记下来，补完后用自己的话写一遍答案。

---

## 阶段 1：环境搭建 + 跑通基线（1~2 晚）

### 1.1 租机器并搭环境

- [ ] 在 Vast.ai 租一台 RTX 3090（Interruptible 实例，约 $0.10~0.15/hr），选带 CUDA 11.x+ 的镜像
- [ ] SSH 上机，安装依赖：
  ```bash
  sudo apt-get update && sudo apt-get install -y libgflags-dev ccache cmake
  ```
- [ ] 克隆并编译本仓库：
  ```bash
  git clone https://github.com/clveryang/cuda_hgemm.git
  cd cuda_hgemm
  ./build.sh -a 86 -t Release -b OFF   # RTX 3090 是 sm_86；A100 用 -a 80
  ```

### 1.2 跑通并记录基线

- [ ] 运行 `./run_sample.sh`，保存输出日志
- [ ] 手动跑一次带正确性校验的小规模测试：
  ```bash
  ./output/bin/hgemm -M=512 -N=2048 -K=1024 -enable_check=true
  ```
- [ ] 阅读 [src/main.cu](src/main.cu)，搞清楚：gflags 参数有哪些、`Tester` 依次评测了哪些 kernel、每个名字对应 src 下哪个文件
- [ ] 阅读 [src/common/tester.h](src/common/tester.h)，理解吞吐量（TFLOPS）是怎么算出来的：GEMM 的浮点运算量为什么是 `2*M*N*K`？

**检查点**：能说出 cuBLAS 在你这张卡上 M=N=K=4096 时大约多少 TFLOPS，并解释这个数字和 RTX 3090 的 FP16 Tensor Core 理论峰值（约 142 TFLOPS，稀疏除外）差距在哪。

---

## 阶段 2：Tensor Core 概念 + WMMA API 入门（1 周）

### 2.1 概念（1~2 晚）

- [ ] 阅读 CUDA C++ Programming Guide 的 **Warp Matrix Functions (WMMA)** 章节
- [ ] 阅读 NVIDIA Ampere 架构白皮书中 Tensor Core 部分
- [ ] 用自己的话回答：
  - [ ] Tensor Core 本质上做的是什么运算？（提示：一个 warp 协作完成固定尺寸的 `D = A×B + C`，fp16 下是 16×16×16）
  - [ ] `wmma::fragment` 存在哪里？为什么说它的内部布局是"编译器黑盒"？
  - [ ] 为什么 `load_matrix_sync` / `mma_sync` / `store_matrix_sync` 必须整个 warp 一起调用？

### 2.2 读第一个 kernel：wmma_naive（1~2 晚）

- [ ] 精读 [src/wmma/wmma_naive.cu](src/wmma/wmma_naive.cu)（仅 51 行），逐行理解：
  - [ ] grid/block 怎么划分的？一个 block 几个线程？一个 warp 负责 C 的哪块区域？
  - [ ] `load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K)` 中第三个参数（leading dimension）是什么含义？
  - [ ] B 矩阵为什么用 `col_major`？
- [ ] 在 main.cu 里取消注释 `wmmaNaive` 和 `simtNaive`，重新编译跑 benchmark
- [ ] 记录三者性能：simtNaive（纯 CUDA core）vs wmmaNaive（Tensor Core）vs cuBLAS

**检查点**：不看代码，口头复述 wmma_naive 的 5 个步骤（fill C_frag → 循环 K_tiles → load A/B frag → mma_sync → store）。能解释它为什么远慢于 cuBLAS（提示：没有数据复用，A/B 被重复从 global memory 读取多次）。

### 2.3 读 wmma_base：加入 shared memory（1~2 晚）

- [ ] 精读 [src/wmma/wmma_base.cu](src/wmma/wmma_base.cu)，对比 naive 版本回答：
  - [ ] block tile（256×128）和 warp tile（64×64）分别是什么？一个 block 里有几个 warp？
  - [ ] 数据流：global → shared → fragment，每一步由谁（哪些线程）搬运？
  - [ ] shared memory 复用把 global memory 读取量降低了多少倍？
- [ ] 跑 benchmark，记录 base 相比 naive 的提升

---

## 阶段 3：Bank Conflict 与访存优化（1 周）

### 3.1 Bank conflict 原理（1 晚）

- [ ] 复习：shared memory 有 32 个 bank，每 bank 每周期服务一个 4 字节（sm_80+ 可配置）请求
- [ ] 回答：warp 内 32 个线程访问 `smem[tid * 32]` 会发生什么？访问 `smem[tid * 33]` 呢？
- [ ] 了解诊断手段：`ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared` 

### 3.2 Padding 方案（1~2 晚）

- [ ] 精读 [src/wmma/wmma_padding.cu](src/wmma/wmma_padding.cu)，与 wmma_base diff 对比：
  - [ ] shared memory 每行 padding 了几个 half？为什么是这个数？
  - [ ] padding 的代价是什么？（浪费了多少 shared memory？对 occupancy 有影响吗？）
- [ ] 跑 benchmark 记录提升

### 3.3 宽指令访存（1 晚）

- [ ] 在 wmma_base / wmma_padding 中找到从 global memory 加载时使用的宽指令（`int4` / `float4`，一次 128 bit）
- [ ] 回答：为什么一次读 16 字节比逐个读 2 字节的 half 快？和 coalescing 的关系是什么？

**检查点**：能画图解释"为什么给 shared memory 每行加 padding 就能错开 bank"。

---

## 阶段 4：异步拷贝与流水线（1~1.5 周）——本仓库精华

先读 [src/common/ptx.h](src/common/ptx.h)（44 行），认识四组宏：`CP_ASYNC_CA/CG`、`CP_ASYNC_COMMIT_GROUP`、`CP_ASYNC_WAIT_GROUP`、（阶段 5 会用到的 `LDMATRIX_*` 和 `HMMA16816`）。

### 4.1 cp.async：异步拷贝（1~2 晚）

- [ ] 阅读 PTX ISA 文档中 `cp.async` 部分，理解：
  - [ ] 传统 global→shared 拷贝为什么要"绕道"寄存器？`cp.async` 如何绕过？
  - [ ] `commit_group` / `wait_group N` 的语义：N=0 和 N=1 的区别？
  - [ ] `.ca`（cache all）和 `.cg`（cache global，绕过 L1）的区别？
- [ ] 精读 [src/wmma/wmma_async.cu](src/wmma/wmma_async.cu)，对比 padding 版本找出改动点
- [ ] 跑 benchmark 记录

### 4.2 双缓冲：Pg2s 和 Ps2r（2~3 晚）

- [ ] 精读 [src/wmma/wmma_async_pg2s.cu](src/wmma/wmma_async_pg2s.cu)：
  - [ ] shared memory 为什么要开两倍（双 buffer）？
  - [ ] 画出主循环的时间线：算第 i 块时，第 i+1 块的数据在哪里、在做什么？
- [ ] 精读 [src/wmma/wmma_async_pg2s_ps2r.cu](src/wmma/wmma_async_pg2s_ps2r.cu)：
  - [ ] Ps2r（shared→register 预取）又重叠了什么？fragment 也要双份吗？
- [ ] 跑 benchmark 记录

### 4.3 多级流水线：Stage2/3（2 晚）

- [ ] 精读 [src/wmma/wmma_async_stage2.cu](src/wmma/wmma_async_stage2.cu) 和 [wmma_async_stage3.cu](src/wmma/wmma_async_stage3.cu)：
  - [ ] stage N 意味着 shared memory 里同时驻留几块数据？`wait_group` 的参数怎么随 stage 变化？
  - [ ] stage 越深越好吗？受什么限制？（shared memory 容量 → occupancy）
- [ ] 跑 benchmark，整理一张表：naive → base → padding → async → pg2s → pg2s_ps2r → stage2 → stage3 各版本 TFLOPS

**检查点**：给你任意一个 stage 数，能画出对应的流水线时间线图，并说出 shared memory 用量。

---

## 阶段 5：MMA PTX 底层路径（1~1.5 周）——进阶

WMMA 把 fragment 布局藏起来了；MMA PTX 路径需要你自己控制每个线程从哪里加载数据，这是理解 cuBLAS/CUTLASS 级别优化的必经之路。

### 5.1 mma 指令与线程级数据布局（2~3 晚）

- [ ] 阅读 PTX ISA 文档 `mma.m16n8k16` 部分的线程-数据映射图（哪个 lane 持有 A/B/C 的哪些元素）
- [ ] 阅读 `ldmatrix` 指令文档：它一次为整个 warp 加载几个 8×8 矩阵？为什么和 mma 的布局"天然匹配"？
- [ ] 对照 [src/common/ptx.h](src/common/ptx.h) 的 `LDMATRIX_X1/X2/X4` 和 `HMMA16816` 宏，看懂内联 PTX 的输入输出约束
- [ ] 精读 [src/mma/mma_naive.cu](src/mma/mma_naive.cu) 和 [mma_base.cu](src/mma/mma_base.cu)

### 5.2 Permuted 布局消除 bank conflict（2~3 晚，本仓库最烧脑的部分）

- [ ] 精读 [src/mma/mma_permuted.cu](src/mma/mma_permuted.cu)：
  - [ ] 写入 shared memory 时地址做了什么异或/重排变换？
  - [ ] 为什么 permuted 比 padding 更好？（不浪费 shared memory）
  - [ ] 动手：在纸上画出一个 warp 写入/读取 shared memory 时各 lane 访问的 bank，验证无冲突
- [ ] 跑 benchmark 对比 mma_base vs mma_permuted

### 5.3 终极目标：stage4（2~3 晚）

- [ ] 按 mma_async → pg2s → pg2s_ps2r → stage2 → stage3 的顺序快速过一遍（结构和 wmma 系列同构，重点看差异）
- [ ] 精读 [src/mma/mma_async_stage4.cu](src/mma/mma_async_stage4.cu)，写一篇自己的注释版或笔记，覆盖：
  - [ ] 完整的 tiling 层次（block/warp/mma 指令三级）
  - [ ] 4 级流水线的时间线
  - [ ] permuted 布局的地址计算
  - [ ] L2 swizzle（README 提到的 block 到 C 矩阵位置的映射优化）在哪里、为什么能提高 L2 命中率
- [ ] 跑完整 benchmark：`./run_sample.sh`，确认 stage 系列在大尺寸下达到 cuBLAS 95%+

**检查点（结业标准）**：把 mma_async_stage4.cu 讲给别人听（或写成博客），对方能听懂数据从 global memory 到 Tensor Core 的完整旅程。

---

## 阶段 6（可选）：Profiling 验证与动手实验（3~5 晚）

- [ ] 用 Nsight Compute 对比几个版本：
  ```bash
  ncu --set full -o report ./output/bin/hgemm -M=4096 -N=4096 -K=4096 -profiling_iterations=1
  ```
  - [ ] 看 Tensor Core 利用率（`sm__pipe_tensor_op_hmma_cycles_active`）
  - [ ] 看 bank conflict 计数，验证 padding/permuted 确实消除了冲突
  - [ ] 看 occupancy 和 shared memory 用量随 stage 数的变化
- [ ] 动手改代码做实验（任选）：
  - [ ] 把 padding 大小改成别的值，观察性能变化
  - [ ] 把 stage3 改成 stage5，看是否还能编译/变快
  - [ ] 改 warp tile 尺寸，观察寄存器压力和性能
- [ ] 延伸阅读：CUTLASS 文档的 efficient GEMM 章节 —— 本仓库的手写优化在 CUTLASS 里都有模板化的对应物

---

## 实验记录表（边学边填）

| Kernel | M=N=K=4096 TFLOPS | 相对 cuBLAS | 关键优化点 |
|---|---|---|---|
| Cublas-Tensor-Op | | 100% | 基准 |
| Simt-Naive | | | 纯 CUDA core |
| Wmma-Naive | | | 首次用上 Tensor Core |
| Wmma-Base | | | + shared memory 复用 |
| Wmma-Padding | | | + 消 bank conflict |
| Wmma-Async | | | + cp.async |
| Wmma-Async-Pg2s | | | + global→shared 双缓冲 |
| Wmma-Async-Pg2s-Ps2r | | | + shared→register 双缓冲 |
| Wmma-Async-Stage2/3 | | | + 多级流水线 |
| Mma-Permuted | | | PTX + permuted 布局 |
| Mma-Async-Stage2/3/4 | | | 全部优化叠加 |

## 参考资料

- CUDA C++ Programming Guide — Warp Matrix Functions 章节
- PTX ISA 文档 — `mma`、`ldmatrix`、`cp.async` 指令
- NVIDIA Ampere GA102 架构白皮书
- CUTLASS 文档：Efficient GEMM in CUDA
- Nsight Compute 文档（profiling 指标含义）
