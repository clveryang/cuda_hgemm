# wmma_base 逐段解读：数据复用与三级分块

> 对应源码：[src/wmma/wmma_base.cu](../src/wmma/wmma_base.cu)。前置阅读：[wmma_naive 逐行解读](wmma_naive.md)、[内存布局基础](memory_layout.md)。

wmma_base 是从"能用 Tensor Core"到"用好 Tensor Core"的关键一跳，核心思想就一个：**数据复用**。

## 药方：数据复用的账本

naive 的病根：C 同一行的每个 warp 都独立从 global memory 读同一条 A 条带，A 被重复读 N/16 次。wmma_base 的药方是**拼团**：

- 一个 block 管 C 的 **256×128** 大瓦片（8 个 warp 拼团）
- A 的条带先搬进 **shared memory**，block 内 8 个 warp 共享——A 的重复读取次数从 N/16 降到 **N/128**（8 倍削减），B 从 M/16 降到 **M/256**（16 倍削减）
- 每个 warp 的任务从 1 块 16×16 涨到 **64×64**（16 块 wmma 瓦片），`C_frag[4][4]` 常驻寄存器，摊薄调度开销

![wmma_base 三级分块](images/wmma_base_tiling.svg)

## grid 与 block 怎么划

启动代码（第 197-198 行）：

```cpp
dim3 block(THREADS_PER_BLOCK);   // block 是一维的：256 个线程
dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));
//        = grid(16,    M/256,                   N/2048)
```

直觉上你会期待 `grid(N/128, M/256)`——C 有多少个 256×128 瓦片就开多少个 block。功能上确实如此，但作者把 N 方向拆成 `16 × N/2048` 两截（x 和 z 维度），**纯粹为了控制 block 的发射顺序**。

kernel 里把硬件坐标翻译回逻辑瓦片坐标（第 53-55 行）：

```cpp
// M 方向第几个行块：z 为奇数时行序反转（蛇形）
block_tile_i = (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * 16) : (blockIdx.y * 16);
// N 方向第几个列块：z 和 x 合并回完整列号
block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * 8;
```

（单位是 16×16 小瓦片数，所以乘 16 和 8——一个 block 竖着盖 16 个、横着盖 8 个小瓦片。）

代入 M=N=4096：`grid(16, 16, 2)`，共 512 个 block。GPU 发射 block 的顺序 x 最快、y 次之、z 最慢，于是：

1. z=0：先横着铺 x=0..15（覆盖 2048 列宽），y 递增——在一条 2048 列宽的竖条纹里从上往下扫
2. z=1：换右半边条纹，y 反向（蛇形），从下往上扫

同一时刻在跑的 block 落在 C 的同一横排附近，读同一条 A 条带——第一个 block 把 A 读进 L2，其余 block 吃缓存；条纹换向时蛇形衔接，上一条纹末尾的 A 还热着就被接着用。这就是 README 里 "L2 Cache: swizzle" 的全部内容。注意硬件不保证 block 执行顺序，这只是统计上让时间相近的 block 数据相近。

block 内部是纯一维 256 线程，自己切 warp：

```cpp
warp_id = threadIdx.x / 32;   // 0..7，按 4 行 2 列铺在 256×128 瓦片上
lane_id = threadIdx.x % 32;   // 0..31，warp 内编号
```

完整坐标链：**(blockIdx.x,y,z) → 哪块 256×128 → warp_id → 哪块 64×64 → mma 循环的 (i,j) → 哪块 16×16 → lane_id → fragment 里的哪 8 个数**。

## C_frag 是什么

一句话：**C_frag 是这个 warp 的"结果账本"——它负责的 64×64 输出区域的累加值，全程住在寄存器里。**

```cpp
wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag[4][4];
```

- 单个 `fragment<accumulator,...>` 对应一块 16×16 输出瓦片的 256 个累加值，摊在 warp 的 32 个线程的寄存器里，**每线程扛 8 个**。哪个线程扛哪 8 个是黑盒，所以不能 `C_frag[i][j][5]` 碰单个元素，只能整体填零、整体累加、整体写出。
- 数组开成 `[4][4]`：warp 管 64×64 = 4×4 块瓦片，一块一本账。
- 全 warp 合计 16 本账 × 每线程 8 个 = 每线程 128 个 half ≈ 64 个寄存器——寄存器压力的主要来源。

关键是生命周期的对比：

| | 创建 | 存活时间 | 死亡 |
|---|---|---|---|
| `A_frag` / `B_frag` | 每个 k_step 里 | 用一次就扔 | 循环体结束 |
| `C_frag[4][4]` | kernel 开头填零 | **整个 K 循环** | 最后写回才落地 |

A、B 像流水一样过境，C 钉在寄存器里不动——这种数据流叫 **C-stationary**（输出驻留），是几乎所有高性能 GEMM 的共同选择：C 的每个元素要被累加 K 次，把被写最多次的数据放在最快的存储（寄存器）里，只在最后一刻才碰一次显存。

## warp 的职责变化：计算 vs 搬运

一个常见的理解："warp 负责的计算和搬运从 16×16 变成 64×64 了"——**计算上对，搬运上机制变了**，这个区别恰恰是理解 base 的关键。

**计算**：每个 warp 负责的 C 输出从 16×16 涨到 64×64（16 块 wmma 瓦片），按输出瓦片划分地盘。

**搬运**：naive 里每个 warp 自己搬自己要算的数据，自产自销。base 里搬运和计算是**两份独立的分工**：

- 搬运按"平均分摊"切：block 每轮要搬的公共条带是 A 256×32 + B 128×32，按 warp_id 平均切成 8 份——warp w 搬 A 的第 `w*32 ~ w*32+31` 行、B 的第 `w*16 ~ w*16+15` 行
- 计算按"输出瓦片"切：warp w 算 C 的第 `(w/2)*64` 行起的 64×64 区域

拿 warp 0 对账：它**搬**的是 A 的 0-31 行，但它**算**的时候要用 A 的 0-63 行——其中 32-63 行是 warp 1 搬进来的；反过来它搬的东西也有一半是给别的 warp 用的。**搬运是"为集体上货"，计算是"各吃各的地盘"，两者通过 shared memory + `__syncthreads()` 解耦**——这也是搬完必须同步的原因：你要用的数据可能在别人手里还没搬完。

把量算清楚（每轮外循环、每 warp）：

| | 搬运量（global→smem） | 消费量（smem→fragment） |
|---|---|---|
| A | 32 行 × 32 = 1024 个 half（2KB） | 64 行 × 32 = 2048 个 |
| B | 16 行 × 32 = 512 个（1KB） | 64 列 × 32 = 2048 个 |

消费大于搬运——差额就是共享的红利：每个数据搬进来一次，被多个 warp 消费。这正是 global memory 流量下降 8×/16× 的微观来源。

一句话总结：**每个 warp 的计算地盘从 16×16 涨到 64×64；搬运则从"自产自销"变成"集体上货、按需取用"，人均搬运量反而比消费量小——省下的就是性能。**

## 宏定义解码（第 13-43 行）

开头一堆宏就是三级分块的参数表，抓住三行：

```cpp
#define BLOCK_ROWS 256    // block 级：C 的 256×128
#define BLOCK_COLS 128
#define WARP_ROWS 64      // warp 级：4×2 排布的 8 个 warp，各管 64×64
#define WARP_COLS 64
#define CHUNK_K 2         // K 方向每轮外循环吃 2 块瓦片 = 32 个元素
```

其余宏全是这三者的派生（每次搬多少行、几个线程搬一行等），看不懂时代入数字算一遍即可。

## kernel 三阶段

### 阶段 1：协作搬运 global → shared（第 93-120 行）

要搬的货：A 条带 256 行 × 32 列 + B 条带 128 行 × 32 列。B 是列主序，它在 smem 里的"一行"其实是某个 N 列的 32 个 K 值——同一份拷贝代码对 A、B 通吃，这是选列主序的又一好处。搬法：

- 每个线程一次搬 `int4`（16 字节 = 8 个 half）——README 说的 **wide instruction** 就是它
- 一行 32 个 half = 64 字节 = 4 个 int4 → **4 个 lane 合搬一行**，一个 warp 一次搬 8 行
- A 有 256 行，8 个 warp 分——每个 warp 负责 32 行，循环 4 轮（`A_smem_iters = 4`）搬完；B 同理 2 轮

搬完 `__syncthreads()`：等所有 warp 把货上齐才能开算。

### 阶段 2：计算（第 123-152 行）

每个 warp 从 shared memory 加载自己那 4 块 A 瓦片、4 块 B 瓦片（注意 leading dimension 变成了 32——smem 里条带就这么宽），然后 4×4 = 16 次 `mma_sync`。第 147 行有个彩蛋：

```cpp
size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;
```

遍历顺序不是逐行从左到右，而是**蛇形**：第 0 行左→右，第 1 行右→左……换行时 B_frag 不用换（上一行最后用的就是下一行第一个要用的），寄存器里的值接着用——README 里 "Register Reuse: Right Left Right Left" 说的就是这行代码。

### 阶段 3：写回也要绕道 shared memory（第 157-173 行）

naive 直接 `store_matrix_sync` 到 global memory，写的是 16×16 的小碎块，合并度差。base 先把 16 块 C_frag 全写进 shared memory（复用 AB 条带的空间，此时它们已没用），拼成完整连续的 256×128，再由每个 warp 用 int4 流式写出——每次写 256 字节连续内存，**完美合并**。宁可多一次 smem 往返，也要让 global memory 写入是大块连续的。

## 两个易忽略的细节

- **shared memory 用量 64KB**（C 中转需要 256×128×2B），超过默认的 48KB 静态上限，所以 `initWmmaBase()` 里要 `cudaFuncSetAttribute` 显式申请动态 shared memory——自己写大瓦片 kernel 忘了这步会直接 launch 失败
- **grid 蛇形映射**（第 53-55 行 + `BLOCK_STRIDE`）：block 在 C 上的行走顺序被刻意安排成 S 形蛇行，相邻发射的 block 尽量踩相邻的 A/B 条带，提高 **L2 缓存**命中率——README 的 "L2 Cache: swizzle access mode" 就在这
- 隐含约束：M、N、K 需分别被 256/128/32 整除，kernel 没做残块处理（benchmark 尺寸都满足）

## 它离 cuBLAS 还差什么

两个"串行等待"没解决：

1. 搬运和计算之间隔着 `__syncthreads()`，搬的时候算力闲着，算的时候搬运闲着——**async / pg2s / stage 系列**就是来重叠这两者的
2. smem 无 padding，`load_matrix_sync` 读条带时存在 **bank conflict**——**wmma_padding** 只改一个数组声明就能提速

## 检查点

1. 推导 `A_smem_iters = 4`：256 行、8 个 warp、每 warp 每轮 8 行，怎么算出来的？
2. 写回阶段为什么不直接 `store_matrix_sync` 到 global memory？
3. 蛇形 `j_s` 到底省了什么？如果去掉会错吗，还是只会慢？
4. 把 `BLOCK_ROWS` 改成 512 会发生什么？（提示：算算 shared memory）
