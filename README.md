# OpenRVGPU

# 1 概述

OpenRVGPU是一个具备原生非对称FP8支持的开源GPU，兼容第5代张量核心指令集（tcgen05）。



# 2 OpenRVGPU架构与ISA





## 2.1 OpenRVGPU整体架构

![img](https://tensor03.cn6.quickconnect.cn/direct/oo/file/1262_KVUTLEK5014V7D8CA8P0TP2Q04.doc/6745IBEU7H5KBAD5HKO10A1624/img?tid=%22GO2gwhFfeQSCaMOQ824aTSjDRNdXeJ938FzRoXzfg3_LFlfy_CbPpCj76CpPeh4PSumpJ7rw8b3kjpO0%22&linkId=%2217sCFxTt1MoBkABrZNcdn0sLUmEnCf4b%22)



从架构角度看，当前 Cmodel 的核心目标不是复用 NVIDIA PTX 的文本语法或二进制编码，而是在 RISC-V custom 指令空间中重建 tcgen05 的软件可见抽象。

## 2.2 OpenTenorCore ISA 

## 2.3 OpenRVGPU ISA

### 2.3.1 以开源GPU ISA为基础

Vortex / Ventus 基础ISA

### 2.3.2 OpenTenorCore ISA 

以下是完整的兼容tcgen05的RISC-V扩展指令集，主要用于描述：

1）TMEM 如何作为 Tensor Core 专用 scratchpad 被软件显式管理。

2）MMA 如何通过 instruction descriptor 描述精度、形状和稀疏属性。

3）异步操作如何通过 fence、commit、wait 和 mbarrier 建立完成与可见性关系。

Vortex 则把这些语义重新编码到 RISC-V R-type custom 指令中，使 tensor 扩展可以被现有 RISC-V decode、issue、execute 框架接纳。







| 字段   | 位段    | 位宽 | 作用                           |
| ------ | ------- | ---- | ------------------------------ |
| opcode | [6:0]   | 7    | 使用custom1和custom2自定义扩展 |
| rd     | [11:7]  | 5    | 目标寄存器                     |
| funct3 | [14:12] | 3    | 区分具体操作                   |
| rs1    | [19:15] | 5    | 源寄存器1                      |
| rs2    | [24:20] | 5    | 源寄存器2                      |
| funct7 | [31:25] | 7    | modifier                       |



### 2.3.3 TMEM/TMA ISA



| 指令              | opcode  | funct7    | funct3 | rd   | rs1          | rs2                     | PTX对应                                          |
| ----------------- | ------- | --------- | ------ | ---- | ------------ | ----------------------- | ------------------------------------------------ |
| TMA_REL_PERMIT    | 0101011 | qualifier | 001    | x0   | x0           | x0                      | tcgen05.relinquish_alloc_permit                  |
| TMA_ALLOC         | 0101011 | qualifier | 010    | x0   | dst_addr     | nCols                   | tcgen05.alloc                                    |
| TMA_DEALLOC       | 0101011 | qualifier | 011    | x0   | taddr        | nCols                   | tcgen05.dealloc                                  |
| TMA_CP            | 0101011 | qualifier | 100    | x0   | taddr        | s_desc                  | tcgen05.cp                                       |
| TMA_SHIFT         | 0101011 | qualifier | 101    | x0   | taddr        | x0                      | tcgen05.shift.down                               |
| CPABULK_TENSOR_LD | 0101011 | qualifier | 110    | x0   | tensor_map_t | cpabulk_transfer_args_t | cp.async.bulk.tensor.<N>d.shared::cluster.global |



**字段含义：**



| 字段                    | 含义                                                         | tcgen05依据                                                  |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| dst_addr                | 写回分配结果 taddr 的目的地址                                | tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32 [dst], nCols；PTX描述是把分配得到的 Tensor Memory 地址写到 dst 指向的位置 |
|                         |                                                              |                                                              |
| nCols                   | TMEM中分配的逻辑列数                                         | nCols 是 alloc/dealloc 的列数操作数；分配/释放单位是 32 columns |
| s_desc                  | TMA搬运的数据在shared memory的layout描述符。                 | tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc；这里 PTX 明确写的是 s-desc |
| taddr                   | TMEM 基地址/分配句柄，用于指向先前 alloc 得到的 Tensor Memory 区域；dealloc/cp/shift 都以它作为 TMEM 目标或操作对象 | tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc； |
| tensor_map_t            | 指向内存中 tensor-map 的 128B opaque object，tensor-map容纳了 5 维 tensor 所需的 global base address、box size、global stride、element stride、rank、element type、interleave、swizzle、L2 promotion 和 OOB fill 等字段 | 对标 PTX cp.async.bulk.tensor 使用的 tensor-map/CUtensorMap 概念，NVIDIA PTX 中 tensor-map 是 128B opaque object，可放在 .const、.param 或 .global 空间 |
| cpabulk_transfer_args_t | Lmem中存放Vortex 自定义的一个 32B LMEM 参数块的地址。该参数块保存 LMEM 的destination/source 地址、最多 5 维 tensor 坐标，以及 load 指令在qualifier选择 .mbarrier::complete_tx::bytes 时需要绑定的 mbarrier 地址。 | cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, {tc0, tc1}], [mbar]; |





**qualifier****:**

| 指令           | 字段                                                         | 含义                                                         | tcgen05依据                                   |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------- |
| TCU_REL_PERMIT | qualifier[0]                                                 | CTA group 选择位**cta_group：** 0 = .cta_group::1, 1 = .cta_group::2 | .cta_group = { .cta_group::1, .cta_group::2 } |
| qualifier[6:1] | ——                                                           | reserved                                                     |                                               |
| TCU_ALLOC      | qualifier[0]                                                 | CTA group 选择位**cta_group：** 0 = .cta_group::1, 1 = .cta_group::2 | .cta_group = { .cta_group::1, .cta_group::2 } |
| qualifier[1]   | dst 地址空间选择位**shared::cta：** 0 = generic, 1 = shared::cta | {.shared::cta}；完整语法是 tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32 [dst], nCols |                                               |
| qualifier[6:2] | ——                                                           | reserved                                                     |                                               |
| TCU_DEALLOC    | qualifier[0]                                                 | CTA group 选择位**cta_group：** 0 = .cta_group::1, 1 = .cta_group::2 | .cta_group = { .cta_group::1, .cta_group::2 } |
| qualifier[6:1] | ——                                                           | reserved                                                     |                                               |
| TCU_CP         | qualifier[0]                                                 | CTA group 选择位**cta_group：** 0 = .cta_group::1, 1 = .cta_group::2 | .cta_group = { .cta_group::1, .cta_group::2 } |
| qualifier[3:1] | copy shape / multicast 编码**shape/layout：** 128x256b, 4x256b, 128x128b, 64x128b.warpx2，32x128b.warpx4... | .shape = { .128x256b, .4x256b, .128x128b, .64x128b, .32x128b } |                                               |
| qualifier[5:4] | 可选解压格式编码**decompress：** none, b6x16_p32, b4x16_p64  | {.dst_fmt.src_fmt}，其中 .src_fmt = { .b6x16_p32, .b4x16_p64 }，.dst_fmt = { .b8x16 }；也就是可选解压格式 |                                               |
| qualifier[6]   | ——                                                           | reserved                                                     |                                               |
| TCU_SHIFT      | qualifier[0]                                                 | CTA group 选择位**cta_group：** 0 = .cta_group::1, 1 = .cta_group::2 | .cta_group = { .cta_group::1, .cta_group::2 } |
| qualifier[6:1] | ——                                                           | reserved                                                     |                                               |







### 2.3.4 同步/控制（MBAR）ISA





| 指令               | opcode  | funct7    | funct3 | rd    | rs1       | rs2         |
| ------------------ | ------- | --------- | ------ | ----- | --------- | ----------- |
| MBAR_FENCE         | 1011011 | qualifier | 000    | x0    | x0        | x0          |
| MBAR_COMMIT        | 1011011 | qualifier | 001    | x0    | mbar_addr | ctaMask     |
| MBAR_INIT          | 1011011 | qualifier | 010    | x0    | mbar_addr | count       |
| MBAR_ARRIVE        | 1011011 | qualifier | 011    | x0    | mbar_addr | count_or_tx |
| MBAR_EXPECT_TX     | 1011011 | qualifier | 100    | x0    | mbar_addr | txCount     |
| MBAR_COMPLETE_TX   | 1011011 | qualifier | 101    | x0    | mbar_addr | txCount     |
| MBAR_WAIT          | 1011011 | qualifier | 110    | x0    | mbar_addr | phase_token |
| MBAR_TEST_TRY_WAIT | 1011011 | qualifier | 111    | state | mbar_addr | phase_token |





**字段含义：**



| 字段        | 含义                                                         | tcgen05依据                                                  |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| mbar_addr   | mbarrier对象地址。供 TCU_COMMIT 指定要跟踪完成事件的 mbarrier。 | tcgen05.commit ... [mbar]；                                  |
|             |                                                              |                                                              |
| ctaMask     | 目标 CTA 位掩码。仅在 TCU_COMMIT 带 .multicast::cluster 时有效。 | tcgen05.commit ... [mbar] {, ctaMask}；                      |
| phase_token | mbarrier 相位token，表示此时WAIT的mbarrier处于哪个phase      | mbarrier.arrive... state, [addr] 返回 opaque 64-bit phase state；mbarrier.test_wait/try_wait ... [addr], state 用该 state 测完成 |
| txCount     | 预期异步事务计数。这是 MBAR_ARRIVE.expect_tx 的操作数字段，表示在 arrive 之前先对 mbarrier 做一次 expect-tx(expectCount=txCount)，让 barrier 额外跟踪后续异步事务完成数。 | mbarrier.arrive.expect_tx... [addr], txCount 中 txCount 是 expectCount |
| count_or_tx | “到达计数 / 事务计数”复用字段，qulifier字段中expect_tx为0时，表示 arrive-on 的 count；qulifier字段中expect_tx为1时，也就是MBAR_ARRIVE.expect_tx 模式，此时count_or_tx表示 txCount。 | mbarrier.arrive... {, count} 中 count 指定 arrive-on 的 count;mbarrier.arrive.expect_tx... txCount 指定 expect-tx 的 expectCount；tcgen05.commit 是 .mbarrier::arrive::one。 |







**qualifier****:**

| 指令           | 字段                                                         | 含义                                                         | tcgen05 PTX依据                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MBAR_FENCE     | qualifier[0]                                                 | 面向异步 tcgen05 操作与 thread-scope 执行序同步之间的专用顺序栅栏**before/after_thread_sync:**0=before_thread_sync, 1=after_thread_sync | tcgen05.fence::before_thread_sync / tcgen05.fence::after_thread_sync； |
| qualifier[6:1] | ——                                                           | reserved                                                     |                                                              |
| MBAR_COMMIT    | qualifier[0]                                                 | CTA group 选择位**.cta_group：** 0 = .cta_group::1, 1 = .cta_group::2 | tcgen05.commit.cta_group::1                                  |
| qualifier[1]   | mbar 操作数地址空间限定符选择位**.shared_cluster：**0 = generic, 1 = .shared::cluster | tcgen05.commit...{.shared::cluster}{.multicast}.b64 [mbar]； |                                                              |
| qualifier[2]   | multicast 使能位。**multicast：** 0 = no multicast, 1 = .multicast::cluster | .multicast::cluster                                          |                                                              |
| qualifier[6:3] | ——                                                           | reserved                                                     |                                                              |





当前 Cmodel 将 tensor 指令划分成三组。custom-1 负责 TMEM 管理和 tensor data movement，custom-2 负责 tcgen05 风格的同步与 mbarrier，custom-3 负责 TensorCore compute 以及 TMEM 与寄存器之间的显式读写。

这个划分的好处是，每一组指令的语义边界比较清楚：custom-1 更接近数据搬运层，custom-2 更接近异步完成和同步层，custom-3 则是计算层。RISC-V 指令中的 funct3 用于选择同一组内的具体操作，funct7 则作为 qualifier，用来承载 PTX 中 .sp、.ws、.cta_group、.multicast、before/after 等修饰语。

TCU_MMA是整个指令映射中最关键的一条。PTX tcgen05.mma 的操作数数量较多，包含 D 的 TMEM 地址、A 操作数、B shared-memory descriptor、instruction descriptor、predicate、以及多个 qualifier。单条 RISC-V R-type custom 指令无法直接容纳这些字段，所以 Vortex 采用“两级承载”的方式：[rs1] 直接携带 32-bit idescriptor_t，[rs2] 指向 LMEM 中的 32B operand_block_t，funct7 则承载高频 qualifier。这样既保持了 RISC-V 指令格式的固定性，也保留了 tcgen05 instruction descriptor 的语义中心。



所有TCU指令枚举如下：

enum class TcuType {

 // ---- custom-1 (EXT2, 0x2B): TMEM management + cp.async.bulk.tensor ----

 TMEM_REL_PERMIT,   // tcgen05.relinquish_alloc_permit

 TMEM_ALLOC,     // tcgen05.alloc

 TMEM_DEALLOC,    // tcgen05.dealloc

 TMEM_CP,       // tcgen05.cp (shared-mem -> TMEM)

 TMEM_SHIFT,     // tcgen05.shift.down

 CPABULK_TENSOR_LD,  // cp.async.bulk.tensor.<N>d (DRAM -> shared)

 CPABULK_TENSOR_ST,  // cp.async.bulk.tensor.<N>d (shared -> DRAM)



 // ---- custom-2 (EXT3, 0x5B): tcgen05 sync + full mbarrier ----

 MBAR_FENCE,     // tcgen05.fence::{before,after}_thread_sync

 MBAR_COMMIT,     // tcgen05.commit

 MBAR_INIT,      // mbarrier.init / mbarrier.invalidate

 MBAR_ARRIVE,     // mbarrier.arrive{.expect_tx,.arrive_drop}

 MBAR_EXPECT_TX,   // mbarrier.expect_tx (standalone)

 MBAR_COMPLETE_TX,  // mbarrier.complete_tx

 MBAR_WAIT,      // mbarrier.wait (blocking, no timeout)

 MBAR_TEST_TRY_WAIT, // mbarrier.test_wait / mbarrier.try_wait



 // ---- custom-3 (EXT4, 0x7B): tcgen05 compute ----

 TCU_MMA,       // tcgen05.mma{.ws,.sp}

 TCU_LD,       // tcgen05.ld (TMEM -> RF)

 TCU_ST,       // tcgen05.st (RF -> TMEM)

 TCU_WAIT_LD,     // tcgen05.wait::ld

 TCU_WAIT_ST,     // tcgen05.wait::st



 // Phase-2: legacy MMA_LOAD / MMA_STORE / WMMA internal routing tags removed;

 // TCU_MMA is now the single ISA-visible compute entry. Internal fan-out is

 // managed by TensorAsyncFrontend without per-stage TcuType discrimination.

};