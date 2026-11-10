# ROCm Rust on gfx1030：硬件能力差距与软件替代方案研究档案

**仓库**: pifu-zh/cuda-oxide @ adapt-rocm（上游基线 fc747196f）
**硬件**: AMD RX 6950 XT（Navi 21 / gfx1030 / RDNA2：wave64 硬件、wave32 编译默认）
**配套**: 能力矩阵见 [gfx1030_support_matrix.md](../gfx1030_support_matrix.md)；实现代码 `cuda-oxide/crates/cuda-oxide-codegen/src/amdgcn.rs`（Rust prep pass）+ `cuda-oxide/tests/translate_amdgcn.py`（单一事实源改写器）+ `host-hip/cuda-hip-shim`（cu\*-ABI HIP shim）。
**方法**: 每项均溯源到源码/探针/真机数值验证；证据标签同矩阵（`[e2e]`/`[run]`/`[probe]`/`[hist]`/`[static]`）。

> 本档案是 2026-10-19 环境灾难后 **唯一** 的在库研究档案（原 `~/workspace/docs/research/` 全毁）。
> 蒸馏自幸存的 nv-amdgcn-map.md 注册表（develop-skill 仓库）与 gfx1030 分支 40 个 [PORT] 提交的验证记录。

## 0. 替代策略分类法（goal §7.3）

| 类别 | 含义 |
|---|---|
| NATIVE | 目标硬件原生支持 |
| INSTRUCTION COMPOSITION | 用现有指令组合实现 |
| ALGORITHM SUBSTITUTION | 换算法实现同一功能 |
| SOFTWARE EMULATION | 软件模拟缺失的操作/语义 |
| ALTERNATIVE BACKEND | 切换编程模型或 Kernel 后端 |
| NOT FEASIBLE | 已验证技术障碍（附证据） |

## 1. 研究来源与方法

- **aiter**（重点对象）：TMA 类异步搬运研究。关键发现：gfx1030 硬件有 `buffer_load→LDS 直通`通道（m0+lane\*4 扇出不占 VGPR）但 aiter 自己只在 gfx942 用了；其 gfx1030 路径（opus `make_gmem` `#else`）退化为同步寄存器中转——**"上游不用"≠"硬件没有"**，该结论由我们自己的汇编器探针/真机 flag 实验确立。
- **Triton / Composable Kernel / FlyDSL**：各自架构分支与回退路径（Triton prologue 寄存器预取=async_copy 0；CK static_assert 拒绝 gfx1030；FlyDSL s_waitcnt 编码差异实证）。
- **LLVM AMDGPU 后端**：llc `-mcpu=gfx1030` 指令选择边界（`Cannot select` = 能力信号，fail-loudly 纪律的基础）；汇编器负探针（如直通 LDS 仅 b32 宽度）。
- **闭环纪律**：残留审计 → 查 nv-amdgcn-map 注册表 → 无条目则探针实证 → 回填。注册表镜像在 develop-skill 仓库（外部），本档案为在库权威版。

## 2. 替代方案目录（按 NV 能力组织）

### 2.1 线程/同步原语

| NV 能力 | gfx1030 缺口与证据 | 替代策略 | 状态与证据 |
|---|---|---|---|
| `barrier0`（CTA 同步） | 无 nvvm 语义差异 | NATIVE（`llvm.amdgcn.s.barrier`） | ✅ [e2e] barrier_smoke 例 + 本机 batch |
| mbarrier（相位/计数） | 无硬件 mbarrier | SOFTWARE EMULATION：单字 i64 状态机（pending[31:0]/expected[47:32]/phase[63:48]，原子操作；expect_tx=vmcnt 部分排空等价，字节级不支持） | ✅ [e2e] 本机 pytest 三 kernel 语义断言（Step 16） |
| numbered/counted barrier | S_BARRIER 无编号，直映射死锁旁观 warp | SOFTWARE EMULATION：LDS 计数器+世代自旋，入口清零+s.barrier 全组前提；id 信封 0..15 | ✅ [hist] 128 线程含旁观 warp GPU PASS；[e2e] 本机 pytest |
| warp shuffle（idx/bfly/up/down × f32/i32/f64 + 动态 mask） | 无跨 lane 寄存器交换指令 | INSTRUCTION COMPOSITION：分段 ds_bpermute（NV lane 号→AMD 字节偏移×4；wave64 正确的 lane=mbcnt.lo+hi；f64=lo/hi 双程） | ✅ [e2e] 本机 pytest 全形态 + warp_reduce 例真卡 |
| redux.sync.add | 无硬件 redux | INSTRUCTION COMPOSITION：5 轮 xor-butterfly ds_bpermute | ✅ [e2e] redux_sum 例（全 warp 和 496） |
| laneid | 语义偏移 | INSTRUCTION COMPOSITION：mbcnt.lo+mbcnt.hi 两段 | ✅ [e2e] pytest |
| ntid sreg（blockDim） | AMDGPU 无 sreg 等价物 | SOFTWARE EMULATION：kernarg 尾随形参 x/y/z（宿主恒推三个） | ✅ [e2e] coords3d 3-D launch 24/24 + warp_reduce 77 形状 |

### 2.2 内存序/原子

| NV 能力 | 缺口与证据 | 替代策略 | 状态与证据 |
|---|---|---|---|
| isspacep.{local,shared,global} | 名字/语义差异（NV 局部=AMD private AS5） | INSTRUCTION COMPOSITION：amdgcn.is.{private,shared,global} 同名替换 | ✅ [e2e] atomics 例 |
| membar.{cta,gl,sys} + scope 名 | NV scope 名 AMD 后端不认 | INSTRUCTION COMPOSITION：fence + scope 映射（cta→workgroup，gl→agent，sys→系统域） | ✅ [e2e] 三 scope 断言 + [hist] MP litmus 200 轮 |
| core::sync::atomic 内嵌 PTX-asm 族 | AMDGPU 后端拒 PTX asm | INSTRUCTION COMPOSITION：LLVM atomic load/store/cmpxchg；seqcst asm 对折叠单条 seq_cst load | ✅ [e2e] pytest 12d/12e |
| panic `trap;` asm | PTX 助记符被拒 | INSTRUCTION COMPOSITION：llvm.trap → s_trap | ✅ [e2e] pytest |

### 2.3 数据搬运（aiter 研究线，详见 §3）

| NV 能力 | 缺口与证据 | 替代策略 | 状态与证据 |
|---|---|---|---|
| cp.async（gmem→smem 异步标量） | 无硬件 cp.async | INSTRUCTION COMPOSITION：软件流水（s_clause 批量+vmcnt(N) 部分排空+寄存器中转） | 🔶 已实现+量化；同步形态已转 SUPPORTED（能力矩阵 2026-11-03）；流水带宽数字见 §3（2026-11-04 诚实重测） |
| TMA / cp.async.bulk.tensor | RDNA2 无 TDM（真 TMA 等价物 gfx1250/RDMA4 才有） | 同上（双缓冲深流水）；OOB 填零=V# 描述符硬件行为（恒开） | ✅ [e2e] 2026-11-04 诚实重测：gmem→LDS 归约形态 553 GB/s=**96.1%**（完美替代档）；gmem→gmem 复制形态 435 GB/s=75.6%（良好档）；旧 78.4% 作废（测量伪影，见 §3） |
| TensorMap 硬件 swizzle | 无硬件 swizzle | ALGORITHM SUBSTITUTION：软件 padding（行距 272/144B 先例） | ✅ [hist] aiter mha 实证 |

### 2.4 数学/数值

| NV 能力 | 缺口与证据 | 替代策略 | 状态与证据 |
|---|---|---|---|
| idp4a（dp4a）同符号 | — | NATIVE：v_dot4（llvm.amdgcn.sdot4/udot4，clamp=false） | ✅ [e2e] dotprod 例四符号组合 |
| idp2a（dp2a） | 无单指令 | INSTRUCTION COMPOSITION：extract+sext/zext+乘加展开 | ✅ [e2e] dotprod 例 |
| libdevice `__nv_*` 数学 | NVVM bitcode 不入 AMD 模块 | ALTERNATIVE：`__ocml_*_f32/f64` 符号映射 + ocml.bc/oclc 配置链接 | ✅ [e2e] math_tan（≤2ULP [hist]） |
| PTX packed-math 内嵌 asm（bf16x2/f16x2/f32x2/b16 原子/ex2.f16） | gfx1030 无 v_pk_bf16（llc 探针拒）；f32x2 打包为 RDNA3；v_pk_f16 须 v 约束 | INSTRUCTION COMPOSITION：bf16x2→bf16→f32 移位展宽+f32 运算+RNE 回缩；f16x2 cvt→fptrunc/RTZ/relu clamp；f32x2→`<2 x float>` 向量 bitcast+fma.v2f32；b16 原子→LLVM 原子 i16 load/store；ex2.approx.f16→llvm.amdgcn.exp2.f32 | ✅ [e2e] 2026-10-27 **双实现 7/7**：Python（translate_amdgcn.py，llc-pass）与 Rust（amdgcn.rs prep pass，集成构建 df79fb980）同族同步转绿——atomic_f16/bf16x2_arith/bf16x2_fma/cvt_f16x2/cvt_packed/f32x2_arith/ex2_approx_f16 |
| 设备端 DWARF 调试信息 | 前端（MIR→IR）不产 !dbg metadata（.amdgcn.ll 计数=0、hsaco 零 .debug_* 节） | 路径已证可行：llc --debugger-tune=gdb 对带 !dbg IR 产 7 个 .debug_* 节 + DW_TAG_compile_unit/subprogram（scripts/probe_dwarf_gfx1030.sh） | IMPLEMENTABLE：剩余工作=自定义后端实现 debuginfo API（MIR DI 建图） |

### 2.5 硬件边界（NOT FEASIBLE 的证据基线）

| 能力 | 证据 | 判定 |
|---|---|---|
| MFMA 矩阵核 | [probe] gfx1030 缺 mai-insts（编译探针）；LLVM 拒绝 | HARDWARE_LIMITED；FFMA 回退先例 [hist]（DeepGEMM 路线 3.05 TFLOPS 功能正确） |
| WMMA | [probe] 缺 wmma-256b-insts | HARDWARE_LIMITED |
| fp8/fp4 硬件转换 | RDNA2 无（软件位操作解码可行=DeepGEMM 实证） | **✅ SUPPORTED（SOFTWARE EMULATION，2026-11-05；标量+向量打包双收口；2026-11-10 吞吐定档）**：E4M3FN/E5M2/E2M1 ↔ f32 纯软件位操作转换（cuda-device::fp8），真卡穷举验证——标量（256+256+16 pattern 解码 + 5180 编码值）+ **向量打包 x2 族 404,872 项零失败**（`__nv_fp8x2_e4m3`/`e5m2`/`__nv_fp4x2_e2m1` 语义：全 2^16/2^8 pattern × 2 lane 穷举解码 + 全 pattern 空间往返 + RNE spread lane 混合对编码，独立双实现比对，NaN 规范化经参考路径钉死）。**性能档（fp8_bench，triad 协议，copy 参照 516.9 GB/s）**：标量族 **91-98% = 90% 完美替代档**、packed encode ~100%、packed decode 81-82% = 80% 优秀档——被替代厂商路径（有转换单元的硬件）本就 memory-bound，软件转换贴近 copy 流即等价免费 | 本族无 CUDA 真卡对照（约定按 cuda_fp8.h 布局文档化+自一致验证）；fp8x4（u32 四值）可由 x2 组合未单独收口；packed decode 距 90% 线为微架构调优空间 |
| thread-block cluster / switch multicast | RDNA2 无（上游 CUDA 13 同样硬件门控） | HARDWARE_LIMITED；shim 显式 NOT_SUPPORTED |
| ~~直通 LDS 宽度仅 b32~~ | **[2026-11-04 推翻]**：重建微基准 ds_write_b128/ds_read_b128 全变体真机正确（旧负探针结论有误，原始探针脚本毁于 10-19 无法复核） | buffer_load 直通 LDS 仍以 b32 为 llc 实测上限；手写 ISA 可用 b128 |

### 2.5b 设备端 printf（vprintf → ockl hostcall）——2026-10-26 新增

| 项 | 内容 |
|---|---|
| 原始功能 | NVVM `vprintf(fmt, args)`（CUDA：参数包 marshaling 到缓冲区，host 侧格式化） |
| 原实现依据 | clang HIP 前端在源码级 lowering；NVVM IR 里就是 vprintf 调用+declare |
| gfx1030 缺口 | AMDGPU 无 vprintf 符号（hsaco `U vprintf` 未解析）；ROCm printf 走 **hostcall 协议** |
| 上游机制（取证） | 模块元数据 `amdgpu_printf_kind="hostcall"` → 运行时经 implicit kernarg 注入 hostcall buffer；ockl.bc 提供 `__ockl_printf_begin(i64)→i64` / `_append_string_n(h,ptr,len,islast)` / `_append_args(h,nargs,7×i64,islast)`（llvm-dis 反推 ABI） |
| 替代方案 | prep pass IR 级展开：fmt 须溯源到模块级 addrspace(1) constant（addrspacecast 或 {ptr,i64} 链）；args 须为 alloca（字段 i32/i8/i16→zext、i64 直通、f64→bitcast、f32→fpext+bitcast、[4×i8] packed load，共 7 槽）或 inttoptr-0（无参）；ockl.bc+oclc 三配置（isa_version_<n>/abi_version_600/finite_only_off）链接 |
| 分类 | INSTRUCTION COMPOSITION + ALTERNATIVE BACKEND（hostcall 服务协议） |
| 正确性验证 | [run] printf 例 8 测试族全过（"=== ALL PRINTF TESTS PASSED ==="），含 varargs 打包、线程索引输出、返回值语义（arg count 非 char count） |
| 成本 | 每 printf 调用 ~10 条指令展开 + hostcall 往返（host 侧格式化）；7 参数上限 |
| 剩余不确定性 | 动态 fmt 字符串（非常量全局）在信封外（保留 vprintf 链接期显式失败）；>7 参数 printf 需多轮 append_args（未实现，本批无此形态） |
| 值得实现？ | 是——解锁全部 GPU 侧调试输出这一通用能力；99 例批中 printf 族直接受益 |

### 2.6 闭包引用捕获（HMM 依赖）——本轮新增

| 项 | 内容 |
|---|---|
| 原始功能 | 非 move 闭包按引用捕获宿主变量；闭包结构体内嵌宿主指针，kernel 内解引用 |
| 原实现依据 | cuda-macros launch.rs 注释明示："non-move closures push the closure struct (which contains host references the GPU dereferences **via HMM**)" |
| gfx1030 缺口 | ROCm 10.1 对宿主栈内存无 HMM/SVM 一致访问；kernel 解引用宿主栈地址 → HSA_STATUS_ERROR_MEMORY_FAULT（故障地址=宿主栈 0x7ffc…，gdb+rocdevice 日志实证） |
| 上游处理 | 依赖 NVIDIA HMM（统一内存驱动能力）；未提供非 HMM 回退 |
| 替代方案 | **move 捕获**（拷贝语义）——同例 Test 1-8/10-13 十二种闭包形状全部真卡验证可用 |
| 分类 | NOT FEASIBLE（本卡 HMM 语义）；ALGORITHM SUBSTITUTION（move 捕获等价语义，功能可用、零性能损耗——按值进 kernarg） |
| Rust/工具链改动 | 无需（语言层已有 move 语法）；可考虑：编译器对 gfx 目标上的引用捕获闭包给诊断提示（后续项） |

## 3. 深度案例：soft TMA（aiter 技术提取 → 自研实现 → 量化）

**过程**（本节即 goal §7.5 的完整示范：研究不得停留在文档）：
1. **源码侦查**：aiter 仅 gfx942 用 buffer_load→LDS 直通；gfx1030 路径退化为同步寄存器中转（上游未利用的优化通道）。aiter 在 gfx1030 的实际做法 = s_clause 批量 + vmcnt(N) 部分排空软件流水。
2. **原语自证**（不信上游声明）：V# 描述符 OOB 硬件填零恒开（真机 8 组 flags）；vmcnt FIFO 序实证。~~直通 LDS 仅 b32~~ 已被 2026-11-04 重建微基准推翻（ds_write_b128/ds_read_b128 全变体真机正确；旧负探针脚本毁于 10-19 无法复核）。
3. **自研微基准**（2026-11-02 重建，2026-11-04 全规模闭环）：7 个手写 gfx1030 ISA kernel（plain1 串行参考 / df4_bN 深流水 gmem→VGPR→gmem 复制 / lds_bN 深流水 gmem→LDS→归约），8×255MiB 区域轮转 + 中位数协议。**历史 365 GB/s = 78.4% 数字作废**：该测量 kernel 的 prologue 块基址公式为 wg×qpb×16，正确值 = ×512（一个 quad = 32 lane × 16B = 512B）；32 路块切片重叠只覆盖 dst 前 9,165,312 B，其余未写——所报"带宽" = 9.16 MB footprint 上的事务速率（df4/lds ≈29× 虚高）且 L2 常驻，验证路径（lds 5% 容差、block-0 指纹）对 4000B 周期 pattern 的基址漂移盲视，仅 df4 memcmp（布局敏感）最终暴露。**2026-11-04 诚实数字（修正块基址 + 恢复 vmcnt(N) 流水 + 补齐计数器第二半增量后，7 变体 × 8 轮转 × 320 块全 verify ok）**：lds_bN = **553 GB/s = 96.1-96.2%（完美替代档）**；df4_bN = 435 GB/s = 75.6-75.7%（对经验上限，良好+）；**对厂商拷贝参照路径（hipMemcpy D2D 轮转中位 476.3 GB/s，2026-11-04 实测）= 91.4% → 达完美替代线**（替代语义 = 相对被替代的厂商路径；绝对口径仍有余量，深流水 store 路径调优列为可选开放项）；plain1 = 464 GB/s = 80.6%（优秀档）。分母 = **经验上限 575.5 GB/s**（256-bit @ 18G；纯读 98.3% + triad 90-95% + 本轮 lds 96.1% 三独立源共证；卡 0x73a5 / 16GB / vBIOS 名"6950 XT"但内存配置为 256-bit，ROCm props 的 memoryBusWidth=256 正确，spec-sheet 1152 不适用于本卡）。附带修正：微基准驱动 side 的 verify launch 需 memset→hipDeviceSynchronize→launch 屏障（大 fill 与 compute kernel 无屏障会竞态出洞）；pass 间需 hipDeviceSynchronize 防重叠。RDNA3 dlds / RDNA4 TDM 硬件出现后同基准复跑。
4. **意外发现**：LDS 消费者越界伪装硬件竞态（须线性恒等映射）；LLVM IV 链模 2³² 重关联致 MUBUF 基址 wrap（num_records clamp 解）；显式 vmcnt 后 LLVM 不插额外排空。
5. **升级条件**：RDNA3 dlds / RDNA4 TDM 硬件出现后同基准复跑。

**损失说明**：原 tma-like.md/soft-tma.md 档案与基准脚本毁于 2026-10-19；本节数字来自幸存注册表。重建时以本节设计要点为准。

## 4. 开放研究项（按价值排序，2026-11-02 刷新）

已收官（原 1-5 全清）：
1. ~~async launch 挂死调查~~ **已解决（2026-10-25）**：三层根因逐一定位并修复——(a) 段错误=sync 原生宏缺 ntid 追加；(b) 全零输出=ntid 局部变量声明在 if 块内、push 裸指针悬垂；(c) HSA 宿主栈地址故障=Test 9 非 move 引用捕获依赖 HMM（§2.6）。终局：host_closure 十二形状×四路径 48+ launch 真卡全绿。
2. ~~设备端 printf~~ **已解决（2026-10-26/28）**：vprintf→ockl hostcall 展开双侧落地，八测试族+>7 参链式真卡零乱码（§2.5b）。
3. ~~grid_constant~~ **已解决**：embedded loader gfx 直载，转 FULL。
4. ~~batch 扩表~~ **已完成**：207 候选扫描收官，EXAMPLES 105 例，容器批 101/105+3 已知排除（2026-11-01）。
5. ~~性能基线重建~~ **已收束（2026-11-01）**：§8 90% 线双指标达标（triad 90-95%、FFMA 92.3%），归因入矩阵 §8 更新块。

**软件替代性能成绩梯度（2026-11-02 用户定档）**：硬件不支持、软件替代的功能，相对参照路径——**60% 及格 / 70% 良好 / 80% 优秀 / 90% 完美替代 / 超硬件理论 = 超级大发现**；功能可行性与性能可行性分开报告的纪律不变（§7.6）。现有定档：soft TMA **78.4% = 良好**（收线）；其余软件替代（mbarrier/shuffle/redux/编号 barrier）功能已证、性能档待量化。

当前开放（按价值）：
- ~~**pinned host 内存适配**~~ **已收官（2026-11-02）**：H2Dp 27.6/D2Hp 28.8 GB/s = 2.0-2.1×，重叠流水 2.25×。
- ~~**DWARF 调试信息**~~ **已收官（2026-11-05/11-09）**：Full + LineTables 双档卡上验证（矩阵 SUPPORTED）。
- ~~**panic 消息格式化**~~ **已收官（2026-11-09）**：kernel 宏源级重写 `panic!(fmt, args...)` → gpu_printf 机制 → ockl stdout（真卡渲染实证），字面量路径保持 as4 回读通道；双通道语义入矩阵。
- ~~**fp8/fp4 软件解码通用化**~~ **已收官（2026-11-05）**：标量+向量打包 x2 双收口，404,872 项穷举零失败。

## 5. 维护规则

- 新增替代实现必须：实现 → 独立正确性测试 → 真机证据 → 本目录加行 → 矩阵同步。
- 新识别的硬件边界必须带探针证据（汇编器负探针/编译探针/真机 flag 实验）。
- 每次类别变更（如 IMPLEMENTABLE→SUPPORTED）须引用具体测试/提交。
- nv-amdgcn-map.md（develop-skill 镜像）与本档案双向同步；档案内死引用须标注。
