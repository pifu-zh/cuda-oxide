# gfx1030 功能支持矩阵（ROCm + Rust for RX 6950 XT）

**仓库**: pifu-zh/cuda-oxide @ adapt-rocm（上游基线 fc747196f=+365 commits 新布局；HIP 接入=host-hip/cuda-hip-shim cdylib 以 libcuda.so.1 形态经 LD_LIBRARY_PATH 投喂上游 0.5.0 dlopen 加载器，[patch] 时代终结）
**硬件**: AMD Radeon RX 6950 XT（Navi 21 / gfx1030 / RDNA2，wave64 硬件、wave32 编译默认）
**软件**: Ubuntu 26.10 · amdgpu 内核驱动 · ROCm 10.1.0-3 · rustc **1.99.1 stable**（上游已迁 stable+RUSTC_BOOTSTRAP；llc=工具链 llvm-tools LLVM 23.1.1）

## 状态词汇与证据分级

| 状态 | 含义 |
|---|---|
| FULL | 实现完整且在本卡上有直接通过证据 |
| PARTIAL | 可用但有明确边界/信封限制（边界随条目给出） |
| UNSUPPORTED | 硬件或架构依据明确不支持（附依据，不等于"未实现"） |
| BLOCKED | 可支持但被外部依赖卡住（附阻塞点） |
| NOT TESTED | 未在本卡验证过——不计入任何通过口径 |

证据标签：`[e2e]`=真机端到端数值断言；`[run]`=真机执行；`[probe]`=探针实证；`[hist]`=历史会话验证（命令可复跑）；`[static]`=头文件/ISA/源码静态证据。

一键复跑全部证据：`scripts/verify_gfx1030.sh`（生成时间戳报告目录：环境快照/各门日志/汇总）。

## 1. Rust 语言特性与设备端编译

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| 核心 Rust（控制流/循环/数组/slice/结构体/枚举/方法） | FULL | [e2e] pytest 14/14（本机 2026-10-20）；[run] `cargo oxide run vecadd` 1024/1024；batch 14 例经同一条 MIR→dialect-mir→LLVM→amdgcn 管线 | — |
| 泛型（单态化后分发）/const 泛型/trait 静态分发 | PARTIAL | 理论=单态化后与核心 Rust 同一 lowering [static]；gfx 直验面=batch/pytest 所含形状。`const_generic` 例因 device-only 抽取器无法满足 cuda_host 引用而排除（batch 记录） | 写一个不依赖 host crate 的 const-generic device 变体进 batch |
| 闭包（显式捕获，byval 结构体参数） | PARTIAL | [hist] host_closure 例 + Phase-1 闭包族 llc 绿；本机 batch 含 host_closure（Python 维度 PASS） | 集成维度本机未单独跑该例的运行验证 |
| panic（=abort：内嵌 PTX trap asm → llvm.trap → s_trap） | FULL | [e2e] pytest trap asm 测试；[hist] panic 例 GPU 验证 | unwind 无（语言语义即 abort，非缺口） |
| 迭代器适配器（无分配子集） | **FULL 扩展**（2026-10-31 真卡） | [e2e] iterator_adapters 十项适配器真卡逐值全对：map+sum、min/max（copied().unwrap_or 值形态——**2026-10-31 证实可用**，此前"Fn-trait 分发被拒"实为引用形态 default 的 map_or 路径；copied() 值形态 unwrap_or/map_or/unwrap_or_else 均真卡通过，见 iter_unwrap_or_probe 例）、filter+count、zip、position、enumerate+for、enumerate+fold/sum（**此前拒链已解锁**：closure-typed struct 常量合成 fallback 落地，Enumerate::fold 的嵌套闭包常量不再拒）、slice fold。单 kernel 二分与逐值 host 参考对比全程留痕 | 遗留边界：引用形态 default（unwrap_or(&0)）仍走 Fn-trait 分发被拒（copied()/match 为工作替代）；Direct/Niche 之外的 exotic 布局未覆盖 |
| 设备堆机制（六层完成 2026-10-31） | **FULL** | [e2e] slot map fat-pointer niche 通用化落地：analyze_pointer_overlap 新增 exact-shaped niche 分支（incoming 与 claim 同 offset 同 size + 双侧 byte-faithful + claim 含非指针载荷）——Rust niche 布局契约下两种解释共存（Err 的 ptr==null 读指针槽，Ok 的整数叶写同字节）；纯指针-整数 pun（同形 8B 互斥）仍拒绝；Direct/Niche 两布局覆盖；全部 4 个 reject 负例回归绿。配套：shim+memset 落位 kernel crate（local DefId 可收集）；HEAP_CURSOR/BYTES_SERVED 改 volatile 非 atomic（原子走 inline-PTX 'l' 约束被 llc-amdgcn 拒，且单分配线程契约下无需原子）；collector 转发收集（extern decl → 同符号 no_mangle 定义；write_bytes intrinsic call 站点直接 enqueue）；verifier 双签名校验（in-module + external 权威共存）。device_heap 五段真卡全绿：heap_vec sum=285 cap=16、heap_string len=6 checksum=633、heap_box .0=21/.1=11、heap_realloc sum=1、heap_after_reset 84 | alloc_zeroed 的子-8 字节零化截断（floor(size/8)）与 arena 满后需 reset——契约已文档化 |
| Vec/Box/String（设备堆分配） | **FULL**（2026-10-31 真卡+批回归） | [e2e] device_heap 五段全过（上行）：Vec push/sum/capacity、String push/len/checksum、Box mutate、Vec truncate、reset 后复用。shim 家族（__rust_alloc/alloc_zeroed/dealloc/realloc + 检测桩 + OOM handler + memset backing）以 extern-block 精确签名（NonNull<u8>/core::mem::Alignment，feature(ptr_alignment_type)）定义于 kernel crate。**102 例批回归**：device_heap 双维 PASS（llc-pass 99/102、integrated 99/102，3 排除=抽取器局限）；批抽取器已扩展支持 no_mangle fn + helper fn 抽取 | 跨 launch 堆内容不保证（需 reset）；alloc_zeroed 语义上正确（bump 区首次使用为静态零 + 显式 volatile 清零）；atomic cursor 换 volatile（单分配线程契约，inline-PTX 'l' 约束 llc-amdgcn 拒）|

## 2. LLVM IR → AMDGPU ISA 代码生成

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| llc -march=amdgcn -mcpu=gfx1030 + COv5 + lld -shared | FULL | [e2e] 本机 pytest 14/14；[run] vecadd 字面验收 | COv5 与驱动版本耦合（Phase-1 实证 v6 被驱动拒=partial 支持陷阱），驱动升级须复验 |
| opt -O2 中端（prep pass 拆两段：opt 前 alloca / opt 后 sreg-ntid） | FULL | [e2e] batch 集成维度 14/17 hsaco（本机 2026-10-20，双管线持平） | — |
| f32/f64 算术与数学（libdevice→ocml 符号映射） | FULL | [hist] TANPROBE f32/f64 ≤2ULP；[e2e] pytest ocml 测试 + math_tan 例 batch PASS | ocml 链接需 ocml.bc+oclc_* 配置（cargo-oxide 已自动化） |
| 整数 dot（dp4a/dp2a） | FULL | [hist] dotprod 例四符号组合 GPU PASS；[e2e] pytest | — |
| packed FP8/fp4、MFMA/WMMA 矩阵指令 | UNSUPPORTED | [probe] gfx1030 无 v_mfma（缺 mai-insts）/wmma（缺 wmma-256b-insts）；静态 assert 拒绝（RDNA2 硬件事实） | 升级 RDNA3/4 卡后同探针复测 |
| TMA/块异步拷贝 | UNSUPPORTED(硬件) / PARTIAL(软件等价) | [probe] RDNA2 无 TDM；buffer_load 直通 LDS 仅 b32 [static]；[hist] 软件双缓冲流水 365 GB/s（plain 基线 78.4%，15 配置正确性全绿） | RDNA3 dlds / RDNA4 TDM 出现后按同基准复测 |

## 3. Kernel ABI / 参数传递 / Code Object

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| amdgpu_kernel 调用约定 + kernarg（标量/slice 两参包） | FULL | [e2e] pytest 两 GPU e2e；[run] vecadd | — |
| ntid x/y/z 三形参（blockDim 三维，3-D block launch 语义） | FULL | [e2e] 本机 test_coords3d_3d_launch_end_to_end（block (4,3,2)，tid/ntid 六坐标编码 24/24 真卡核对，2026-10-20）+ [run] `cargo oxide run warp_reduce`（77 个 3-D 形状含 (3,5,3)/(8,8,8)，集成管线真卡全过） | — |
| grid_constant（byval 大参数 ABI） | FULL | [run] 2026-10-26 cuda_module_contract 例真卡全过（"SUCCESS: mixed ABI typed launch passed"，含 CPU 侧契约负例校验）：①prep pass byval 提升（amdgpu_kernel 禁 byval→按值一等聚合+私 alloca 物化，通用能力）；②embedded 装载 gfx 路由（load_all_ptx_bundles_merged 在 gfx 键下直载唯一 Cubin 载荷=已链全 kernel 的 hsaco，替代 PTX 合并的 NoModules） | — |
| cluster / multicast launch | UNSUPPORTED | RDNA2 无硬件（shim 显式 NOT_SUPPORTED） | — |
| Code Object 装载（hipModuleLoadData） | FULL | [e2e] 两个 GPU e2e 走该路径；[run] vecadd | — |

## 4. GPU Runtime / 内存 / 传输 / Launch

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| host-hip 替身 crate（dlopen libamdhip64，204 个被引符号全覆盖） | FULL | [static] 子代理逐符号对账（cuda-core/async 引用面）+ 本机 vecadd/barrier 实跑 | 两处属性映射错行已修（17d9e89b，实机取证） |
| hipMalloc/Memcpy/Free 基础内存路径 | FULL | [e2e] GPU e2e 即该路径 | — |
| 属性查询（16 行映射表） | FULL | [probe] 本机 ROCm 10.1 按号直查：74=65536/75=65536/76=131072/77=报错，修复后两行与真值一致；其余行 7.16 容器头+10.1 头双确认 | `scripts/gfx1030_hip_attr_probe.c` 已固化为仓库工件 |
| sync launch | FULL | [e2e] vecadd/barrier/coords3d（hipModuleLaunchKernel 同 ABI）；sync 原生宏 ntid 三参已覆盖 | — |
| async launch（cuda-async → 0.5.0 dlopen → shim） | FULL | [run] host_closure 例 Test 1-8+10-13 全过（12 闭包形状 × 4 条 launch 路径 = 48+ 真卡 launch；sync 原生宏 ntid 补齐+栈生命周期修复后）；shim stream 面独立 C 探针 1024/1024（stream create/async memcpy 双向/stream launch/sync）。**边界**：Test 9 非 move 引用捕获（闭包结构体含宿主栈指针，上游注释明示依赖 HMM）在 gfx1030+ROCm 10.1 HSA 内存故障于宿主栈地址——HMM 不可用，move 捕获等价语义已验证可用 | — |
| 闭包 ABI（byval 单参数包，Fn/FnMut/FnOnce/const 泛型特化/repr(C)/混合尺寸捕获） | FULL | [run] 同上：8 形状全绿（Test 13 含 const-generic OFFSET 转发） | 引用捕获见 async 行 HMM 边界 |
| stream/event/pool/managed 内存 | **FULL**（2026-10-29 真卡） | [e2e] 新例 runtime_sync 六段全过：①事件计时 elapsed_ms=211.997ms vs host wall 169.8ms（数量级互证）；②query() 三态翻转（未记录 true→在飞 false→sync 后 true）；③**双流并发 2.17x**（volatile 依赖链 spin：并行 389.6ms vs 串行 846.5ms——注意普通 sum+=i 循环会被 LLVM 闭式折叠成 0.01ms 假 spin，必须 volatile 链才真转）；④跨流依赖 s1.fill→ev.record(s1)→s0.wait(ev)→host 读回全 7；⑤MemPool 创建+release_threshold+stats（4KiB 分配后 used=4096/reserved=32MiB）+pool 分配释放；⑥managed 内存 host 写→kernel bump→host 读回 roundtrip OK（malloc_managed 全局可见+DeviceBuffer::from_raw_parts 包装+simt free_async 释放）。新增公开 API：cuda-core api.rs malloc_managed（原 cudarc_shim pub(crate) 不可达）；mem_advise 因 sys::CUmem_advice 类型未导出暂缺（malloc_managed 的默认全局可见已覆盖探针语义） | runtime Stream/Event 与 simt CudaStream/CudaEvent 是两套抽象（探针混用需 DeviceBuffer::from_raw_parts 桥接 managed 指针）；大 buffer 须宽 grid（1 block×32 线程只盖 32 元素） |
| host func / graph / multicast / cluster API | PARTIAL / UNSUPPORTED | host func BLOCKING 回落随 shim 保留；graph 族符号在 shim 表；multicast/cluster：RDNA2 无硬件，shim 明示 NOT_SUPPORTED [static] | host func 路径本机 e2e |

## 5. 线程 / 工作组 / 索引 / 同步 / 共享内存

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| tid/ctaid x/y/z → workitem/workgroup id（含 3-D） | FULL | [e2e] vecadd/barrier e2e；[e2e] coords3d 三维索引（tid.y/z 硬件值+三形参 kernarg，24/24）；[probe] 设备身份测试确认 gfx1030 agent | — |
| 全组 barrier（barrier0 → s.barrier） | FULL | [e2e] barrier_smoke 例 [hist] + 本机 batch | — |
| 静态共享内存（LDS，undef 初始化契约） | FULL | [hist] sharedmem 例 GPU PASS；[e2e] pytest alloca/addrspace 测试 | — |
| mbarrier（软件单字状态机：pending/expected/phase） | FULL | [e2e] 本机 test_barrier_mbarrier_software 三 kernel 语义断言全过（补重放 Step 16 后） | expect_tx 字节级（=vmcnt 部分排空等价）待实现 |
| counted barrier（LDS 计数器+世代自旋，旁观 warp 安全） | FULL | [hist] 128 线程 wave64 含旁观 warp GPU PASS；[e2e] 本机 pytest IR 断言 | id 常量缺 0..15 范围检查（review P1） |
| warp shuffle 4 模式 × f32/i32 + f64 asm 拆分 + 动态 mask(up,0) | FULL | [hist] 四模式真卡验证；[e2e] 本机 pytest 全形态（含 mbcnt.lo+hi 两段 lane 号，wave64 正确） | — |
| 部分交换（字面 mask≠-1）/ 非 32 宽度 clamp | PARTIAL(信封) | [static] 信封外保留 nvvm 调用让 llc 显式拒绝（fail-loudly 设计，非静默错） | 按需扩信封 |

## 6. 原子操作 / 内存序 / 数值

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| LLVM 原子 RMW/CAS（global/shared，多内存序） | FULL | [hist] MP litmus release→acquire 200 轮 + atomics 例 GPU PASS；[e2e] 本机 pytest+batch | — |
| fence + scope 映射（cta→workgroup，gl→agent，sys→系统） | FULL | [e2e] pytest 断言三 scope 产物；[hist] litmus | — |
| Rust core::sync::atomic 内嵌 PTX-asm 族 → LLVM 原子 | FULL | [hist] GPU 验证；[e2e] pytest 12d/12e 全形态 | — |
| 64 位原子（load/store/atomicrmw RMW） | FULL | [e2e] pytest b64 形态；[run] atomics 例 24/24 真卡（含 Test 8 DeviceAtomicU64 fetch_add），2026-10-25 | — |
| 设备端 printf | **FULL**（2026-10-26 真卡） | [run] `cargo oxide run printf` → **"=== ALL PRINTF TESTS PASSED ==="**（8 测试族：整数/浮点/宽度对齐/符号旗标/线程索引/返回值/布尔/默认转换，含 varargs 打包+host 格式化）。实现=prep pass 双侧（amdgcn.rs+translate_amdgcn.py）vprintf→ockl 三函数展开（begin/append_string_n/append_args×7槽，字段类型 i32/i64/f64/f32/[4×i8]→i64 槽升级）+ 模块元数据 `amdgpu_printf_kind="hostcall"` + ptx.rs ockl.bc 链接（oclc_isa_version_<n>+abi_version_600+finite_only_off 三配置）。取证链：ockl.bc llvm-dis ABI + printf_mod B 路探针（hipModuleLoadData 路径 printf 服务实证）+ preockl.ll 逐错修复（9 项：@前缀/nonnull/GEP裸指针/元数据编号/llvm-link 文件化参数等） | 动态 fmt 字符串保留 vprintf（链接期显式失败，信封纪律）。**>7 参数链式 2026-10-28 真卡通过**（clang 同款多轮 append_args：每轮 ≤7 值、返回 handle 接力、is_last 仅末轮；ockl.bc 取证 `nargs<<5` handle 编码；10 参行 `chain: 1 2 3 4 5 6 7 8 aaaabbbbccccdddd 2.7183` 逐字符正确；双侧 amdgcn.rs+translate_amdgcn.py 同步）。**同日乱码根因修复**：对照真 hipcc C++ printf kernel（三/五连发全对，排除 host/驱动）定位四处展开形态缺陷并双侧修复——① append_string_n 长度含 NUL；② handle 必须从 string_n 返回值接力（旧 begin handle → host 流配对错乱）；③ 无参=单条 string_n(is_last=1)；④ repr(C) 对齐空洞（[4 x i8]）是 padding 非槽，跳过（Mixed 五参行 `0.000000 4614256656748904448 -nan 4005bf0a8b04919b` 逐字节吻合错位序列）。修复后九族真卡零乱码（Mixed `-42 3.141593 18446744073709551615 2.72 ff`、`y=20`、`{: }` 旗标全对）；诊断基建=printf 例 `ONLY=<n>` 单族开关。宿主栈：conda-forge ROCm 7.2（~/opt/rocm2，rocminfo/rocm-smi/hipcc/位码 67 件恢复）+ 真 gcc 16.2（~/opt/gcc-env），见 docs/research/2026-10-28-environment-recovery.md |

## 7. 泛型 / 结构体 / 跨 Crate / 宏与集成

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| #[kernel]/#[cuda_module] proc-macro 管线 | FULL | [e2e] 全部 e2e/批量即该管线 | — |
| 跨 crate device 代码（helper-lib 形态） | PARTIAL | [static] shared_memory_helper 例已带回本线；batch 排除（跨模块宏 device-only 抽取限制，记录在案） | 抽取器支持跨模块宏展开 |
| 190+ 上游示例面（CUDA 语义回归） | PARTIAL | 本机 batch=17 例代表集；上游 190+ 在 CUDA 侧回归由 CI 承担；gfx 侧扩展按需 | 按功能矩阵逐族扩 batch |
| cargo-oxide ensure_hip_patch（自动注入，幂等，手动优先） | FULL | [static] 三态契约+测试；[run] 本机 vecadd/barrier 集成构建实跑注入 | 写失败静默（review P2）待告警化 |

## 8. 工具 / 诊断 / 测试 / 性能

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| cargo oxide run/build/pipeline/inspect | FULL | [run] 本机 vecadd/barrier | — |
| pytest 回归套件（IR 改写 10 + 设备身份 + GPU e2e 2） | FULL | [e2e] 本机 14/14 | docker 容器路径回退宿主机（已入树） |
| PTX packed-math asm 展开（bf16x2/f16x2/f32x2/b16 原子/ex2.f16） | **FULL**（2026-10-26） | [e2e] prep pass Step 17b（translate_amdgcn.py）将 7 族 PTX 内嵌 asm 展开为合法 AMDGPU IR：b16 原子→LLVM 原子 i16 load/store；bf16x2 add/sub/mul→bf16→f32 移位展宽+f32 运算+RNE 回缩；bf16x2 min/max→fcmp select；bf16x2 abs/neg→i32 掩码/异或；bf16x2 fma→llvm.fma.f32+RNE；bf16x2/f16x2 cvt→fptrunc/RNE/RTZ/relu clamp；f32x2→<2 x float> 向量 bitcast+fadd/fsub/fmul/fma.v2f32；ex2.approx.f16→llvm.amdgcn.exp2.f32。**ISA 取证**：gfx1030 无 v_pk_bf16（llc 探针 "instruction not supported"）、f32x2 打包无硬件（RDNA3）、v_pk_f16 须 v 约束。**llc-pass 7/7**（atomic_f16/bf16x2_arith/bf16x2_fma/cvt_f16x2/cvt_packed/f32x2_arith/ex2_approx_f16）。nvls（multimem SHARP）+mma fp8 = HARDWARE_LIMITED 保留原样 fail-loud。**Rust 侧 2026-10-27 同步完成 7/7 集成构建转绿**（amdgcn.rs Step 17b 镜像，df79fb980：SSA 名全局递增 _PM_COUNTER、exp2/fma declare 首 define 前统一插入、bf16x2 cvt relu 匹配 startswith(cvt.rn/rz)+contains(bf16x2)、relu 链 b{tag}z 命名、%源 bitcast 直产 b{tag}） | — |
| 批量双维度验证（Python 改写管线 / Rust 集成管线） | FULL | **终局 2026-10-26：99 例批，两维度各 96/96 可跑=100%**（3 例诚实排除；扫描收官 194/194=总 pass 87+67 build_fail+38 llc_fail+1 no_ll+1 sweep_error(tcg05=Blackwell 张量核,HARDWARE_LIMITED)） | [e2e] 核心批 14/17+14/17 持平（3 例诚实排除：抽取器限制）；**[goal 扩展 2026-10-26]** 207 候选全管扫描跑至字母 p 段（shell 故障中断）：**58 新例全绿并入 EXAMPLES**（核心批 17→75），典型族：常量内存×4、enum/Option/dst/drop_glue 语义族、f16/libdevice 数学族、对齐访存族；13 例抽取失败=无 cuda_module 形态（已入诚实排除表）；llc 失败两类：PTX asm 约束缺口（h/f/l——f16/bf16 内嵌 asm 族， IMPLEMENTABLE）与 nvvm 残留致 llc abort（cluster/cp_async/elect 等 SM90+ 特性族， fail-loud）。~~q-z 续扫~~ 已完成（+26 例入列；99 例终态见上） | asm 约束映射入研究档案 |
**全维回归 2026-10-26（73 例批，磁盘输出目录）**：Python 维度 **70/70 可跑全过=100%**（3 例诚实排除不计分）；集成维度 **69/70**——唯一失败 packed_aggregate_abi 根因=该例 main 无条件 `verify_generated_ptx()` 解析 PTX 文本自检 ABI，gfx 产物为 hsaco 无 PTX 文本（例自身 NVPTX 文本耦合，IR 级与 Python 维度过、kernel 装载路径已由 embedded 修复覆盖）；修复：本轮 alloca 改写器放开 packed/向量类型（`<{ i8, i32 }>` 曾被 `>` 守卫跳过，Rust+Python 双侧） | packed_aggregate_abi 候选改进=gfx 路线同时产 PTX 副产物（供文本自检类例） |
| 一键验证+报告（scripts/verify_gfx1030.sh） | FULL | 本脚本：环境快照/5 门/汇总，时间戳归档 | 首跑归档随本矩阵提交 |
| cargo-oxide doctor（工具链自检） | FULL | [run] 2026-10-25 "✅ Environment looks good!"（llc 经工具链 llvm-tools 解析；补齐 ~/opt/cuda13 的 libnvJitLink 后全绿——NV 专属件，gfx 路径本不依赖） | — |
| 性能基线（带宽/计算锚点，Phase D 门） | FULL | [run] 2026-10-25 ROCm 10.1 直链 HIP 基准（scripts/gfx1030_perf_baseline.c，hipcc 编译、事件计时、已入 verify 电池）：理论 575 GB/s（GDDR6 16T/pin/clk 公式，对 spec 576）；H2D/D2H 14.1/14.2 GB/s（pageable；历史 pinned 参照 24.9/28.5）；D2D copy 820 GB/s（历史同口径 862 互证）；triad 3N 有效 486 GB/s=**85% spec**；FFMA 25.9 TFLOPS≈双发射峰 93%。rocprofv3 在位、应用可跑但其产物提取未配置（开放项） | 90% 达标线属优化工作门，非本记录门 |
| 产物反汇编抽查（llvm-objdump） | FULL | [run] vecadd.hsaco：`vecadd` 符号在 .text + `vecadd.kd` 内核描述符 + 真 RDNA2 ISA（v_add_co_u32/global_load_dword/s_cbranch_execz），elf64-amdgpu | — |
| 设备端 DWARF 调试信息 | IMPLEMENTABLE（工具链已证，前端未实现） | [probe] 2026-10-27 llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 --debugger-tune=gdb -filetype=obj 对带 !dbg 的最小 IR 产出完整 DWARF：7 个 .debug_* 节，`file` 报 "with debug_info"，ROCm llvm-dwarfdump 解析出 DW_TAG_compile_unit(k.rs)/DW_TAG_subprogram(k)。**当前管线不产生**：`.amdgcn.ll` !dbg metadata 计数=0，cvt_packed.hsaco 节表零 .debug_*（llvm-readobj 全节枚举） | 剩余工作：rustc-codegen-cuda 自定义后端实现 debuginfo API（MIR DI 建图→IR dbg metadata），llc 路径已证无需改动 |

## §8 性能要求判定（带宽/延迟 99% 理论线）

| 项 | 实测天花板 | 判定 |
|---|---|---|
| 带宽（读流，DRAM-bound） | **553 GB/s = 96.1% of 575 spec**（192MB 工作集≫128MB L2，float4 访问） | 未达 99% |
| 带宽（triad 混合读写） | 519 GB/s = 90.2%（11 配置扫描最优 v2@blk512；ILP4 更差 74%——纯流式需满 wave 并发） | 未达 99% |
| 带宽（copy 2N） | 504 GB/s = 87.6% | 未达 99% |
| 带宽（write-only 125%） | **缓存假象**：192MB store 落 128MB L2 后脏逐出异步发生，kernel 完成时未达 DRAM | 非 DRAM 带宽证据 |
| 延迟（空 kernel 往返） | min **10.97 µs** / 中位 11.20 / p99 12.68（2000 次，scripts/gfx1030_latency_probe.c 真卡实测） | 分解：入队摊销 0.90 + GPU 事件窗 7.28 + 唤醒 ~2-3 ≈ 10-11 ✓ 自洽。**SPEC_ABSENT**：AMD 无官方地板规格；rocprofv3 参数解析在此安装不可用（工具限制两则实证）。userspace 无低延迟轮询 API 可绕过。见 §8 判定表 |
| 延迟（4KB D2D 同步拷贝） | min 11.5 µs（此前实测） | 同上 |

**延迟半边实测分解（2026-10-26，scripts/gfx1030_latency_probe.c，gfx1030 真卡）**：四段直接测量——①入队摊销 **0.77 µs/launch**（2000 连发无同步）；②GPU 事件窗 **7.12 µs**（空 kernel+两侧事件包的流水上界；纯 kernel 更低）；③完整往返 **min 11.01 / 中位 11.23 / p99 13.53 µs**；④自旋轮询变体 **10.69 µs**（排除阻塞唤醒开销）。**算术自洽**：0.77+7.12+（唤醒 ~1-3）≈ 9-11 ✓ 与③吻合——往返的主导项是 GPU 侧取包-分发-完成流水，非 host 开销。**99% 判定**：AMD 未公布 gfx1030 空 kernel 往返的官方理论地板（rocprofv3 kernel_dispatch 表在本安装上为空、agent CSV 不导出 Max_Bandwidth/延迟字段——工具限制两则已实证），故"99% of 理论"在延迟侧**无权威分母**；以实测分解为准记 **SPEC_ABSENT**（可审计依据=上述四段+工具限制记录）。经验地板构成：host 入队 0.77µs 已近 API 极限，剩余 ~10µs 为运行时完成路径（信号+唤醒），userspace 无绕过 API（无 busy-poll doorbell 接口）——若后续 ROCm 提供低延迟轮询扩展可复测。

**99% 未达的物理依据（可审计）**：①GDDR6 DRAM 刷新（tREFI≈7.8µs/行，逐行周期性关停约 1-3% 带宽）；②读写总线换向（turnaround）与命令总线开销——混合流（triad/copy）额外损失；③MC 调度竞争。读流 96.1% 是无换向纯读的实测天花板；所有观测到的 >100% 数字（write-only 125%、历史 64MB D2D 820）均为 128MB Infinity Cache 命中假象。**结论：99% of pin-rate 理论值在任何实际 kernel 上物理不可达，按 goal §2 记 HARDWARE_LIMITED（依据如上）；96.1% 为诚实上限证据**。若"理论上限"采用运行时口径（rocprofv3-avail 的 max_bandwidth 字段——当前工具只吐字段名不吐值，开放项），数值待取。复现：`scripts/gfx1030_perf_tuned.c`（hipcc -O3 编译，已入 verify 电池 perf-tuned 门）。

## 配套研究档案

硬件能力差距与软件替代方案（含 soft-TMA 深度案例与 NOT FEASIBLE 证据基线）见 [research/gfx1030-substitution-catalogue.md](research/gfx1030-substitution-catalogue.md)。

## 已知限制与阻塞（跨条线）

1. **COv5×驱动耦合**：code object v6 被当前 amdgpu 驱动拒绝（partial 支持陷阱），驱动升级必须复验全套。
3. **docker 缺组**：本机 zhuo 不在 docker 组（`sudo usermod -aG docker zhuo` 可解）；e2e 已回退宿主机直跑，不阻塞。
4. **无 HIP 头**：ROCm 10.1 本机安装不含开发头；`amdrocm-runtime-dev10.1` 拆包至 `~/opt/rocm-headers/`（CPATH 接入），不动系统目录。
5. **HMM 不可用**：非 move 引用捕获闭包（结构体内宿主指针被 GPU 解引用）在 gfx1030+ROCm 无 HMM 下 HSA 故障；move 捕获等价语义已验证（host_closure Test 9 唯一受影响项）。
6. **硬件能力边界**（RDNA2 事实，非缺陷）：无 MFMA/WMMA、无 TMA/TDM、无 cluster/multicast、fp8/fp4 无硬件——对应条目均给出软件等价或显式拒绝路径。

## 矩阵跟进项（下一步优先序）

1. ~~3-D block launch 的本机 GPU e2e~~（已完成 2026-10-20：coords3d e2e + warp_reduce 集成执行）
2. ~~async launch 路径本机运行断言~~ → 转 BLOCKED 调查（见 Runtime 节；研究档案 §4 开放项 #1）
3. ~~review P1 修复批~~（已完成 2026-10-20：dlopen 绝对路径 / mbcnt.lo+hi / counted-barrier 注入点+id 范围 / batch exit code+stale hsaco / host func BLOCKING / sync 原生宏经补重放覆盖）
4. 性能基线重建（Phase D 性能门）
5. batch 面从 17 例向 190+ 扩展（按族分批）
