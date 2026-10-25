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
| 迭代器/Vec/格式化等依赖设备端分配/std 的设施 | NOT TESTED | 上游 CUDA 侧同样限定；gfx 侧未单独验证 | 如需，先做 device-side allocator 探针 |

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
| grid 常量/cluster 等高级 launch 特性 | UNSUPPORTED / NOT TESTED | cluster：RDNA2 无硬件（同 MFMA 依据）；grid_constant：上游 #1327 packed-as3 形态差异在 Python 维度有适配，集成维度 NOT TESTED | 逐例核对 batch 记录 |
| Code Object 装载（hipModuleLoadData） | FULL | [e2e] 两个 GPU e2e 走该路径；[run] vecadd | — |

## 4. GPU Runtime / 内存 / 传输 / Launch

| 功能 | 状态 | 证据 | 缺口/下一步 |
|---|---|---|---|
| host-hip 替身 crate（dlopen libamdhip64，204 个被引符号全覆盖） | FULL | [static] 子代理逐符号对账（cuda-core/async 引用面）+ 本机 vecadd/barrier 实跑 | 两处属性映射错行已修（17d9e89b，实机取证） |
| hipMalloc/Memcpy/Free 基础内存路径 | FULL | [e2e] GPU e2e 即该路径 | — |
| 属性查询（16 行映射表） | FULL | [probe] 本机 ROCm 10.1 按号直查：74=65536/75=65536/76=131072/77=报错，修复后两行与真值一致；其余行 7.16 容器头+10.1 头双确认 | `scripts/gfx1030_hip_attr_probe.c` 已固化为仓库工件 |
| sync launch | FULL | [e2e] vecadd/barrier/coords3d（hipModuleLaunchKernel 同 ABI）；sync 原生宏 ntid 三参已覆盖 | — |
| async launch（cuda-async → 0.5.0 dlopen → shim） | **BLOCKED（调查中）** | 模块加载已通（2026-10-25 两修复：execution-arch gfx 路由——HIP CC 10.3 被误读 sm_103；ltoir 加 hsaco 文件工件路由——闭包工件原走 libNVVM 必拒）；现象=首个 async 闭包 launch 后 kernel 永不完成（strace：启动期 111 ioctl 后阻塞单等待，非自旋）。复现：`CUDA_OXIDE_TARGET=gfx1030 cargo oxide run host_closure`（adapt-rocm） | 下一步：非闭包 async 变体隔离 / rocprof 查 kernel 在跑否 / stream+event shim C 探针 |
| stream/event/pool/managed 内存 | NOT TESTED | 符号在表 [static]；本机无运行断言 | 同上，逐面补 e2e |
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
| 64 位原子 load/store（seqcst 折叠） | FULL | [e2e] pytest b64 形态；64 位 atomicrmw RMW 本机 NOT TESTED | 补 64 位 RMW 断言 |
| 设备端 printf | NOT TESTED | HIP 支持存在；管线未验证 vprintf 路径 | 低优先探针 |

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
| 批量双维度验证（Python 改写管线 / Rust 集成管线） | FULL | [e2e] 本机 14/17+14/17 持平（3 例诚实排除：抽取器限制）；exit code + stale hsaco 两处假 PASS 源已修（2026-10-20） | — |
| 一键验证+报告（scripts/verify_gfx1030.sh） | FULL | 本脚本：环境快照/5 门/汇总，时间戳归档 | 首跑归档随本矩阵提交 |
| cargo-oxide doctor（工具链自检） | FULL | [run] 2026-10-25 "✅ Environment looks good!"（llc 经工具链 llvm-tools 解析；补齐 ~/opt/cuda13 的 libnvJitLink 后全绿——NV 专属件，gfx 路径本不依赖） | — |
| 性能基线（rocprof/带宽/达标率） | NOT TESTED | 历史流程在 gpu-perf skill（主机 ROCm 7.14 时代）；10.1 时代未重建 | Phase D 后续：重建 L4 基线 |
| 产物反汇编抽查（llvm-objdump） | FULL | [run] vecadd.hsaco：`vecadd` 符号在 .text + `vecadd.kd` 内核描述符 + 真 RDNA2 ISA（v_add_co_u32/global_load_dword/s_cbranch_execz），elf64-amdgpu | DWARF 调试信息仍未验证（NOT TESTED） |

## 配套研究档案

硬件能力差距与软件替代方案（含 soft-TMA 深度案例与 NOT FEASIBLE 证据基线）见 [research/gfx1030-substitution-catalogue.md](research/gfx1030-substitution-catalogue.md)。

## 已知限制与阻塞（跨条线）

1. **COv5×驱动耦合**：code object v6 被当前 amdgpu 驱动拒绝（partial 支持陷阱），驱动升级必须复验全套。
3. **docker 缺组**：本机 zhuo 不在 docker 组（`sudo usermod -aG docker zhuo` 可解）；e2e 已回退宿主机直跑，不阻塞。
4. **无 HIP 头**：ROCm 10.1 本机安装不含开发头；`amdrocm-runtime-dev10.1` 拆包至 `~/opt/rocm-headers/`（CPATH 接入），不动系统目录。
5. **async 闭包 launch 挂死**（调查中，见 Runtime 节与研究档案 §4）。
6. **硬件能力边界**（RDNA2 事实，非缺陷）：无 MFMA/WMMA、无 TMA/TDM、无 cluster/multicast、fp8/fp4 无硬件——对应条目均给出软件等价或显式拒绝路径。

## 矩阵跟进项（下一步优先序）

1. ~~3-D block launch 的本机 GPU e2e~~（已完成 2026-10-20：coords3d e2e + warp_reduce 集成执行）
2. ~~async launch 路径本机运行断言~~ → 转 BLOCKED 调查（见 Runtime 节；研究档案 §4 开放项 #1）
3. ~~review P1 修复批~~（已完成 2026-10-20：dlopen 绝对路径 / mbcnt.lo+hi / counted-barrier 注入点+id 范围 / batch exit code+stale hsaco / host func BLOCKING / sync 原生宏经补重放覆盖）
4. 性能基线重建（Phase D 性能门）
5. batch 面从 17 例向 190+ 扩展（按族分批）
