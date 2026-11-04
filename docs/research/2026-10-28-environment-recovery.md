# 2026-10-28 环境重建记录：/opt 清空后的 gfx1030 工具链重组

## 事件

2026-10-27 23:35（UTC+8），宿主机 `/opt` 被外部清空并重建（仅剩
`bytedance/`、`google/`、`ZCode/`）。随之消失的组件：

- `/opt/rocm`（ROCm 10.1 完整用户态：hipcc、ld.lld、ockl/ocml/oclc 位码、
  llvm-dwarfdump、rocminfo、amd-smi）
- 系统 C/C++ 编译器（gcc/clang 从未以 dpkg 形式存在；此前可用的 `cc` 一并消失）
- `docker`、`pixi`、部分 coreutils（`timeout` 等符号链接断裂）
- `~/.pixi`、`~/.venvs` 之外的包管理器

不可恢复——以下记录用**现存真实组件**重组了等价工具链，全部为真机执行，
无 mock。

## 重组后的组件清单（全部在 `$HOME` 下）

| 组件 | 位置 | 构成 |
| --- | --- | --- |
| `cc` / `gcc` | `~/bin/cc`（symlink `gcc`） | wrapper → `zig cc`（pip 包 `ziglang` 0.17，clang 22.1.8 内核） |
| `hipcc` | `~/bin/hipcc` | wrapper → `cc -D__HIP_PLATFORM_AMD__ -I~/opt/rocm-headers -L~/opt/hip-lib -lamdhip64 -lc++ -lc++abi -Wl,-rpath,...`；`--offload-arch` 接受但惰性（宿主侧仅编 HIP-API 主机码，kernel 在预构建 hsaco 里） |
| `ld.lld` | `~/opt/rocm-shim/llvm/bin/ld.lld` | symlink → rustup 工具链 `bin/gcc-ld/ld.lld`（LLVM 23.1.1） |
| `HOST_ROCM`（测试用） | `~/opt/rocm-shim` | 上两项的布局镜像，`test_gfx1030.py` 已改指此处 |
| HIP/HSA 运行库 | `~/opt/hip-lib` | triton wheel 自带 `libamdhip64.so`/`libhsa-runtime64.so`/`libamd_comgr.so`/`libdrm*` 的 **soname 符号链接**（`.so.1`/`.so.7`）+ 系统运行库补链（libelf/libnuma/libdrm/libstdc++） |
| HIP 头 | `~/opt/rocm-headers`（幸存） | ROCm 10.1 HIP 开发头（amdrocm-runtime-dev 拆包产物） |
| ockl/ocml 位码 | `~/opt/rocm-bc/amdgcn/bitcode/` | `ockl.bc`/`ocml.bc` 取自 triton wheel；`oclc_*.bc` 九件为 **llvm-as 现造**（见下） |
| 设备识别探针 | `~/bin/hipinfo` | zig cc + `~/opt/rocm-headers` 编译的 HIP API 探针，输出 `agent N: <gcnArchName> (<设备名>)`；PCI 证据取 sysfs `1002:73a5`（Navi 21） |
| 后端链接器 | `~/opt/link-libs/smart-ld` | rust-lld 包装：exe 链接补 Scrt1/crti/crtn + `--dynamic-linker`（rustc 的 ld.lld flavor 不传 gnu crt）；`-shared` 时剥掉这些参数（否则 proc-macro .so 引入未定义 `main`，dlopen 失败） |

## 关键坑位（重建时必读）

1. **rustup sysroot 的 LLVM 位码脚本**：`libLLVM-23-rust-1.99.1-stable.so`
   是 `INPUT(libLLVM.so.23.1-rust-1.99.1-stable)` 脚本，真身**无 `.so`
   后缀**——lld 拒绝无扩展名文件（"unrecognized file extension"）。修复：
   两个 sysroot lib 目录（`lib/` 与 `lib/rustlib/x86_64-unknown-linux-gnu/lib/`）
   都补 `libLLVM.so.23.1-rust-1.99.1-stable.so` 符号链接，并把脚本改指
   `.so` 名。
2. **zig cc 会把自带 libunwind 静态链进 .so**——rustc 加载该后端 .so 后
   panic 展开走 zig 的 `_Unwind_Resume` 与 rustc 进程内 unwinder 冲突，
   `codegen_crate` 阶段 SIGSEGV。因此**后端 .so 必须用 rust-lld 链**
   （`-C linker=smart-ld -C linker-flavor=ld.lld`），系统 `libgcc_s.so.1`
   提供 `_Unwind_*`。
3. **rustc 的 ld.lld flavor 不传 gnu crt 对象**——可执行文件缺
   PT_INTERP（启动即段错误）、.so 不该有它们（proc-macro `main` 未定义）。
   smart-ld 按是否 `-shared` 分流（见上表）。
4. **`libgcc_s.so`/`libstdc++.so` 等 dev 符号链接缺失**——`~/opt/link-libs`
   内补 `.so → .so.N` 符号链接，链接期 `-L native` 引用。
5. **zig cc 的 C++ 模式要显式 `-lc++ -lc++abi`**，否则报 `vector` 找不到；
   加上后 zig 全自洽链接（自带 libc++/crt/lld），C++ 主机码可编可跑。
6. **ockl 链接所需的 oclc_*.bc**（ROCm 10.1 原件已失）：用 llvm-as 从
   IR 现造，datalayout 抄自 ockl.bc 反汇编。符号与取值（gfx1030）：
   - `__oclc_ISA_version`(i32)=1030、`__oclc_wavefrontsize64`(i8)=**1**
     （gfx1030 默认 wave64 启动；曾错设 0 导致 hostcall 乱码）、
     `__oclc_wavefrontsize_log2`(i32)=6
   - `__oclc_ABI_version`(i32)=500（注意大小写 `__oclc_ABI_version`）
   - `__oclc_finite_only_opt`/`__oclc_unsafe_fp_opt`/
     `__oclc_correctly_rounded_sqrt32`/`__oclc_daz_opt`（各 on/off 两件，i8）
   - 环境变量：`ROCM_PATH=~/opt/rocm-bc`（find_rocm_bitcode 的
     `<rocm>/amdgcn/bitcode` 布局）
7. **cargo-oxide 后端覆盖**：`CUDA_OXIDE_BACKEND=$HOME/workspace/src/
   cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda/target/debug/
   librustc_codegen_cuda.so`（rust-lld 构建产物；cargo-oxide 默认会重编
   后端并走 zig cc → 触发坑 2）。
8. **pytest 探针 crate**：`CARGO_NET_OFFLINE=true` +
   `CARGO_TARGET_DIR=<probe>/target`（在线查 replaced source 的失败会挡住
   path 依赖图解析）。
9. **docker 缺失**：`_docker_available()`/`_env_diag()` 对二进制缺失
   （FileNotFoundError）按"不可用"处理，自动走宿主直跑路径。

## printf 乱码根因（2026-10-28 深挖修复，双侧已改）

重组栈初版的"host 侧流解析乱码"定性**有误**——真根因在我们 prep pass
的展开形态与 ockl/host 契约不匹配，共四处（对照系 = 真 hipcc 编译的同
形态 C++ printf kernel，三连发/五连发全对，排除 host/驱动）：

1. **strlen 不含 NUL**：clang 形态 append_string_n 的长度含 NUL
   （llvm-dis of hipcc IR：strlen 差值 +1）。10.1 时代不含 NUL 可过，
   7.2 起必须含。
2. **handle 不接力**：clang 展开把 append_string_n 的**返回 handle** 传给
   append_args；我们此前沿用 begin 的旧 handle → host 端流配对错乱
   （`f32 default: {}` 值丢失、消息中途截断 `y=2`、二进制垃圾行）。
3. **无参形态**：clang = 单条 `append_string_n(is_last=1)`，无 args 消息，
   其返回值即 vprintf 结果；我们此前多发一条 `append_args(nargs=0,…,1)`。
4. **repr(C) padding 当槽**：`#[repr(C)]` 打包 struct 的对齐空洞显式化为
   数组成员（如 `i32` 后的 `[4 x i8]`）。`GpuPrintfArg::Promoted` 契约只产
   标量（i32/u32/i64/u64/f64），数组成员必是 padding——prep pass 此前把
   它当参数读，6 槽 vs fmt 5 个 %，第 2 参起整体错位（Mixed 行
   `0.000000 4614256656748904448 -nan 4005bf0a8b04919b` 逐字节吻合
   pad/pi/max/e 错位序列）。

四处修复均已双侧落地（amdgcn.rs + translate_amdgcn.py）。修复后
printf 九族真卡输出零乱码（含 Mixed 五参混合、10 参链式、`{: }`
空格旗标、`y=20`）。诊断基建：printf 例新增 `ONLY=<n>` 环境变量可单跑
第 n 族（乱码二分用）。

## 工具链终态（2026-10-28 第二轮重组：真 gcc + conda-forge ROCm 7.2）

- **真 gcc**（conda-forge gcc_linux-64 16.2.0，`~/opt/gcc-env`）取代 zig cc
  shim：`~/bin/cc|gcc|g++` wrapper。后端 .so 由真 gcc 链接——smart-ld/
  手工 crt/libLLVM 位码脚本补丁全部退役（gcc 走标准 bfd 路径，NEEDED
  干净）。zig cc 的 libunwind SIGSEGV / ld.lld crt 坑不再适用。
- **ROCm 7.2 完整用户态**（conda-forge，`~/opt/rocm2`）：hipcc（真 clang
  22 HIP frontend）+ rocminfo（VBIOS Marketing Name 读数恢复：
  "AMD Radeon RX 6950 XT"）+ rocm-smi + 67 件 oclc/ockl/ocml 位码 +
  libamdhip64/libhsa/libamd_comgr/libdrm 配对集。
- `~/opt/hip-lib`：soname 符号链接池（→ rocm2 真身；**必须含
  libamd_comgr.so.3 带版本尾**——hip 7.2 按 .so.3 dlopen，只有无版本名
  会 hipInit 报 NO_DEVICE）。
- `~/opt/rocm2-shim/amdgcn/bitcode` → rocm2 的 clang resource 位码布局
  （find_rocm_bitcode 的 `<rocm>/amdgcn/bitcode` 契约）。
- bindgen（cuda-bindings 构建）：`BINDGEN_EXTRA_CLANG_ARGS=-I~/opt/rocm2/
  lib/clang/22/include`（clang builtin 头；系统 llvm-22 resource dir 残缺）。
  LD_LIBRARY_PATH **不得**前置 rocm2/lib——其 libLLVM.so.22.1 会被
  libclang-22 错配（M68k 符号缺失）。
- hipcc shim（~/bin/hipcc）：g++ + `--offload-arch`/`-x hip` 过滤（宿主
  侧只编 HIP-API 主机码）。pytest 宿主直跑路径走它。
- micromamba：`~/opt/tools/micromamba`（/tmp 会被清，持久件一律放
  ~/opt）。
- **rocprofiler-sdk（AMD 官方 deb 解包，`~/opt/rocprof-sdk/opt/rocm-7.2.4`）**：
  repo.radeon.com/rocm/apt/7.2.4 四件套 `rocprofiler-sdk_1.1.0 +
  rocprofiler-sdk-roctx + rocprofiler-sdk-rocpd + hsa-amd-aqlprofile_1.0.0`
  （24.04 变体，dpkg-deb -x 用户级解包）。rocprofv3 --version 正常
  （1.1.0/97f5574）。**抓取层根因定位（2026-10-29 第二轮）**：`--kernel-trace/-hip-trace`
  仍零输出，但本轮闭环了三件事——① **依赖归零**：补
  rocprofiler-register_0.6.0 deb（`librocprofiler-register.so.0` 是
  官方 hsa 的 NEEDED 注入点，conda 构建的 hsa/hip **均无此链接**=
  之前零回调的最强解释）+ comgr 符号链接进 sdk lib 目录；② **官方
  hsa-rocr_1.18.70204** deb 解包并切默认池（hip-lib → 官方 hsa，带
  register 注入点；官方 hsa dlopen comgr 走 $ORIGIN——comgr 必须在
  它的 lib 目录）；切换后全链回归绿（pytest 15/15、printf 九族零
  乱码）。③ **抓取仍静默** → 最终判定=KFD perf 权限：aqlprofile 的
  event page/计数器 ioctl 需 root 或 amdgpu.perfmon=1（本机
  /sys/module/amdgpu/parameters/perfmon 不存在、sudo -n 不可用）。
  归类 **BLOCKED（环境权限）**，非软件缺陷；持 root/perfmon 的环境
  按本档案应可直接抓取。另：/tmp 会被外部清理，perf_baseline 已固化
  ~/opt/perf/（真 hipcc 编译，-x hip 必需因 .c 后缀默认 C 模式）。

## 复现命令（最小链路）

```bash
export CUDA_TOOLKIT_PATH=$HOME/opt/cuda13 RUSTC_BOOTSTRAP=1
export BINDGEN_EXTRA_CLANG_ARGS="-I$HOME/venvs/porvl/lib/python3.14/site-packages/ziglang/lib/include"
export LD_LIBRARY_PATH="$HOME/opt/hip-lib:$PWD/../host-hip/cuda-hip-shim/shim-lib"
export CUDA_OXIDE_BACKEND=$HOME/workspace/src/cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export ROCM_PATH=$HOME/opt/rocm-bc CARGO_NET_OFFLINE=true

# 后端 .so（rust-lld 路径；flags 见 ~/opt/link-libs/smart-ld 用法）
cd crates/rustc-codegen-cuda
UL=/usr/lib/x86_64-linux-gnu
RUSTFLAGS="-C linker=$HOME/opt/link-libs/smart-ld -C linker-flavor=ld.lld \
 -C link-arg=$UL/Scrt1.o -C link-arg=$UL/crti.o -C link-arg=$UL/crtn.o \
 -C link-arg=--dynamic-linker=/lib64/ld-linux-x86-64.so.2 \
 -L native=$HOME/opt/link-libs -L native=$UL" cargo +stable build

# 真卡跑 printf（含 >7 参链式）
cd .. && CUDA_OXIDE_TARGET=gfx1030 cargo oxide run printf
```

## 验证状态（2026-10-28）

- pytest `tests/test_gfx1030.py`：**15/15**（含 vecadd 数值、barrier/mbarrier、
  coords3d 3-D 真卡 e2e——hipinfo 识别 + shim 工具链 + 真 GPU 执行）
- printf 例（重组栈）：exit=0，>7 参链式行精确正确；部分旧行乱码见上节
  定性
- 后端 .so：rust-lld 构建，无 zig unwinder 污染；最小文件 rustc 加载不崩

## 设备堆工程记录（2026-10-29/30，五层攻坚）

**目标**：解锁内核内 `Vec`/`String`/`Box`（设备堆族）。机制选型：语言原生
`#[global_allocator]`（展开生成 local shim DefIds，collector sysroot 守卫
is_local 自动放行），比改守卫白名单或手写 extern shim 更符合语言语义。

**已落地**（adapts collector + cuda-device + mir-importer）：
1. `cuda-device::heap`：`DeviceAlloc`(GlobalAlloc) + 256 MiB 静态 bump
   arena（16 对齐、atomic cursor、dealloc no-op + reset 复用、volatile
   零填充避开未支持的 write_bytes intrinsic）。
2. collector 守卫精准化：`handle_alloc_error`（OOM 报告器，有 MIR 体）与
   `__rust_no_alloc_shim_is_unstable_v2`（检测桩）移出 SHIMS 拦截清单
   ——两者皆非分配入口，且桩有 no-op 设备定义。
3. mir-importer write_bytes intrinsic 分发臂：符号级 call →
   `__cuda_device_memset_u8`（cuda-device::heap 提供 volatile 循环
   backing，经 DeviceAlloc::alloc 恒真分支保证收集）。
4. foreign ABI 权威降级机制已实现（is_emitted_name 查询）——见第五层。

**第五层缺口（BLOCKED→IMPLEMENTABLE）**：`#[global_allocator]` 生成的
local shim（`__rust_alloc_zeroed` 等，bin crate local DefId，已入队收集）
符号与 alloc crate 调用站点的 foreign 权威组合，验证器两连拒：
- "call to an in-module mir.func must not claim external ABI authority"
  → 已实现 is_emitted_name 降级修复；
- 降级后 → "MirCallOp argument 0 type does not match callee signature"：
  NonZero/Alignment 结构在调用点与 callee 签名两处各自翻译，结构等价但
  类型句柄不同——需要导入器类型系统的结构化统一（dedup by structural
  type），属 MIR 类型系统的深水区。

**负例回归**：device_alloc_reject（无 allocator 时）保持编译期 fail-loud。

**复现**：device_heap 例（五段断言就绪）——第五层修复后直接
`cargo oxide run device_heap` 验证。

## 设备堆第五层攻坚结果（2026-10-31，fail-closed 边界确认）

四层落地后链路推进到【最后一个门】：`Result<NonNull<[u8]>, AllocError>`
（alloc_impl_runtime 的返回类型，Vec/String/Box 所有分配路径的公共汇聚点）
的 niche slot map——`Ok(NonNull<[u8]>)` 的 fat-pointer 载荷（ptr+len 16 字节）
与 integer claim 在 byte 8 重叠，mir-lower 的 provenance 门按设计 fail-closed
（"refusing to erase LLVM pointer provenance"，enum_layout.rs:424）。

攻坚过程（全部留痕）：write_bytes intrinsic 分发（mir-importer 臂 + collector
write_bytes→memset enqueue）；shim 定位从 cuda-device 挪进 kernel crate
（local DefId 可收集）；export 命名与 call 站点对齐（fqdn 而非 symbol）；
verifier 对 in-module+external 权威组合的双签名校验；诊断打印保留在
verbose 路径。

**结论**：Vec/String/Box 族解锁的剩余工作 = enum slot map 对 fat-pointer
niche（指针+长度对）的通用化——单点、范围明确。fail-closed 保证无误算风险。
capability 边界注释已写入 device_heap 例（保留全部断言为复现契约）。

## 设备堆第六层：slot map 通用化落地（2026-10-31，device_heap 五段真卡全绿）

analyze_pointer_overlap 新增 exact-shaped niche 分支：
- 触发条件：layout ∈ {Niche, Direct}；incoming field 与某 colliding claim
  【同 offset 同 size】；incoming 与 claims 均 byte-faithful；claim 内含
  【非指针载荷】（指针叶字节和 < claim size，排除纯指针-整数 pun）。
- 共存语义：指针 claim 保留（niche 读侧 = Err 的 ptr==null 判别），
  incoming 整数叶覆盖同字节（Ok 载荷写侧）。指针叶必须完全落在
  incoming 范围内，部分重叠仍 fail-closed。
- 配套：shim/memset 挪进 kernel crate；cursor/计数器 volatile 化
  （atomic inline-PTX 'l' 约束 llc-amdgcn 拒绝）；collector 转发收集
  （extern decl → 同符号定义；write_bytes call 站点直接 enqueue local def）；
  verifier 双签名校验（external 声明 + in-module mir.func 同存）。

**device_heap 五段真卡全绿**（RX 6950 XT）：heap_vec sum=285 cap=16 /
heap_string len=6 checksum=633 / heap_box .0=21 .1=11 / heap_realloc
sum=1 / heap_after_reset 84。Vec/String/Box 整族解锁。
矩阵与批清单同步（device_heap 入 102 例批）。

## 第六层收尾（2026-10-31，102 例批 device_heap 双维 PASS）

memset 符号曾经三条收集路径（normal callee / standalone root /
write_bytes 分发）各命名不同——importer 臂只能匹配其一。统一：全部
enqueue 站点把 memset export 钉为裸常量 "__cuda_device_memset_u8"
（模块内唯一定义），臂经 facts 钩子（mir_importer::memset_symbol）
读同一常量。批抽取器扩展：no_mangle fn 与 helper fn（arena_alloc）
随顶层项抽取。**102 例批最终**：llc-pass 99/102、integrated 99/102
（device_heap 双维 PASS；3 排除=抽取器局限）。

## 迭代器边界解锁（2026-10-31，enumerate 链真卡全绿）

两项 IMPLEMENTABLE 收敛：
1. **Closure-typed struct 常量**：const_alloc 对 Closure 类型合成
   zero-synthesis fallback（无 ADT 元数据；闭包体在 MIR locals，非常量数据），
   同路径的 usize 常量发零 MirConstant（同宽）。Enumerate::fold/sum 链、
   unwrap_or_else 全部真卡通过。
2. **值形态 Option**：copied() 后 unwrap_or / map_or（值闭包）/
   unwrap_or_else（非捕获闭包）真卡通过——此前"Fn-trait 分发被拒"的
   边界实为【引用形态 default】（Option<&T>::unwrap_or(&0)）。match 仍为
   引用形态的替代。

**验证**：iterator_adapters 八 kernel 逐值全对（新增 map_or_probe、
enum_fold_sum）；iter_unwrap_or_probe、iter_enum_fold 独立探针入批；
102 例批双维 99/102（device_heap 双维 PASS）→ 104/105 例批 101/105
（env 脚本化后环境重置免疫）。批清单 +5 例（device_heap/runtime_sync/
iterator_adapters/iter_enum_fold/iter_unwrap_or_probe）。

## rocprofiler 深挖收束（2026-10-31 第二轮，strace + 官方 hsa 复测）

官方 hsa-rocr（含 rocprofiler-register NEEDED）切换后复测：
- tool 初始化时长 0.00007s → 0.0014s（真实挂载，glog 确认）；
- 四模式（kernel/hip/sys/runtime-trace）+ csv 显式格式 + -d 输出目录
  仍零输出零文件；手动 LD_PRELOAD tool 同样静默；
- strace -f ioctl：app 正常运行（GPU 执行 0.5s），无被拒 kfd ioctl；
- rocprofv2 legacy 路线段错误（aqlprofile64 + register 同路径）。

**判定不变**：抓取通道需驱动侧 perf 事件支持（root / amdgpu.perfmon=1），
且 hsa-rocr→rocprofiler-sdk 的回调通告在重组栈上仍静默——非依赖缺失
（ldd 全解析、register 单例共享、官方 hsa-rocr 已通告）。fail-closed，
无误算风险。§8 计数器归因以 **HIP 事件对**（scripts/perf_events_attribution.c，
真卡验证 FFMA 21.39 TFLOPS=理论 99.5%）作为替代证据链。

## rocprofiler 第三轮（2026-11-01，stock 容器全链路解锁）

官方镜像 `docker.m.daocloud.io/rocm/dev-ubuntu-24.04:7.2.4-complete`
（Ubuntu 24.04.4 + ROCm 7.2.4 全栈，HIP 7.2.0/HSA 8.20.4）容器内复测，
GPU 透传 `--device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 999
--security-opt seccomp=unconfined`：

1. **kernel/runtime-trace**：可写 CWD（tool 要在 CWD 写 `.rocprofv3` 状态目录，
   ro 挂载 CWD 上 abort signal 6）→ rc=0，hip_api/kernel/memory_allocation/
   agent_info CSV 齐全。
2. **HW 计数器（--pmc）**：需 `--cap-add PERFMON --cap-add SYS_ADMIN`
   （默认 docker caps 无 CAP_PERFMON——重组栈上的"KFD perf 权限 BLOCKED"
   实为此墙 + 重组栈缺 register 链路两层）。SQ_WAVES probe、FETCH_SIZE/
   WRITE_SIZE/GL2C_HIT/MISS/TA_* 全部真实计数（counter_collection CSV）。
3. **gfx1030 计数器约束**：GL2C 块每 pass ≤4 计数器（12/7/5 组合一律
   error 38 "Request exceeds the capabilities"，栈对该 agent 不做多 pass
   拆分）；`SQ_INSTS_*/SQ_WAIT_ANY` 在 avail 列表存在但 agent 上不可采集；
   派生指标 FETCH_SIZE/WRITE_SIZE（4 桶 KB 表达式）单发可采。
4. **已知坑**：tool 的 abort signal handler 结束后 app 可能挂死空转
   （kernels still active）——外层 `timeout` 必备。
5. 计数器归因产出已入矩阵 §8 更新块（读放大 2.4%/写放大 2.1%/L2 命中 25%/
   waves 精确）。

**改判**：rocprofiler 抓取层 BLOCKED（重组栈）→ **FULL（stock 栈 +
容器 caps 条件）**。§8 计数器归因证据链由 HIP 事件对升级为真 HW 计数器。

### §8 收束（同日）

perf_baseline 规范化（长预热 6000 launch + 丢弃计时块 + 计时前 2×256MB
H2D 重灌）后 triad **515-547 GB/s = 90-95%**（三会话复现）——90% 线**达标**；
FFMA 口径修正（27.85 TFLOPS 上界 @2720MHz）实测 **92.3%** **达标**。快态
协议经验规律与伪影记录（posted-write 完成语义）详见矩阵 §8 更新块。
推翻的中间假设：denormal 数据效应（三相位诊断证伪——实为相位/重灌效应）；
ffma 21.5 上界（2100 MHz 误标定）。

## 容器优先工作流（2026-11-01，用户指令：host 不再放任何工具链）

又一次环境重置（~/opt 全灭）后，host 侧重组路线退役，改为**容器优先**：

- host 只保留：docker（VPN 代理现端口 7891；GitHub 凭证走 SSH，origin 已切
  git@github.com:pifu-zh/cuda-oxide.git）、仓库 checkout、~/.cargo+~/.rustup
  （挂载进容器复用，容器 glibc 兼容 host rustc/cargo/cargo-oxide 二进制）。
- 常驻容器 `gfx1030-dev`（镜像 rocm/dev-ubuntu-24.04:7.2.4-complete，
  /repo 挂外层仓库根——SIMT workspace 经 ../cuda-core 引用共享 crate，
  挂载必须盖住两层）。入口：`cuda-oxide/scripts/gfx1030_container.sh
  ensure|shell|exec CMD...`；`prepare_gfx1030_env.sh` 变为转发壳。
- 容器内容物（`setup_gfx1030_container.sh` 幂等重建）：micromamba、
  conda-forge CUDA 13.4 工具链（/opt/cuda13，bindgen 必需——cargo-oxide
  对 gfx 路径也硬校验 CUDA 13.0+ 工具链目录）、python3-pytest、hipinfo
  探针（/home/ubuntu/bin，契约格式 `agent N: gfx1030 (Marketing Name)`）、
  容器 gcc 重编的后端 dylib。建议 `docker commit` 快照。
- 容器内已验证：pytest 电池 **12 passed / 3 skipped**（hipinfo/identity/
  端到端全绿，比 host 时代还干净——无环境残缺项）；`CUDA_OXIDE_TARGET=
  gfx1030 cargo-oxide run vecadd` 真卡 **1024 元素全对**。环境件全清单：
  CUDA_TOOLKIT_PATH=/opt/cuda13 + LIBCLANG_PATH=/opt/rocm/lib/llvm/lib +
  BINDGEN_EXTRA_CLANG_ARGS（clang resource 头）+ LD_LIBRARY_PATH=
  host-hip/cuda-hip-shim/shim-lib + CUDA_OXIDE_TARGET=gfx1030（缺它管线
  默认产 PTX 形态 → embed 后 gfx loader 报 CUDA_ERROR_INVALID_IMAGE，
  已在 gfx1030_container.sh 内固化）。
- perf 探针（perf_baseline/triad4/denorm_ab）与 rocprofv3 抓取同样在容器
  内跑（--cap-add PERFMON --cap-add SYS_ADMIN 已含在 ensure 参数里）。

## 设备端 panic 消息格式化（2026-11-02 调查，IMPLEMENTABLE——精确阻塞点已定位）

目标：panic! 消息经 printf/hostcall 链打到 host 再 trap（现状 = 裸 trap 无消息）。

**管线现状取证**：`panic!("literal")` 在 opt-level 3 的 MIR 里被 rustc 内联成
「语句构造消息 + `asm!("trap;")` 终结子」的块；collector 侧 StrConstScan /
const_str_text（tcx eval + try_get_slice_bytes_for_diagnostics）能完整取出消息
文本（verbose 已证）。importer 在 `is_dropped_panic_call`（block.rs:87）整块
跳过——"materialized &str constant has no device lowering" 是既有设计。

**本轮实现到最后一厘米的部分**（已回滚，路径全部验证过）：
- cuda-device 根模块 `__cuda_panic_str(&'static [u8; 64])` 钩子（打印+换行），
  collector 侧 span 键注册表（collector const_str_text 提取 →
  record_panic_message(span lines) → importer 查询）+ 入队臂
  （find_cgu_definition_for_symbol 跨 CGU 符号查找）+ 例子。
- collector 顺带修复一个真 ICE：`find_cgu_definition_for_symbol` 的本地分支
  用 `module_children(LOCAL_CRATE)` 在该查询上下文不受支持
  （"not supported for this key"），改为 `mir_keys` 扫描（已保留提交）。

**精确阻塞点（四道验证器墙，全部实测）**：消息字节要从 importer 合成出
`ptr<ui8, as0, RawConst>` 级别的指针操作数，而：
1. `MirGlobalAllocOp` 结果指针必须 as1/as4（as0 被拒）；其 global_type 是
   数组时结果点子也是数组。
2. `MirCastOp` PtrToPtr 跨点子（数组→ui8）即使带 AbiBoundary authority 也被拒
   （"requires matching pointee/element types"）。
3. `establish_declared_pointer_type` 要求点子精确相同（只做同点子
   addrspace/kind retype）。
4. `MirRefOp` 要求 result pointee == operand 的点子（ref 消费的是点子类型的
   值，不是指针）。

**[2026-11-04 进展：地基件落地 bdc2416b0]** `build_array_op_from_bytes` 改为
`Option<rustc Ty>`：Some 臂保持 rustc-layout 步长与元素数交叉校验（全部既有
调用不变，1213/1213 测试过）；None 臂 = `dialect_element_byte_size`——从
dialect 类型直接导出元素字节大小（原始标量 + 其嵌套数组；tuple/struct 拒绝，
padding/字段序只有 rustc layout 知道）。这打通了「字节来自 MIR 之外」的数组
合成——panic 消息正是这种（collector 按 span 取文本，importer 无 rustc Ty）。
四道验证器墙中第 2/3 道（跨点子 cast / 点子精确）不再需要绕过：合成数组值
走 promoted-const 同款 MirRefOp（同点子 ref）+ GEP-0（正常 GEP 非 cast）。

**[2026-11-04 step 2a：ockl 传输探针双绿，0cc085855]** 直调
`__ockl_printf_append_string_n` 的两个探针 kernel 在真卡打印完整字符串：
(a) 模块级 constant 指针；(b) addrspace(1) 全局 → addrspacecast → generic——
**ockl hostcall 的字符串拷贝是设备侧行为，as1 全局指针直接可用**；panic
消息传输语义钉死（脚本 scripts/panic_probe/，链接配方在文件头注释）。

**[设计 v3（探针后定稿，替代 collector-registry 方案——消息字节就在 panic
块的 MIR ConstOperand 里，无需 TyCtxt）]**：
1. cuda-device 加真实 Rust 助手 `__cuda_panic_str(msg: *const u8, len: u64)
   -> !`：`__gpu_vprintf("%s\n", buf)`（debug.rs:193 既有钩子——真实 FnDef，
   mir-lower convert_vprintf → Step 17 ockl 展开 → metadata+ockl.bc 链接全部
   自动）+ `asm!("trap;")`（diverging，调用即终结，无需后继块）。
2. importer `is_dropped_panic_call` 臂：扫描块内语句的 ConstOperand（Allocated
   + provenance 跟随，const_alloc 机制现成）取消息字节 → `emit_promoted_
   immutable_global`（现成，去重）物化 as1 全局 → MirConstantOp(len) →
   发射**分歧调用**到 `__cuda_panic_str`（与被丢弃 panic 调用同形）替代裸 trap。
3. 唯一工程点：call 发射原语需接受**预物化 Value** 实参（现 emit_function_call
   从 value_map 翻译 mir::Operand 局部——合成值不在其中；helpers.rs 的
   MirCallOp 构造层加一个 pre-materialized 变体）。
4. 端到端例子 + 卡上测试（host stdout 捕获 "device panic: ..."）+ 双侧镜像
   评估（助手为真实 Rust，预计无需 prep-pass 手术 = 无镜像负担）。

**[2026-11-04 step 3：管线全链落地（代码完成，最后一环 = abort 时排水）]**
全部组件就位并经真机编译运行：(1) cuda-device `__cuda_panic_str(msg,len)`
助手——`#[device] extern` 声明 ockl begin/append_string_n（走 device-extern
管线的 declare 发射 + 原符号链接；**教训：裸 `unsafe extern "C"` 不带
DEVICE_EXTERN_PREFIX，collector 不识别 → declare 缺失 → lowered 验证
"Symbol not found"**），常量用数组值（`&[u8]` 切片常量无翻译 handler）；
(2) collector：check_panic_machinery 内按块扫消息（**取最长 &str——内联
panic 机器同时携带 file!() 位置串，"恰好一个"判据会漏**），(导出名,块号)
键注册 + 助手入队（**module_children 定向走 cuda_device::debug 路径——
全树扫描撞 8192 上限；扫中 generic 兄弟须 own_params 守卫防 Instance::mono
ICE**）；(3) importer 臂：emit_promoted_immutable_global 物化 →
**StaticAddress authority cast（Erased 全局指针 → 声明类型，这是全局指针
 establishment 的唯一合法通道；注意 ArrayToPointer 验证要求 source/target
同 addrspace 且 raw kind，PtrToPtr 要求同点子——三墙叠加的合法序列 =
Erased 全局 → authority retype → 同 as ArrayToPtr）** → emit_raw_call
（预物化 Value 变体）→ 既有 trap；(4) vprintf 门两处放宽（**真正生效的门
在 fn 首行 `!contains("@vprintf(")`——行 2143 的次级门是死代码**）+ declare
按符号去重（llvm-link 拒重定义）+ metadata 门（__ockl_printf 即注入）。
端到端：panic_message 例编译→加载→launch→**trap 真触发**（HSA exception
实证）；**最后一环 = 消息排水**：ockl preview hostcall 是 fire-and-forget，
缓冲在内核完成时排干，而 trap 中止队列 → 消息丢失（延迟自旋探针无效——
abort 路径不看缓冲）。**[2026-11-05 已收官：clean-exit + named-as4-global 形态]** 排水问题的
真正解法不是阻塞 hostcall（30M 周期自旋探针无效——preview 协议 fire-
and-forget，abort 路径不看缓冲；100ms globaltimer 探针挂死未复诊），
而是**绕开 runtime 排水**：(a) panic 消息全局改放 as4 常量内存 + tagged
rust-static key（`cuda-oxide:rust-static:v1:<len>:__cuda_panic_msg_<hash>`）
—— exporter 的 as4 命名分支会以嵌入符号名导出全局，host 端
cuModuleGetGlobal 按名解析（符号名 = `__cuda_panic_msg_<fnv1a32(字节)>`，
确定性可算）；(b) panic wave **干净退出**（arm 发 mir.return 替代
trap+unreachable）——kernel 正常完成则 hostcall 排水正常、host 可自由
memcpy；(c) 助手把 `panic: <msg>\n` volatile 写入 handoff 静态缓冲
（冗余通道）。host：get_global 按名读取 + 逐字节比对。**panic_message 例
真卡 PASS**（32 字节逐字节回读一致）。语义收窄已文档化（clean-exit 代替
queue-abort；format-arg panic 仍 plain trap；gpu_assert! 保持 trap）。
**教训：多级 mtime 时区错位（host 本地 vs 容器 UTC）让 cargo 误判 fresh，
dylib 迟迟不含新代码——strings 查特征字符串是定位手段；真正生效的门在
fn 首行而非中途**。1213/1213 回归过；批回归随 e5c7ee7ea 状态 101/105。

**[2026-11-04 step 2b：%s 探针裁定 + ptr 槽修复]** 独立 %s 探针
（vprintf("%s\n", {ptr} 缓冲) 走 Python Step 17 全链）暴露**真管线缺口**：
_emit_arg_slots 无 `ptr` 字段提升分支（信封回落保留 vprintf → llc 变参拒绝）
——双侧已修（ptrtoint 升 i64 槽）。**%s 语义裁定**：hipcc 的 %s 真卡正常
（对照 kernel 全对），但 ockl 协议的字符串拷贝发生在**设备侧**（ockl.bc 的
vprintf 实现对 %s 调 append_string_n 预拷贝进 hostcall buffer）；host 侧
格式化器**没有**指针解引用服务——我们把裸指针推成 raw slot → host 打出
乱码（2 字节）。**结论：panic 助手 v4 = 直调 ockl appenders**（cuda-device
内 `extern "C"` 声明 `__ockl_printf_begin/append_string_n`——真实 foreign
item 走既有 foreign-callee 机制；ockl.bc 链接门 `contains("__ockl_printf")`
自动触发；metadata 注入门需同步放宽到"出现 __ockl_printf 即注入"），长度
由 importer 侧已知（消息全局 N 精确），不走 %s、无 host 依赖。
**正确路径（下一步工程）**：str/byte-string 常量 → 设备端「按值物化」需要
`build_array_op_from_bytes` 的无 rustc-Ty 变体（现在强制要 rust_array_ty 做
元素布局校验），物化成数组值后走 promoted-const 同款
`MirRefOp(StaticAddress)` + GEP-0 + call。工作量 = fmt/常量字符串 lowering
子项目（DWARF 同族基础设施），非补丁级。

## soft TMA 微基准重建（2026-11-02，手写 ISA；小规模正确性已证，全规模记账 bug 开放）

旧微基准随 10-19 清空丢失。本轮从零重建：`cuda-oxide/scripts/soft_tma_bench/`
（gen_kernels.py 生成 7 个手写 gfx1030 ISA kernel + soft_tma_host.c 驱动）。

**设计**（aiter gfx1030 路线 + vmcnt 部分排空）：
- plain1（串行参照）/ df4_bN（深流水 gmem→VGPR→gmem 全拷贝）/ lds_bN
  （深流水 gmem→VGPR→LDS ds_write_b128→ds_read 归约消费）。
- 关键 ISA 事实（本轮实测）：**ds_write_b128/ds_read_b128 在 gfx1030 完全可用**
  （单条 16B LDS 写——推翻旧档案"直通 LDS 仅 b32"对 DS 路径的适用性；该结论
  只对 MUBUF 直通成立）；wave32 下进位加法须 `vcc_lo` +
  `v_add_co_ci_u32_e64 ..., null, ...` 形态；`v_add_u32` 不存在（用
  `v_add_nc_u32`）。

**装载链四坑**（全部实测排掉）：COv6 被驱动拒（管线 Phase-1 已知教训复现）
→ 必须 `-mcode-object-version=5`；ld.lld -shared 对 COv6 输入静默不产出
ET_DYN（v5 输入才正常）；描述符 SGPR 预算必须 ≥ 实际用量（漏了 =
hipErrorNoKernelImageForDevice）；结果布局/累加器 VGPR 不得与流水线组寄存器
重叠。

**已证**：小规模（blocks≤40、qpb≤512）lds/df4/plain 全部逐值正确——深流水
+ ds_write_b128 + vmcnt(B) 部分排空的机制成立；plain1 全规模（320 block）
也正确（423-427 GB/s，N 口径 73-74%）。

**开放（诚实记录）→ [2026-11-04 已闭环]**：全规模"~34% 提前退出/FAIL"的
根因 = **gen_kernels.py prologue 块基址公式错误**：`s_lshl_b32 s11, s10, 4`
（块基址 = wg×qpb×16），而一个 quad = 32 lane × 16B = **512B**（正确 = <<9）。
后果：32 路块切片以 26,112B 步长重叠，dst 只被写到前 9,165,312 B，其余未写；
所报带宽 = 9.16MB footprint 上的事务速率（df4/lds ≈29× 虚高）且 L2 常驻。
lds 的 5% 容差验证与 block-0 sum 指纹对 4000B 周期 pattern 的基址漂移盲视，
仅 df4 的逐字节 memcmp（布局敏感）暴露——期间被误判为"vmcnt 竞态""调度
饥饿""运行时 kernarg 乱序"等假说全部证伪（调试器自身的 region 步长 ÷4、
dst 步长错位、atoi 占位符 pinning 三个伪影亦被排除）。连带修正：
(a) 计数器仪表第二半 `v154 += b` 增量此前漏发（cnt 恒 = (qpb/8+1)×b 谎报）；
(b) verify launch 需 memset→hipDeviceSynchronize→launch 屏障，pass 间需
hipDeviceSynchronize（大 fill 与 compute kernel 无屏障竞态出洞）；
(c) vmcnt(N) 流水恢复（"vmcnt 竞态"从未成立），6 变体 × 8 轮转 × 320 块
全 verify ok。
**[2026-11-04 诚实全规模数字]**（分母 = 经验上限 575.5 GB/s；256-bit @ 18G，
由纯读 98.3% + triad 90-95% + 本轮 lds 96.1% 三独立源共证；卡 0x73a5/16GB
内存配置 256-bit，ROCm props busWidth=256 正确，vBIOS 名"6950 XT"不代表
512-bit）：lds_bN = 553 GB/s = **96.1-96.2%**（完美替代档）；df4_bN =
435 GB/s = 75.6-75.7%（良好档，copy 形态开放优化）；plain1 = 464 GB/s =
80.6%。旧 78.4% 数字正式作废（同族测量伪影）。详录见
research/gfx1030-substitution-catalogue.md §3。

## cp.async 软仿真 lowering（2026-11-03，Python Step 18 落地，Rust 镜像待做）

用户指正 C 微基准不是 kernel 优化本体——真正的落点是 **Rust 路径的 cp.async
lowering**。取证：cp_async 在 gfx1030 无任何 lowering（24 个裸
`llvm.nvvm.cp.async.*` 调用直达 llc，cp_async_small 构建 exit 101）。

**Python Step 18 已落地**（translate_amdgcn.py）：
- `cp.async.ca/cg.shared.global.{4,8,16}` → 同步向量 load(as1) + LDS store(as3)，
  dst/src 操作数文本原样透传（SSA/全局符号/常量 GEP 表达式 llc 全部原生物化；
  深度感知逗号切分处理 GEP 内嵌逗号）。
- `.s` 零填变体（i32 src_size）→ 字节粒度 load + select 零填（const src_size：
  只读前 S 字节 + 常量零填；runtime：smin 钳制 + select，保证不越界），
  按 dword zext/shl/or 组装存回。
- `commit.group` / `wait.all` / `wait.group.N` → 删除（同步仿真：程序序即
  发射序；LDS 同 wave 程序序可见）。
- `cp.async.mbarrier.arrive[.noinc][.shared]` → 删除（异步完成跟踪在同步
  仿真下天然满足；桥接相位由显式 mbarrier_arrive 驱动）。
- 语义收窄记录：依赖 tx 字节计数的 mbarrier 桥接模式须改用 wait_all。
- IR 级 lowering 的优势：llc 原生物化所有地址形态，无需手写符号/GEP 物化
  （对比：微基准需要手写 v_mov/进位加法链）。

**已证**：cp_async_small 完整 .ll（14 调用点）全部 lowering 零残留，
gfx1030 llc 通过（`--filetype=obj` rc=0）。

**开放（诚实记录）**：真卡 E2E fault——test_cp_async_4 GPU fault
（0x798eb43ff000，Page not present）；compile_all/bridges 跑完但 out=0
FAIL（SMEM 读回为零 = cp.async 的数据未进 LDS 或 out 写回未落地）。
疑点：例程把 SMEM 指针 `as *mut u32` 转成 generic 再传给 cp_async（原
intrinsic 声明 as3 但值经 generic 铸造）；lowering 的 as3-annotated store
与 generic 值的交互需要一轮 gdb/printf 定位。**修复前不得宣称 cp.async
SUPPORTED**。Rust amdgcn.rs 镜像同步待做。

### cp.async E2E fault 根因 + 修复（2026-11-03 续）

**fault 根因（ptrtoint 仪表化定位）**：index_1d helper 返回 -1 哨兵 →
cp.async 的 GEP 索引 = -1 → src = d_in - 4 → 4B load 跨入未映射页 → fault。
链路：Step 5 给 index_1d helper 定义追加了 ntid 形参并在体内引用 %ntid_*，
但 **MIR 导入的调用点只传 frame ptr**（1 参调 4 参函数）——llc 内联缺参 =
poison，垃圾寄存器决定 helper 走计算路径还是哨兵路径。

**修复（Step 5b）**：prep pass 给 index_{1,2,3}d 调用点补齐
`i32 %ntid_x/y/z` 三实参（frame ptr 形态覆盖 as5/as0/裸 %name）。
**嵌套检测器**：index_1d 体内调 `one_dimensional_launch(frame)`（1-D
launch 形状检测 = ntid_y==1 && ntid_z==1）——同一缺参问题一层嵌套——
regex 覆盖 `_dimensional_launch` 后缀（注意 mangling 是 `22one_...`，
数字前缀与 `_dimensional` 之间有 `one`，不能 `[0-9]+_dimensional`）。
幂等性已测（二跑不双补）。

**修复后真卡**：test_cp_async_4 **PASS**（out[0]=100= input[0]，rc=0
无 fault）；compile_all 无 fault（该 kernel 无 out 参数，exercise 形式）。

**剩余开放**：mbarrier_bridges kernel 全序列挂起（逐桥 early-return 均
独立完成且桥 1 数据正确；全序列挂点在桥间转移，疑 mbarrier_init 相位
重置与仿真状态的交互）。**修复前 cp.async 报"部分 SUPPORTED"：基本
ca/cg 形式（4/8/16B + zfill）上卡数据正确；mbarrier 桥接模式待修**。

### bridges 挂起根因 + 修复（2026-11-03 续二）——cp.async 全绿

**挂起根因**：bridges kernel 的 4 桥中桥 3/4 用 `init(count=2)` + 
`.noinc` arrive + 显式 arrive = **需要 2 次递减**才归零；Step 18 首版把
cp.async.mbarrier.arrive 全删 → 只剩显式 arrive → pending 2→1 永不归零
→ test_wait 自旋挂起。桥 1/2 (count=1) 单 arrive 归零所以通过。

**修复**：cp.async.mbarrier.arrive（plain 和 .noinc）= 一次正常 arrive
（递减 1）：plain 的"tx 计数递增"在同步仿真下天然满足；.noinc 的"不递
增"指跳过 tx 跟踪——两者 net = 递减 1。lower 到 `__port_mb_arrive`
（bar 操作数文本原样透传，llc 物化 SSA/符号/常量 GEP）。

**真卡（三 kernel 全 PASS）**：test_cp_async_4 out[0]=100 ✓；
compile_all 无 fault ✓；**mbarrier_bridges out[0..3] = input[0..3]
（memcmp 通过，无挂起无 fault）** ✓✓✓。

**cp.async = SUPPORTED**（SOFTWARE EMULATION 类，同步形态）：ca/cg
4/8/16B + zfill(.s) + mbarrier 桥接全部真卡逐值验证。语义收窄：异步
重叠（vmcnt 深流水）为后续优化（微基准已证机制）；tx 字节计数跟踪
内化于同步语义。

### Rust 镜像 + 集成管线验证（2026-11-03 续三）——cp.async SUPPORTED 落地

Rust amdgcn.rs Step 18（rewrite_cp_async）+ Step 5b（append_index_call_
ntid_args）镜像落地（fmt-string 的 `{j-1}` 需预计算 jm1；usize 减法下溢
需 jm1 移入 else）。集成管线真卡：cp_async_small（ca 4/8B 全 PASS）+
cp_async_zfill（src_size=4/2/0 三测试 PASS——第三测暴露 smax 钳底需求：
runtime src_size=0 时 smin(0, sz-1) 下溢 -1 GEP 跨页 fault，修为
smax(sz-1, 0) 钳底）。批回归 101/105 持平（Step 18/5b/smax 无回退）。
能力矩阵 cp.async 行更新 SUPPORTED（软仿真同步形态）。docker snapshot
镜像（72GB 陈旧层）清理后磁盘 38%。
