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
  （1.1.0/97f5574）。**剩余问题**：`--kernel-trace/-hip-trace -o … -- app` 均 exit=0 但不产
  任何输出文件。诊断已定位：tool 库**挂载成功**（tool initialization/
  finalization 日志来自 librocprofiler-sdk-tool.so 内部，手动 LD_PRELOAD
  同样），但 hsa→profiler 的 agent/缓冲注册没有发生（0 kernel 记录、
  0 输出）。下一步实验排序：① 用官方配对的 hsa-runtime_7.2.4 deb 替换
  rocm2 的 hsa（当前 hsa=conda-forge 7.2.53211 构建，agent 注册协议可能
  有差异）；② HSA_TOOLS_REPORT_LOAD_FAILURE=1 + rocprofiler trace 日志
  级环境变量（ROCP_ROCPRF_LOG_LEVEL=trace 已试无增量）；③ 检查
  rocprofiler-sdk 是否需要自有 rocpd daemon。conda-forge 无任何
  rocprofiler 包，官方 deb 是唯一来源。

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
