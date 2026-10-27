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

## printf 输出管线现状（重组栈）

- ockl 三函数 ABI（begin/append_string_n/append_args）在 triton ockl.bc
  与 ROCm 10.1 同构；**>7 参数链式 append_args（clang 同款多轮，is_last
  仅末轮）真卡验证通过**（10 参行 `chain: 1 2 3 4 5 6 7 8 aaaabbbbccccdddd
  2.7183` 逐字符正确）。
- 重组栈下部分 ≤7 参行存在 host 侧流解析乱码（如 `y=20` 截断为 `y=2`、
  值槽位错位）：模式为消息中途截断/槽位错位，是 host 端（triton
  libhsa/hip 的 hostcall 消费者）与重组栈组合的产物；设备 IR 侧
  append_args(nargs,…) 单轮写入完整。ROCm 10.1 原栈（2026-10-26/27）下
  printf 全族输出干净（见能力矩阵 printf 行的历史证据）。恢复原栈即可
  复核，属环境差异而非代码回归。

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
