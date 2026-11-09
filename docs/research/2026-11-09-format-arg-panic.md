# 2026-11-09 format-arg panic 消息渲染：源级宏重写收口 + gfx-m1 容器迁移

## 结论

能力矩阵「panic! 消息设备上报」双通道收口：

- **字面量路径**（已有，2026-11-05）：MIR 级 as4 常量 + 确定性回读，PASS 不变。
- **format-arg 路径**（本轮新增）：kernel 宏在源级把显式
  `panic!("fmt {} {}", a, b)` 重写为 gpu_printf 机制的报告块 + 裸 `return`
  ——真卡 stdout 渲染 `panic: value = 300 / limit = 100`，干净退出（无
  HSA exception / coredump 噪音）。panic_format 新例实证。

## 设计抉择：为什么走宏层而不是 MIR 层

先侦察了 MIR 路径：rustc 1.99 把 format_args! 全部内联进 panic 块，形态完
整可提取（模板 = 字节串常量 `\x08value = \xc0\x0e / limit = 100\x00`，新
fmt 模板编码：`\xNN` 长度前缀 piece + `\xc0` 参数占位符，常量参数 100 已
被 const-fold 进 piece；参数 = ArgumentType::Placeholder { value: NonNull,
formatter: fn }，数据流可回溯 `&raw const (*_16)` ← `_16 = &_10` 拿到被引
用 local 及其类型）。MIR 提取+importer 发射可行但要动 collector/importer
两侧并解析新模板编码。

改走宏层的决定性理由：**gpu_printf! 宏已经解决了同一个问题**——Rust 风格
占位符解析、C %spec 选择（GpuPrintfArg trait 单态化定 spec）、C vararg 提
升、repr(C) 参数打包、`__gpu_vprintf` 调用，全链 printf FULL 已验证。panic
的 format_args! 与 gpu_printf! 的输入同构（字面量 fmt + 参数表达式）。宏
层重写 = 复用已验证机制 + 新代码不到 150 行 + panic_fmt 块根本不进 MIR
（collector/importer 零改动）。

关键事实修正：clean-exit 之后 ockl 排水是**发生的**（panic_format 实证）。
fc293f95f 当初"ockl 不可用"的结论只对 trap 设计成立（trap 中止队列丢缓冲
）；换 clean exit 后 ockl 通道复活——format-arg 路径正是用它。

## 实现

- `cuda-macros/src/printf.rs`：`panic_report_expr(fmt, args)` ——以
  `"panic: {fmt}\n"` 构造 GpuPrintfInput 复用 gpu_printf_impl。
- `cuda-macros/src/kernel/mod.rs`：`rewrite_format_arg_panics` —— syn
  VisitMut 走 kernel 体；**语句级 `panic!(...);` 是 `Stmt::Macro` 而非
  `Stmt::Expr(Expr::Macro)`**（首轮漏配的根因）；表达式位置同规则；闭包体
  跳过（closure 的 return 语义不是 wave 退出）；替换块以 `return;` 结尾，
  类型 `!` 在任何位置可接受。范围门：unit-return kernel + 字面量 fmt +
  ≥1 实参。
- collector.rs：探针清理 + 过时注释修正（"format-arg 仍 plain trap" 的旧
  注释改为指向宏重写；真正保持 trap 的是 core 内部机械 panic、设备函数内
  panic、非 unit kernel panic——三者均无干净的 wave 退出路径）。
- 展开测试 3 项（重写形态 / 范围外保留 / 闭包保留），cuda-macros 147/147。

## 通道语义（矩阵同步）

| 路径 | 通道 | stdout | 确定性回读 |
| --- | --- | --- | --- |
| 字面量 `panic!("msg")` | as4 常量 `__cuda_panic_msg_<hash>` + handoff 缓冲 | 无（helper 不走 ockl；例程 doc 的 stdout 期望系 ockl 时代陈旧注释，已修正） | ✅ cuModuleGetGlobal 逐字节 |
| format-arg `panic!(fmt, args)` | gpu_printf → ockl hostcall（`amdgpu_printf_kind=hostcall` 元数据） | ✅ 排水到 stdout（异步于 sync，进程退出前到达） | 无（内容运行时才知道） |

双通道互斥是设计而非缺陷：字面量的确定性回读不需要内核真的执行到 panic
（常量在模块里），它的证据是 wave 行为；format-arg 的内容只有运行时存在，
stdout 排水就是它的通道。

## gfx-m1 容器迁移（用户指令：cuda-rust 工作全部容器内）

新容器 `gfx-m1`（image `gfx-rocm:7.2.4`，host 无原生 ROCm——7.2.4 即唯一
栈；GPU /dev/kfd + /dev/dri 直通 ✓）。与旧 gfx1030-dev 的最大差异：
**挂载 /home/zhuo 全量**。适配结果：

- 持久件落 `/home/zhuo/opt`（容器重建可存活）：micromamba 2.10.0 +
  CUDA 13.4 toolkit（`/home/zhuo/opt/cuda13`，conda-forge cuda-nvcc/
  cuda-cudart-dev/libcurand-dev）——CUDA 13 仅满足构建期契约
  （cuda-bindings bindgen 的 cuda.h + 版本门），运行期 GPU 全程 HIP/ROCm。
- 无需 rustup/cargo 符号链接（真 home 在容器内）；rustc 1.99.1 SIMT pin ✓。
- apt 件（pytest/strace/bzip2）需 `-u root`（exec 默认 ubuntu 用户），
  容器重建后需重装。
- 新 INNER_ENV 契约：`PATH=/home/zhuo/.cargo/bin:/opt/rocm/bin` +
  `RUSTC_BOOTSTRAP=1` + `CUDA_OXIDE_TARGET=gfx1030` +
  `CUDA_TOOLKIT_PATH=/home/zhuo/opt/cuda13` + `LIBCLANG_PATH=/opt/rocm/
  llvm/lib` + `BINDGEN_EXTRA_CLANG_ARGS=-I/opt/rocm/lib/llvm/lib/clang/22/
  include` + `LD_LIBRARY_PATH=<repo>/host-hip/cuda-hip-shim/shim-lib` +
  显式 `CUDA_OXIDE_BACKEND=<repo>/.../target/debug/librustc_codegen_cuda.so`。
- 旧容器 root 属主的 target 树需一次性 `chown -R ubuntu`（权限错配会以
  Errno 13 呈现）。
- 无 CAP_PERFMON/SYS_ADMIN（rocprofiler 抓取在此容器不可用；性能抓取需
  回旧镜像形态或加 cap）。
- **批跑必须 CARGO_NET_OFFLINE=true**：逐例新鲜抽取目录无 Cargo.lock，
  cargo 会按 host 侧 ~/.cargo/config.toml 的 127.0.0.1 代理查 registry——
  容器内无该代理，首连即死（首轮批 0/105 build-fail 的根因；单例构建因
  缓存温热而幸存）。离线 + 温热 registry 缓存即解（复验过单例离线构建）。
- vecadd 1024/1024、panic 双例、fp8 穷举均在新容器复现 ✓。

## 测试统计（本轮）

| 项 | 结果 |
| --- | --- |
| panic_format 真卡（gfx-m1） | stdout 渲染 PASS + 干净退出 |
| panic_message 真卡回归 | readback PASS（32 字节） |
| vecadd 真卡回归 | 1024/1024 |
| cuda-macros 测试 | 147/147（3 新增） |
| fmt（改动文件） | 我的区域全部清洁 |
| 批回归（105 例） | 见矩阵批量验证行 |
