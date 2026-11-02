---
name: rocm-adaptation
description: AMD ROCm 软件栈的 GPU 适配知识——栈结构、HIP/HSA Runtime、Code Object 编译链接与 kernel 装载、profiler 抓取、容器化工作流、版本兼容与已知坑。Use when 在 ROCm 平台上适配语言/工具链、诊断 kernel 装载或性能问题、搭建可复现 ROCm 环境、或使用 rocprofiler 抓取计数器。配合 gpu-adaptation-methodology 方法论、目标架构的 Architecture Profile（如 gfx1030-rdna2）、以及具体语言集成 skill（如 rust-gpu-toolchain）使用。
---

# ROCm Adaptation

前提：已读 `gpu-adaptation-methodology`（证据纪律、缺口因果定位、替代策略）。
硬件相关的能力与限制在 Architecture Profile（如 `architecture/gfx1030-rdna2/`），
语言工具链相关在对应 Integration skill——本 skill 只写"ROCm 软件栈如何工作"。

## 1. 栈地图（自上而下）

```
语言工具链（如 rustc 自定义后端 / hipcc）
  ↓ 产生
Code Object（ELF-amdgpu，含 .text + kernel descriptor .kd）
  ↓ 装载
HIP Runtime (libamdhip64) ⇄ rocprofiler-register（profiler 注入点）
  ↓
HSA Runtime (libhsa-runtime64) → AQL 队列 / aqlprofile（事件页、计数器）
  ↓
libdrm_amdgpu + /dev/kfd（内核驱动；perfmon 类 ioctl 需 CAP_PERFMON/SYS_ADMIN）
  ↓
GPU（ ISA = 架构代号决定，如 gfx1030）
```

关键交叉点（适配最常卡住的地方）：
- **Code Object 版本与 ISA 代号**：加载器按 `-mcpu`/`--offload-arch` 产生的
  ISA 代号匹配执行架构。交叉编译产物里的代号会被 hipcc 的 CC 值误导
  （实测：CC 10.3 被误读为 sm_103 式路由）——装载失败先核对产物里的真实 ISA。
- **comgr**（编译对象管理器）：Runtime 按 soname `libamd_comgr.so.3` dlopen
  ——只有无版本名的文件会表现为 `hipInit` 返回 NO_DEVICE（极难关联）。
- **rocprofiler-register**：官方构建的 hsa 带此 NEEDED（profiler 注入点）；
  非官方构建（conda 等）的 hsa/hip 没有它 → profiler 工具挂载了也零回调。

## 2. Kernel 装载与启动

- 装载路径有三条，行为不同，诊断时先确认走的是哪条：
  ① 文件工件（.hsaco/.co 落盘后 `hipModuleLoadData`）；
  ② 内嵌 bundle（code object 塞进宿主二进制的载荷槽，装载器按 fatbin 惯例
  提取）；③ 未链接 code object 经 comgr 现场链接。
- kernel 启动按 **kernarg 槽**传参：槽数必须 ≥ 签名形参数，否则遍历元数据
  越界崩在 libamdhip64 内部（不是用户栈）。打包结构体参数的 padding 不是槽。
- **stream/event/pool/managed** 全部经 HIP 可用（含双流并发 2.17× 实测、
  MemPool 统计、managed 内存往返）；设备端自旋等待必须用 volatile 链，
  普通读循环会被优化折叠成瞬时完成。
- 设备端动态并行（CDP 类）硬件不支持；host 侧二次 launch 是替代。

## 3. 设备端 printf（hostcall 协议）

- 机制：模块元数据声明 `amdgpu_printf_kind="hostcall"` → Runtime 经 implicit
  kernarg 注入 hostcall buffer → 设备端调 ockl 助手
  （`__ockl_printf_begin` / `_append_string_n` / `_append_args`）。
- 协议要点：`append_args` 每轮 ≤7 个值槽（轮次编码进 handle 位域），
  返回的 handle 必须接力给下一轮，`is_last` 仅末轮置位——漏接力表现为
  输出错位/截断/乱码。`append_string_n` 长度**含 NUL**。
- 格式串必须是模块级 addrspace(1) constant；参数打包遵循 C vararg 提升
  （i8/i16→i32、f32→f64）。动态格式串在信封外（保留 vprintf 链接期失败）。
- 本项目未做：panic 消息走此链（str 常量的设备端按值物化缺基础设施，
  详见上游研究档案）。

## 4. Profiler 抓取（rocprofv3）

- **通道前提**（逐条核对，缺一即"静默零输出"）：
  1. hsa 必须是官方构建（带 rocprofiler-register NEEDED）；
  2. 进程需 CAP_PERFMON + SYS_ADMIN（计数器 ioctl 的权限墙）；
  3. 工作目录可写（工具往 CWD 写 `.rocprofv3` 状态目录，只读挂载上直接
     signal 6）；
  4. `--pmc`/trace 类必须显式给选项，空跑收集 = "No tracing options"。
- **gfx1030 级约束**（其他架构需重测）：L2C 块每 pass ≤4 计数器且栈不自动
  多 pass 拆分（超限报 error 38）；部分 `SQ_INSTS_*` 在 avail 列表存在但
  采集不到；派生指标（FETCH_SIZE/WRITE_SIZE）单发可采。
- 工具 abort 后应用进程可能挂死空转——外层 `timeout` 必备。
- 可用指标清单：`rocprofv3 --list-avail`；派生表达式（`_sum` 归约）在列。

## 5. 环境与容器（可复现工作流）

- **容器优先**：宿主只留 docker/仓库/语言工具链挂载，重资产全进容器。
  实测要点：挂载必须覆盖"被引用的兄弟目录"（workspace 级 path 依赖）；
  容器内用户 uid 对齐宿主避免产物归属混乱；GPU 透传
  `--device=/dev/kfd --device=/dev/dri --group-add <video gid> --group-add
  <render gid>` + seccomp=unconfined + 上述 caps。
- 官方镜像（`rocm/dev-ubuntu-24.04:<ver>-complete`）自带完整用户态
  （hipcc/comgr/ockl 位码/rocprofv3），从镜像抽取用户态到宿主
  （`docker cp`）也是一条恢复路径，但注意传递依赖会连坐（libxml2/icu 级）。
- **环境易变性假设**：工具链目录、凭证、网络代理都可能消失。先探测再使用；
  容器内容物用幂等 setup 脚本重建并 `docker commit` 快照。
- CUDA toolkit 目录常被工具链构建校验（bindgen 解析 cuda.h）——用 conda-forge
  的 `cuda-nvcc`/`cuda-cudart-dev` 组件拼装即可满足（布局
  `targets/<arch>-linux/include` 是被认可的形态）。
- 传递依赖连坐：容器构建的库拉 Ubuntu 运行库（libxml2.so.2、libicu*.so.74），
  conda 版 soname 不同不能替代——从同版本容器 `docker cp -L` 抽取。

## 6. 版本兼容注意（截至 ROCm 7.2.4 实测）

- `append_string_n` 长度含 NUL：旧栈（10.1 时代）不含可过，7.2 起必须——
  跨版本复验时先核协议细节。
- `hipHostMalloc` 分配 = 真钉页（带宽 2.0-2.1× pageable）；`hipHostRegister`
  注册已分配内存：roundtrip 可用但**带宽停在 pageable 档**（7.2.4 + 消费级
  驱动实测）——注册只有 API 语义价值。
- H2D/D2H pageable ~14 GB/s、pinned ~28 GB/s、D2D copy 完成语义有假象
  （见方法论坑表），triad 混合流见 Architecture Profile。
- `hipLaunchKernelGGL` 与 `<<<>>>` 等价；kernel 名必须用 mangled 名装载。

## 7. 生态坐标

- **AITER**（AMD 通信/算子库）：架构特化分支是替代实现的最佳源码参考——
  找它对 gfx942 专用指令的退化路径。
- **Triton / Composable Kernel**：算子级实现参考；对本项目的可直接复用结论
  是数据搬运的软件流水模式（批量发射 + 计数器部分排空 + 双缓冲）。
- 上游 gate 常按架构代号（gfx90a/gfx942/gfx110x…）写死；gfx1030 落在
  "消费级 RDNA2"档，很多数据中心特性直接被 gate 掉——先查 gate 再实现。

## 坑表（本项目实测）

- `hipInit` NO_DEVICE → 先查 comgr 的版本化 soname，再查驱动权限，最后才怀疑设备。
- profiler 静默零输出 ≠ 工具坏了：按第 4 节四条前提逐项核。
- 跨栈（conda 官方混搭）用户态拼装：`LD_LIBRARY_PATH` 勿前置带自带的
  libLLVM 的目录——会与系统 libclang 错配（符号缺失类错误）。
- bindgen 需要 clang resource 头：`BINDGEN_EXTRA_CLANG_ARGS` 指向 ROCm clang
  的 resource include 目录，否则 stdint 类基础头报错。
- 只读挂载 CWD 会杀掉工具（它们往 CWD 写状态）——先 `cd` 到可写目录。
- 环境重置类灾难：持久件推远端 git + 容器 commit 快照，重建走幂等脚本。
- 容器内构建产物为 root 属主（容器内默认 root 运行）——宿主侧 git 操作忽略
  或 sudo 清理；宿主 uid 对齐进容器可避免（镜像 ubuntu 用户 = uid 1000）。

## 8. 跑通第一个示例（具体入口，2026-11-01 盲测验证路径）

```bash
# 宿主侧：确保常驻容器在位（首次会自动创建）
<repo>/cuda-oxide/scripts/gfx1030_container.sh ensure
# 容器内跑示例（CUDA_OXIDE_TARGET 等环境已固化在入口脚本）
<repo>/cuda-oxide/scripts/gfx1030_container.sh exec bash -c \
  'cd /repo/cuda-oxide && cargo-oxide run vecadd'
# 期望：✓ SUCCESS: All 1024 elements correct!
```

首次编译约 1-2 分钟（后端 dylib 用容器内 gcc 重链）。验证设备侧证据：
对产物 `.hsaco` 查 code object metadata 的 `amdhsa.target`（应为
`amdgcn-amd-amdhsa-unknown-gfx1030`），与 `hipinfo` 报告的 agent ISA 对齐，
排除 CPU 回退。
