# Case Study: ROCm Rust on RX 6950 XT（gfx1030）

首个完整案例。目标：在 RDNA2 消费级卡上最大化 Rust GPU 编程能力，建立可复现、
可验证、可持续扩展的方案。时间跨度 2026-10-19 → 2026-11-02。

## 素材（原始研究档案，位于仓库根 docs/research/）

- [`docs/research/gfx1030-substitution-catalogue.md`](../../../docs/research/gfx1030-substitution-catalogue.md)
  —— 替代方案目录（六分类法、每条带证据；soft TMA 深度案例）
- [`docs/research/2026-10-28-environment-recovery.md`](../../../docs/research/2026-10-28-environment-recovery.md)
  —— 环境考古与恢复（含容器优先改造、panic 消息调查）
- [`docs/gfx1030_support_matrix.md`](../../docs/gfx1030_support_matrix.md)
  —— 能力矩阵（批量验证终局、§8 性能判定）
- 上游仓库：本仓库 adapt-rocm 分支（代码、脚本、测试、提交历史均可见）

## 关键时间线（决策 → 结果）

| 阶段 | 决策/事件 | 结果 |
|---|---|---|
| 起点 disaster（10-19） | 环境被清，未归档的档案随宿主消失 | 教训固化为方法论第 8 节："档案随仓库活" |
| 迁移 adapt-rocm（10-25） | 上游重构 365 commits，nightly→stable 1.99.1 | HIP shim 架构（112 导出 cu*-ABI cdylib） |
| async/闭包根因链（10-25） | C 探针洗清 shim 面 → gdb 证段错误 → 三层根因 | host_closure 12 形状×4 路径真卡全绿 |
| 批量验证体系（10-26） | 242 例扫描 → 99 例批双维度 | 之后演进到 105 例 101/105 |
| 设备端 printf（10-26/28） | vprintf→ockl hostcall 协议取证（llvm-dis 反推 ABI） | 8 族+>7 参链式真卡零乱码 |
| 环境两次被清（10-27/28） | 用户级重组：conda 真 gcc+ROCm 7.2 用户态 | 9 坑位全录；perf 基线跨栈零漂移 |
| 抓取层攻坚（10-28→11-01） | rocprofiler 静默 → register NEEDED → 容器 CAP_PERFMON | BLOCKED 改判 FULL；GL2C 计数器归因 |
| 设备堆六层（10-30/31） | `#[global_allocator]` + bump arena + slot map niche 通用化 | Vec/String/Box 整族真卡 |
| §8 性能收束（11-01） | 暖态协议+H2D 重灌+FFMA 口径修正 | 带宽 90-95%、FFMA 92.3% 达标 |
| 容器优先（11-01） | host 净空，全进容器；批回归对齐 101/105 | 容器内 pytest 12/12 + vecadd 全对 |
| panic 消息（11-02） | 调查到最后一厘米：四道验证器墙 | 诚实回滚+归档；ICE 顺带修复 |

## 根因表（已解决问题的浓缩）

| 症状 | 真根因 | 教训 |
|---|---|---|
| kernel 崩在 libamdhip64 内部 | kernarg 槽数 < 签名形参数（元数据遍历越界） | 每形参必有槽；padding 非槽 |
| printf 输出错位/截断 | handle 未从 string_n 返回值接力 + padding 槽误当参数 | 链式协议逐轮接力 |
| hostcall 乱码 | `__oclc_wavefrontsize64`=0（默认应 1） | 配置位码逐项核对 |
| hipInit NO_DEVICE | comgr 无版本化 soname | `.so.3` 命名契约 |
| profiler 零输出 | 非 register NEEDED 构建 + 无 CAP_PERFMON + 只读 CWD | 四前提逐项核 |
| 设备代码"跑通"实为 CPU 路径 | 无设备侧指纹验证 | 输出带设备证据 |
| 后端改了行为不变 | dylib 未按 target 重编 | 固定怀疑序 |
| FFMA "99.5% 理论" | 上界按 2100 MHz 误标定（实际 2720 钉死） | 上界口径要跟实测时钟 |

## 失败实验（保留价值）

- soft TMA 90% 目标：78.4% 收线（良好档）——b32 直通宽度是指令密度结构性上限。
- float4 向量化 triad：反比 scalar 慢（四会话一致）——流式基准规范形态定为 scalar。
- denormal 数据带宽假说：三相位诊断证伪（是测量相位+H2D 重灌效应）。
- panic 消息快速实现：四道验证器墙，回滚归档，正路=常量字符串 lowering 子项目。
- 非 move 闭包引用捕获：HMM 依赖，NOT FEASIBLE；move 捕获等价已验证。

## 可复用产物

见 `cuda-oxide/scripts/`（perf 基线与探针、容器工作流、验证工具）与
本目录各 SKILL.md / PROFILE。
