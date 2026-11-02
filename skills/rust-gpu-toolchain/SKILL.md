---
name: rust-gpu-toolchain
description: Rust GPU 编程与工具链集成的跨平台知识——自定义 codegen 后端架构、MIR 到目标 IR 的管线、GPU ABI 与 kernel 接口、host/device 编译分离、宏驱动的 kernel DSL、验证器契约、测试与正确性策略。Use when 为 Rust 构建/调试 GPU 后端集成、诊断设备代码编译问题（ICE、验证器拒绝、ABI 不匹配）、设计 kernel 宏或 host/device API、或跨 GPU 平台评估 Rust GPU 方案。不绑定 ROCm——CUDA（nvptx 后端）、其他 codegen 后端同样适用；平台特定细节见各 Platform Skill。
---

# Rust GPU Toolchain Integration

前提：已读 `gpu-adaptation-methodology`（尤其第 6 节"中间表示契约"）。

## 1. 可行的集成形态（按侵入度排序）

| 形态 | 机制 | 适用 |
|---|---|---|
| 现成方言 crate（如 rust-gpu / melodium 类） | 上游维护后端 | 最省力，受上游架构门控 |
| **自定义 codegen 后端**（dylib，`-Zcodegen-backend`） | rustc 稳定驱动 + `rustc_private`/SMIR 管线 | 需要全管线控制（本项目形态） |
| 外部编译器双缆（hipcc/cllc 产对象，rustc 只编 host） | FFI 桥 | 语言体验弱，但隔离性好 |

自定义后端关键事实（stable 工具链实测）：
- stable rustc + `RUSTC_BOOTSTRAP=1` 可用 `-Zcodegen-backend` 与
  `-Zalways-encode-mir`/`-Zmir-enable-passes`（SMIR/rustc_public 管线）。
- 后端 dylib 的链接器选择影响进程内 panic 展开：非标准链接（静态 libunwind）
  会让 rustc 进程内 SIGSEGV——用系统 gcc 标准链接。
- **rustc_public 的查询有上下文契约**：`rustc_internal::stable/run` 不得嵌套；
  部分查询键在特定上下文不受支持（实测 `module_children` 对本地 crate 根
  ICE，改用 `mir_keys` 扫描）。

## 2. 双域编译模型

Rust GPU 的核心结构是**同一份源码、两个编译域**：

```
#[cuda_module] mod kernels {      ← proc-macro 标记编译单元
    #[kernel] fn k(a: &[f32]) {…} ← 设备域（MIR → 目标 IR → code object）
}
fn main() { … kernels::load(&ctx)… } ← 宿主域（正常 rustc x86_64）
```

- proc-macro 生成的宿主侧 API（load/launch）通过**锚点符号**（anchor symbol）
  从宿主二进制定位内嵌的设备工件；工件有三形态（文件 / 内嵌 bundle /
  未链接对象），装载行为不同，诊断先确认形态。
- 设备域收敛原则：入口 kernel → 可达闭包收集（worklist）→ 逐个翻译。
  收集/发射/翻译三个决策必须**共享同一谓词**——三处各自判断必然漂移
  （历史事故：发射了没人调的符号、或漏发被调符号）。
- 跨 crate 收集：普通 `pub` 函数正常收集可达闭包即可；**no_mangle 形式的
  合成符号**（如运行时 shim）不在 mir_keys——用 CGU 符号查找 + 本地
  `mir_keys` 扫描（不要用 `module_children` 查本地 crate 根，会 ICE）。

## 3. GPU ABI 与 kernel 接口（MIR 级事实）

- kernel 参数按 kernarg 槽传递：**每个签名形参都要有槽**；打包结构体的
  padding 不是槽。槽数不足 = 崩在 Runtime 内部（元数据遍历越界）。
- `blockDim`/`blockIdx` 类隐式量在 AMDGCN 无 sreg：合成"尾随 kernarg 形参"
  （宿主恒推三值）是可行形态。
- 设备 panic：无 panic runtime。rustc 在 opt 下把 panic 体**内联**成
  `asm!("trap;")` 终结块——消息文本仍在块内（可用 MIR 访问器提取），
  但把消息变成设备值需要常量字符串 lowering 基础设施（见第 5 节）。
- `Barrier`/原子/`isspacep` 类语义按目标 ISA 映射表替换（平台 skill 提供表）。

## 4. 中间表示验证器契约（MIR→LLVM-IR 管线实测；模式通用）

有验证器的 IR 对每个构造有精确契约，手工合成新构造时会连续撞墙。
本项目实测的四堵墙（同样的墙在其他 IR 上以别的名字存在）：

1. 全局分配 op 的结果指针地址空间受限（只认存储地址空间，不认泛型）。
2. 跨点子指针 cast 即使带 ABI 边界授权也被拒（点子必须精确相同）。
3. "声明式指针类型确立"助手只做同点子 retype（地址空间/kind）。
4. 引用类 op 消费**点子类型的值**而非指针（不能拿指针当被引用值）。

工作法则：
- 撞两次墙后停止逐个绕——去找"上游合法产生这种构造"的机制，照机制造。
- 把每堵墙的精确报错原文记录进档案：它们定义了缺失基础设施（如常量
  字符串的设备端按值物化）的准确形状与工作量。
- 合成调用链的插入顺序 = 非终结 op（全局/转换/调用）→ 终结 op（goto/unreachable）
  最后；任何终结子之后不能再有 op。

## 5. Host/Device 协作面

- **printf 走宿主服务协议**（设备端 marshal、宿主格式化）：槽数上限、
  handle 接力、is_last 末轮——细节见平台 skill。它把"设备格式化基础设施
  缺失"变成"设备 marshal + 宿主 printf"，是语言集成的高杠杆替代。
- **panic 无 runtime**：`trap` 是唯一出口；消息上报是待建基础设施。
- Pinned/managed/mempool 的宿主面 API 对带宽影响巨大（2×+），
  语言集成的价值排序里传输路径高于算子微优化。
- 设备侧 spin/同步必须 volatile 链，否则被折叠。

## 6. 构建系统

- 例 crate 常是独立 workspace（空 `[workspace]` 标记），路径依赖对齐
  仓库层级——挂载/搬移时必须覆盖被引用的兄弟目录。
- 后端 dylib 按 `--target` 目录存放，驱动按固定路径加载：**改了后端
  行为却没生效，第一怀疑 dylib 没按正确 target 重编**。
- env 契约（本项目实测）：`RUSTC_BOOTSTRAP=1`、`CUDA_OXIDE_TARGET=<isa>`
  （缺失会走错产物路由——设备 loader 报 INVALID_IMAGE）、`LIBCLANG_PATH`
  与 bindgen resource 头、`CUDA_TOOLKIT_PATH`（部分构建管线硬校验，
  conda-forge CUDA 组件拼装即可满足）。
- 批量回归：设备侧变体抽取（device-only crate 重写）+ 双维度验证
  （IR 管线维度 / 集成构建维度），产物目录放持久磁盘（tmpfs 会 EDQUOT
  连杀编译进程）。

## 7. 测试与正确性策略

- 每能力一例，host 侧独立参考断言；能力清单进机器可读矩阵
  （分类：SUPPORTED/IMPLEMENTABLE/BLOCKED/HARDWARE_LIMITED/…）。
- trybuild 类快照测试可锁"诊断文本"——改诊断措辞会红，是防回归资产。
- 扫描上游示例全集（数百例）做能力覆盖盘点：按 fail 类聚根因
  （asm 约束缺口 / SM90+ 特性 / 抽取器限制），每类给处置，不当黑盒忽略。
