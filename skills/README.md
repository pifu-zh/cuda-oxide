# GPU Adaptation Skills（GPU 适配技能体系）

本目录沉淀本项目（ROCm Rust on RX 6950 XT / gfx1030）适配过程中验证过的
可复用知识，按四层架构组织。**换什么才失效，就放哪层**：

| 层 | 变化轴 | 内容 |
|---|---|---|
| 方法论 | 换平台/架构/语言**都不失效** | [`gpu-adaptation-methodology/SKILL.md`](gpu-adaptation-methodology/SKILL.md) |
| 平台（ROCm） | 换**软件栈**才失效 | [`rocm-adaptation/SKILL.md`](rocm-adaptation/SKILL.md) |
| 架构 Profile（RDNA2/gfx1030） | 换**硬件架构**才失效 | [`gfx1030-rdna2/PROFILE.md`](gfx1030-rdna2/PROFILE.md) |
| 工具链集成（Rust GPU） | 换**语言/工具链**才失效 | [`rust-gpu-toolchain/SKILL.md`](rust-gpu-toolchain/SKILL.md) |

案例记录：[`case-studies/`](case-studies/rocm-rust-gfx1030/README.md)（时间线、
根因表、失败实验；原始研究档案在仓库根 `docs/research/`）。

## 面对新适配任务

1. 识别三元组：硬件架构 / 软件平台 / 语言工具链。
2. 读方法论 → 平台 skill → 架构 Profile → 工具链 skill（缺哪层就用方法论
   自行调查，不停工）。
3. 新增知识按上表回写到正确层级；推翻旧结论时保留原记录并标注 superseded。

## 证据纪律

每条知识带标记：`[e2e]` 真机执行+独立校验 · `[probe]` 探针/负探针 ·
`[measured]` 真机测量（含协议）· `[inferred]` 推理 · `[hist]` 旧记录未复验。
定义与红线见方法论 SKILL.md 第 0 节。

## 安装（ZCode / Codex）

- **ZCode**：`ln -s "$(pwd)/skills/<name>" ~/.zcode/skills/<name>` 即可被
  agent 平台发现加载（SKILL.md 带 YAML frontmatter）。
- **Codex**：通过仓库根 `AGENTS.md` 的 "GPU Adaptation Skills" 一节引用——
  Codex 会话自动读取 AGENTS.md，按需读入对应 SKILL.md。
