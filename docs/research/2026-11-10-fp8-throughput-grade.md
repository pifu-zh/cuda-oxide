# 2026-11-10 fp8/fp4 软件转换吞吐定档 + 仓库卫生

## 吞吐定档（fp8_bench 例，gfx-m1 容器）

协议 = §8 triad 规范的转换版：2s wall 预热稳时钟 → 每个计时块前 H2D 重灌
输入（fresh `DeviceBuffer::from_host`）→ 事件计时 10-launch 块 → 3 块取
best。工作集 128Mi 元素/方向（f32 侧 512MiB，输出流每 rep 数倍翻越
128MiB L2，无 cache 常驻假象）。名义流量 5 B/元素（1 B fp8 进 + 4 B f32
出，或反向；packed 同为 5 B/元素）。

| kernel | GB/s | grade（vs copy 516.9） |
| --- | --- | --- |
| copy_f32（参照） | 516.9 | 100% |
| decode_e4m3 | 472.7 | **91.4%** |
| decode_e5m2 | 478.5 | **92.6%** |
| encode_e4m3 | 489.8 | **94.7%** |
| encode_e5m2 | 505.4 | **97.8%** |
| decode_e4m3x2 | 423.4 | 81.9% |
| decode_e5m2x2 | 421.2 | 81.5% |
| encode_e4m3x2 | 516.8 | **100.0%** |
| encode_e5m2x2 | 514.5 | **99.5%** |

定档（60/70/80/90 尺度，参照 = 被替代厂商路径）：有 fp8 转换单元的硬件上
转换是 memory-bound 的免费操作，故 copy 流即参照——**标量族 91-98% 与
packed encode ~100% 达 90% 完美替代档**；packed decode 81-82% = 80% 优秀
档（双 DisjointSlice 输出形态的微架构代价，非功能缺口；x4 收口时评估单
输出形态）。

测量学注记：encode 族读侧 512MiB ≫ L2 自净；decode 族读侧 128MiB 恰在
L2 边界，但写侧 512MiB 每 rep 强制翻越 L2 数次，输入常驻不可能成立——
H2D 重灌按协议保留为双保险。数值与历史带宽族一致（copy 516.9 vs triad
515-547、纯读 553 的同卡口径互证）。

## 仓库卫生（同轮）

- `git rm --cached` 7 个误提交的示例代码对象（f104ceb2c）。
- rustc-codegen-cuda/.gitignore 补 `*.hsaco`/`*.co`/`.oxide-artifacts/`
  （此前只忽略 *.ll，构建产物反复以 untracked 噪音出现）；按 250 个示例
  lockfile 的既有惯例补交 fp8_decode/panic_message/panic_format 三个
  Cargo.lock（56da3a991）。树内不再有 untracked 噪音。

## 遗留

- fp8x4（u32 四值）：x2 组合即得，收口时顺带评估 packed decode 单输出形态
  的 90% 线差距。
- 批清单未纳入 fp8_decode/fp8_bench/panic_format（独立验证惯例；纳入则批
  分母与基线需同步重跑）。
