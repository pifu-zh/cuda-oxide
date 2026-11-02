# AMD RDNA2 / gfx1030 Architecture Profile

> 证据标记贯穿全文：[e2e]=真机执行+独立正确性校验；[probe]=探针/负探针；
> [measured]=真机测量（含协议）；[inferred]=推理未实测；[hist]=历史记录未复验。
> 平台语境默认 ROCm 7.2.4（Linux，RX 6950 XT）；换软件栈时重新确认本档与
> 平台无关的硬件结论仍然成立。

## 身份与拓扑 [probe]

- 代号 `gfx1030`（Navi 21 / sienna_cichlid），RDNA2，PCI 1002:73a5。
- 40 CU，wave32（`__oclc_wavefrontsize64` 必须置 1 才走 wave64 语义），
  工作组上限 1024，LDS 64 KiB/CU，128 MiB Infinity Cache（L2 级）+ GDDR6。
- 显存：256-bit / 18 Gbps（mclk P0 1124 MHz × 16T/pin，sysfs 实测钉死）
  → **理论带宽 575.5 GB/s**。
- 计算理论：40 CU × 64 lane × 2 FMA/clk × 2 flop × 2.72 GHz（sclk 钉死档）
  = **27.85 TFLOPS f32 双发射上界**（旧口径 21.5 按 2100 MHz 误标定，已修正）。

## 硬件能力矩阵

### 算术/数据类型

| 能力 | 状态 | 证据 |
|---|---|---|
| f32 FMA 双发射 | NATIVE | [measured] 25.7 TFLOPS = 92.3% 上界 |
| f16 / bf16 标量运算 | NATIVE（经 f32 展宽路径） | [e2e] packed-math 7/7 族 |
| bf16x2 打包算术 (v_pk_bf16) | **无硬件**（RDNA3 才有） | [probe] llc "instruction not supported" |
| f32x2 打包 | **无硬件**（RDNA3） | [probe] 同上 |
| dp4a (v_dot4) | NATIVE | [e2e] dotprod 四符号组合 |
| dp2a | 无单指令 | INSTRUCTION COMPOSITION：extract+mad 展开 [e2e] |
| fp8/fp4 硬件转换 | **无**；软件位操作解码可行 | [probe] + DeepGEMM 路线 [hist] |
| MFMA 矩阵核 / WMMA | **无**（缺 mai-insts / wmma-256b-insts） | [probe] 编译探针 |

### 线程/同步原语

| NV 概念 | gfx1030 替代 | 状态 |
|---|---|---|
| barrier0 | s.barrier | NATIVE [e2e] |
| mbarrier 相位/计数 | 单字 i64 状态机（软件仿真）；expect_tx 字节级不支持 | [e2e] 三 kernel 断言 |
| numbered barrier | LDS 计数器+世代自旋 | [e2e] 128 线程含旁观 warp |
| warp shuffle（全形态） | 分段 ds_bpermute（NV lane 号×4→字节偏移；wave64 两段） | [e2e] 全形态 |
| redux.sync.add | 5 轮 xor-butterfly | [e2e] 496 全 warp 和 |
| laneid | mbcnt.lo+hi 两段 | [e2e] |
| thread-block cluster / multicast | **无硬件** | HARDWARE_LIMITED |
| CDP 设备端动态并行 | **无**；host 二次 launch 替代 | HARDWARE_LIMITED |
| 非 move 闭包引用捕获 | 依赖 HMM；本卡无 → move 捕获等价 | NOT FEASIBLE（HMM 语义）[e2e 十二形状] |

### 内存/搬运

| 能力 | 状态 | 证据 |
|---|---|---|
| 直通 LDS 宽度 | 仅 b32（dwordx2/x4 汇编器拒） | [probe] |
| cp.async 标量异步 | 软件流水（s_clause 批量+vmcnt 部分排空+寄存器中转） | [measured] 365 GB/s = plain 78.4%（**良好档**） |
| TMA / bulk tensor | 无 TDM（gfx1250/RDNA4 才有）；同上软替代 | 同上 |
| TensorMap 硬件 swizzle | 软件.padding 先例 272/144B 行距 | [e2e] aiter mha 实证 |
| shared memory 动态 | 支持（启动配置给 bytes；.DynamicSharedArray 需足额） | [e2e] |

### 设备侧调试/输出

| 能力 | 状态 | 证据 |
|---|---|---|
| printf（hostcall） | FULL：8 测试族 + >7 参数链式真卡零乱码 | [e2e] |
| panic 消息 | **IMPLEMENTABLE**：rustc opt 内联 panic 体；消息文本可提取（StrConstScan）；
| | 最后一厘米被 MIR 验证器四道墙挡（详见上游研究档案 2026-11-02 节） | [probe] |
| DWARF 调试信息 | llc 路径已证（7 .debug_* 节）；前端 !dbg 管线未实现 | [probe] IMPLEMENTABLE |

## 性能实测（ROCm 7.2.4 容器，hipEvent 计时；协议见 scripts/perf）

| 项 | 实测 | 相对理论 |
|---|---|---|
| 带宽 triad 2R:1W（64M f32，规范协议） | 515.1 / 520.6 / 547.3 GB/s（三会话） | **90.0-95.1%** |
| 纯读流 2N | 565.7 GB/s | 98.3%（读侧天花板） |
| FFMA f32 链 | 25.7 TFLOPS | 92.3% of 27.85 |
| H2D / D2H pageable | 13.8 / 14.0 GB/s | — |
| H2D / D2H pinned | 27.6 / 28.8 GB/s | 2.0-2.1× pageable |
| 8×32MiB 重叠流水 | 217.6 → 96.6 ms | 2.25× |
| 空 kernel 往返 | min 10.97 µs（四段分解自洽） | SPEC_ABSENT（无官方分母） |
| 软 TMA 流水 | 365 GB/s | 78.4% of plain（良好档，收线） |

带宽数字归因（计数器实测）：读放大 +2.4% / 写放大 +2.1%（线上 786 MB vs 理想
768 MB）；L2 命中 25%（HIT+MISS 精确=512MB/64B）；waves 精确=64M/32；
triad 与纯读之差 = 2:1 混写总线换向（硬件属性，无 kernel 端浪费）。

**测量伪影记录**（防误判）：
- 工作集 < 128 MiB Infinity Cache 时带宽虚高（"write-only 125%"、"D2D 820"均此）。
- posted-write 完成语义：kernel/拷贝完成 ≠ DRAM 落盘（write-only ~800 GB/s 假象）。
- DVFS：mclk/sclk 冷态偏低；暖态协议（长预热+丢弃计时块）后稳定 90%+。
- float4 在 RDNA2 triad 上**反而比 scalar 慢**（87 vs 93%+，四会话一致）；
  ILP4/stride 形态更差——流式基准以 scalar 为规范形态。
- 数据内容（denormal 位型 vs normal float）对带宽**无影响**（三相位诊断证伪）。

## 替代方案目录（摘要；全文见 case-studies/rocm-rust-gfx1030/archives）

- 线程原语族全表（shuffle/redux/mbarrier/编号 barrier）→ 软替代全 [e2e]。
- 数学：`__nv_*` → `__ocml_*` 符号映射 + oclc 配置链接，≤2ULP [e2e]。
- packed-math PTX asm → 合法 IR 展开 7 族（Python+Rust 双实现 7/7）。
- nvls (multimem SHARP)、mma fp8：HARDWARE_LIMITED，保留 fail-loud。

## 版本/环境注意事项

- `__oclc_wavefrontsize64` 必须 =1（gfx1030 默认 wave64；设 0 → hostcall 乱码）。
- oclc 配置三件套（isa_version_1030 / abi_version_600 / finite_only_off）需链接。
- comgr 必须带版本化 soname（`.so.3`）；rocprofiler-register 是 profiler 链前提。
- 跨栈重建（conda/官方 deb/容器混装）已验证 perf 基线零漂移；容器优先工作流
  见 ROCm skill 第 5 节。
