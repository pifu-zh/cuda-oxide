# 2026-11-11 延迟原语定档 + 三小项收口（gpu_assert 带消息 / mem_advise / fp8x4）

## 延迟原语性能定档（warp_ops_bench）

协议 = fp8_bench 同款 triad（2s wall 预热、事件计时 3×10 best-block），
256 线程单 CTA、64 ops/launch，guard = acc==OPS（与值链分离防 DCE——
shfl/redux 的 v 随迭代变化，值链另出 vout 槽）。

| 操作 | ns/op | vs 同卡 s_barrier（59.8ns） |
| --- | --- | --- |
| native s_barrier（参照） | 59.8 | 1.0× |
| 编号 barrier（软件，LDS 原子计数+世代自旋） | 527.5 | 8.8× |
| mbarrier（软件单字状态机） | 462.9 | 7.8× |
| shfl bfly（软件，ds_bpermute+段算术） | 69.1 | 1.2× |
| redux.add（软件，5 轮蝶形） | 495.6 | 8.3× |

**定档（披露口径）**：无 NVIDIA 卡可做同厂商对照，跨厂商参照 = NVIDIA 公
开典型周期（barrier 16-32 cycles、shfl 1-2 cycles，~2.4GHz 换算）——软件
实现为数十至千余周期，**延迟口径低于 60% 线**；按 §7.6 功能/性能分报：功
能替代已独立证明（四例真卡全绿），性能如实记低于线。与带宽型替代（soft
TMA 91.4% 完美档）口径分离——延迟原语的本性是 RDNA2 无对应指令，软件即唯
一路径，价值在解锁语义而非等速。shfl 69.1ns ≈ 同卡硬件 CTA barrier 成本，
四项中最接近可用（单轮 bpermute + 少量整数前处理）；barrier 族 ~8× 主因 =
LDS 原子计数器的 256 线程争用与世代自旋。

## gpu_assert! 带消息（源级已支持、真卡此前 fail-loud → SUPPORTED）

MIR 形态 = `call void @__assertfail(msg, file, line, func, 1)` +
`unreachable`。__assertfail 是 NV 专有符号，amdgcn 路径未定义——带消息形
态此前在真卡上是链接死路。prep pass 17c `rewrite_assertfail_to_ockl`：

1. 解析三指针实参（内联 addrspacecast 直派 + `%name` 溯源链兜底）；
2. 合成 `gpu_assert: <msg> (<file>:<line> in <func>)\n\0` as1 全局（模块级
   统一插到首个 define 前——全局不可在函数体内定义）；
3. 调用点改写为 vprintf(null-args) ockl 链（is_last=1 单串形态）；
4. `unreachable` → `ret void`（kernel）或 `ret <ty> undef`（非 void——失
   败路径返回值本不可用）；干净退出保排水。

真卡：`gpu_assert: value exceeded limit (src/main.rs:24 in
gpu_assert_probe::kernels)` 精确渲染，launch/sync Ok，通过路径零输出。

**开发中踩的坑（三次误配全修）**：①`@__assertfail(` 的 off-by-one——按
`__assertfail(`.len() 跳过把 call 的开括号留在 inner 里（args[0] 以 `(ptr`
开头）；②ret 类型追踪按 define 后首词——`amdgpu_kernel` 前缀吃掉后取到
`amdgpu_kernel` 本身当类型 → `ret amdgpu_kernel undef`；③合成全局写在函
数体内 → llvm-link "expected instruction opcode"。

## mem_advise / mem_prefetch_async（shim 本有、公开面缺失 → SUPPORTED）

shim 的 `cuMemAdvise_v2`（→hipMemAdvise）与 dyn-load 的
`cuMemPrefetchAsync_v2` 本已存在——缺的是 cuda-core `api.rs` 的公开包装。
补 `pub unsafe fn mem_advise / mem_prefetch_async`（doc 说明 HIP 栈上
Err = 平台不支持，按 CUDA 可移植性指引报告而非吞掉）。managed_advise 例
真卡：SET_READ_MOSTLY + SET_ACCESSED_BY 双 advice OK、dev-0 prefetch OK、
advise/prefetch 前后 kernel 往返正确性不变。

## fp8x4（x2 组合收口）

`e4m3x4/e5m2x4_to_f32x4` + `f32x4_to_e4m3x4/e5m2x4_rne` = 两个 x2 半拼接
（lane0 低字节、w 高字节）。x2 层已穷举证明，x4 只验证拼接（字节序/移位）：
抽样 4096 stride + 全字节低半块（5120 pattern × 4 lane）decode、2540/2480
quad encode，全过。fp8_decode 总检查 **450,852 零失败**。

## 测试统计

| 项 | 结果 |
| --- | --- |
| warp_ops_bench 真卡 | guard PASS，五组数字（两次运行稳定） |
| gpu_assert_probe 真卡 | 消息精确渲染 + 干净退出 PASS |
| managed_advise 真卡 | PASS |
| fp8_decode（含 x4） | 450,852 项零失败 |
| panic_message / vecadd 回归 | PASS |
| codegen / cuda-macros / cuda-device 测试 | 1321 / 147 / 127 全过 |
| pytest（NVVM 路径，无 amdgcn env） | 12 passed |

## 遗留（已披露）

- 性能档量化只覆盖四项延迟原语；expect_tx 字节级（mbarrier 行缺口）未动。
- 管线小 wart：async crate 的 E0133 doctest 已复核非失败（前轮档案）；
  cuda-device/heap.rs 的 unsafe_op_in_unsafe_fn 警告为基线噪音未清。
