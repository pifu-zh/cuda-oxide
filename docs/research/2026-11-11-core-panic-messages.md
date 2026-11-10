# 2026-11-11 core 机械 panic 消息渲染：Assert 形态拦截 + 失败块合成

## 结论

编译器自带的 panic（切片越界、整除零、取余零）在真卡上报出与 core::panicking
逐字一致的消息，wave 干净退出：oob_probe 例四条
`panic: index out of bounds: the len is 4 but the index is 4/5/6/7`（逐线
程）+ `attempt to divide by zero` + `attempt to calculate the remainder
with a divisor of zero`，launch/sync 全 Ok。能力矩阵新增 SUPPORTED 行。

## 关键形态发现：越界 panic 在 MIR 里不是 Call

先按 format-arg panic 的路径假设"panic 块调用 panic_bounds_check(index,
len)"去 collector 拦 Call——verbose 探针证实 **kernel 的 MIR 里根本不存在
该 Call**。越界检查在 rustc_middle/rustc_public 的 MIR 中是 **Assert
terminator**（`AssertKind::BoundsCheck{index,len}`/`DivisionByZero`/
`RemainderByZero`，结构化消息数据直接挂在 terminator 上），失败路径是
隐式的（MIR 只有成功 target）。这条形态差异让两层既有机制全部失明：collector
的 Call 扫描（字面量路径）与 kernel 宏的源级重写（源里没有 panic!）。

正确的拦截点 = **assert 翻译器**（translate_assert）：消息文本 + 参数形状
全部在 AssertKind 里，无需任何签名猜测。

## 实现

1. **cuda-device::debug** 三个 helper（消息文本逐字镜像 core::panicking
   1.99，rustc-src 组件里的 rustc_public/mir/body.rs 的 message() 映射为
   权威对照）：`__cuda_panic_bounds_check(index, len)`（渲染顺序 len 在前
   ——镜像 core 的消息文本而非调用顺序）、`__cuda_panic_div_by_zero()`、
   `__cuda_panic_rem_by_zero()`。内部 = gpu_printf 机制（fmt 常量
   `&[u8; N]` 定长数组 + repr(C) 参数包 → `__gpu_vprintf` → ockl 链）。
   **工艺坑**：fmt 不能用 `const FMT: &[u8]`（胖指针切片引用常量，常量分
   发器不支持），必须 `&[u8; N]` 定长数组引用（瘦指针，与宏路径同形）。
2. **collector**：check_panic_machinery 扫 Assert terminator；命中 +
   is_kernel + instance 返回 unit（`TypingEnv::fully_monomorphized()` 下
   fn_sig）→ 记入 per-function 资格集 + 按种类入队 helper。资格集经
   result → generate_device_code → run_pipeline → facts 线程落。
3. **translate_assert**：资格命中的三种 Assert 不再发 `MirAssertOp`+goto，
   改发 `MirCondBranchOp`（成功 → 原 target，失败 → **动态合成的失败块**
   `BasicBlock::new` + insert_after——与 switch 链的中间块同一 API）+ 失败
   块内 [操作数翻译 → emit_raw_call → MirReturnOp]。

## 调试工艺（两条硬坑，均为 opt 内联改名的连锁）

1. **emit 链断裂**：失败块的初版把操作数翻译（loads）与 helper 调用的插
   入点脱钩——loads 追加块尾、call 拿着过期的 prev 插到块首，return 落中
   间（pliron "terminator not last"）。**VERIFY DUMP**（临时全函数 disp
   转储）一眼定位 [call, return, load, load] 乱序 → prev 链贯穿修复。
2. **vprintf 重写器在 opt 内联后失配**（连带修复，影响面远超 panic）：
   helper 被 opt 内联进 kernel 后，名字段被改写——
   `%X.ac.i6 = addrspacecast ptr addrspace(5) %X.i5 to ptr`（内联第 6 份
   用第 5 份的 alloca 名，后缀数字错位）。vprintf→ockl 重写器的 args 指
   针解析只剥尾缀 `.ac`（`.ac.i6` 剥不动）+ 类型表无 cast 名 → 调用点保
   留裸 `@vprintf` → ockl 链接时 undefined value → 模块加载 209。
   修复 = 类型表并入"指向 as5 alloca 的 addrspacecast 别名"（不动点迭代）
   + `.ac` 段剥离兜底（先原名查表、再剥 `.ac` 查表）。**该修复使任何经
   opt 内联的 vprintf 调用点正确展开**——此前这类站点会以链接失败告终。
   诊断手段：python 复刻重写器门控跑真实 .ll 找 bail 点（比读代码快）。

## 测试统计

| 项 | 结果 |
| --- | --- |
| oob_probe 真卡（bounds×4 线程 + div + rem） | 6 条消息精确渲染 + 3 个干净退出 PASS |
| panic_message / panic_format / printf / fp8_decode / vecadd 回归 | 全 PASS |
| cuda-oxide-codegen 测试 | 1321 passed |
| cuda-macros 测试 | 147 passed |
| 批回归（105 例） | 见矩阵批量验证行 |

## 范围（文档化）

- unit-return kernel：wave 退出 = 裸 return。非 unit kernel、设备函数内
  panic、其余 AssertKind（溢出类——release 语义下不触发）保持 trap。
- 消息文本锚定 rustc 1.99 的 core::panicking；rustc 升级若改文案需同步
  （helper 与 core 的文本对照在 rustc_public body.rs 的 message() 映射）。
