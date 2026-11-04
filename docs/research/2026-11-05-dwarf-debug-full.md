# 2026-11-05 设备端 DWARF（CUDA_OXIDE_DEBUG=full）全链 SUPPORTED 记录

## 结论

能力矩阵「设备端 DWARF 调试信息」IMPLEMENTABLE → **SUPPORTED**：`CUDA_OXIDE_DEBUG=full`
（rustc debuginfo 级别 + env override）下 vecadd 端到端构建、真卡运行 1024/1024
全对，hsaco 产 8 个 `.debug_*` 节，llvm-dwarfdump 解析出真实 DIE（CU producer
"cuda-oxide"、DW_LANG_Rust、9 个 DW_TAG_subprogram 带 decl_file/decl_line 指向
真实源码、formal_parameter、inlined_subroutine + ranges、行表 105 行）。

## 前提修正（旧行判定过时）

旧矩阵行写「工具链已证，前端未实现——剩余工作：实现 debuginfo API（MIR DI 建
图→IR dbg metadata）」。实测：**该映射早已存在**——首轮 debug 构建的
`.preopt.ll` 即含 211 个 `!dbg`（DebugKind Off/LineTables/Full 基础设施在
llvm-export/export/config.rs，rustc debuginfo 级别驱动）。真实缺口全部在
amdgcn prep pass 对 debug 形态 IR 的拒绝，属「管线中间层」而非「前端未实现」。

## 三层根因（全在 amdgcn prep pass，Rust+Python 双镜像）

1. **`rewrite_calls` 拒收 `#0, !dbg !N` 逗号粘连尾缀。** `split_whitespace`
   切出 `#0,`（逗号粘连在 token 上），`token[1..].all(is_ascii_digit)` 对逗号
   失败 → 整行跳过 → sreg 读不改写 → `%ntid_` 引用不出现。非 debug 构建只有
   裸 `#0`（无逗号）故从未暴露。修复：token 先 `trim_matches(',')` 并忽略纯
   逗号 token。

2. **define 追加 `between_ok` 字符集不含 `!`/`.`。** kernel 与设备函数 define
   尾巴 `#0 !dbg !N` 落在参数表闭括号与 `{` 之间 → 字符集拒绝 → 形参插入被
   静默跳过。修复：字符集加 `!` `.`（append_ntid_kernarg_params 与
   append_ntid_transitive 两处同改）。

3. **Step-5b/5c 调用点追加只匹配行首 call，且 format 串本身错位。** 结果绑定
   调用 `%vN = call i64 @helper(...)`（本管线唯一存在的调用形态）不匹配
   `starts_with("call ")`；且追加器的 format 串 `{}{}{}, {}{}` 产出
   `head` + NTIDS（**缺逗号**）+ `, ` + `)`（**悬垂逗号**）。此二者此前是
   「死但错」代码——非 debug 构建靠 opt -O2 把 alwaysinline helper 全部内联、
   调用点消失，追加器从未命中。Full-debug 跳 opt（同 nvcc -G）后调用点活到
   llc：define 已加宽而调用点缺参 → llc 把缺参映射为 poison → 垃圾寄存器选择
   -1 sentinel 路径 → kernel 一个不写、输出全零（vecadd debug 首跑 1023/1024
   错误实证；与 Step-5b 当年 cp_async_small 故障同类）。修复：is_call 放宽到
   ` = call `/` = tail call ` 绑定形式；插入改为 head + `", {NTIDS}"`（空参表
   不带逗号）+ 尾部（`)` 由保留的尾巴携带）。

教训：**放宽「永不命中的死代码」匹配前，先验证死代码路径自身正确**——激活一
个从未跑过的分支等于首次运行新代码。

4. **（激活死代码后的连带回归，批 91/105 实证后修复）调用图单遍扫描丢前向
   引用边。** `append_ntid_transitive` 的图扫描在遇到 call 行时检查
   `fn_info.contains_key(&callee)`，但 callee 的 define 可以在文件**后部**——
   扫描到调用时还没注册 → caller→后定义 callee 的边丢失 → 不动点漏掉中间
   函数 → 5c 调用点补了实参而该函数 define 没加形参 → llc
   `use of undefined value '%ntid_x'`（slice_from_raw_parts_cast/
   struct_constant_provenance/switchint_128bit 等六例；get_mut_indexed 调
   文件后部的 from_scope 形状）。此前不炸的原因：需要的边两端恰好都是种子
   （kernel / 体引用 ntid），vecadd 即此幸运形状。修复：两遍扫描（第一遍注
   册全部 define，第二遍收边），Rust+Python 双镜像；新增前向引用单测
   `forward_referenced_helper_still_threads_ntid`。旧 Step-5c 的 6 例恢复同
   样依赖该图——当时靠 define 顺序运气成立，两遍化后不再依赖。

## 顺手修正（同族）

- alloca 元数据剥离循环原为覆盖式（`metadata = format!(", {}", tail)`）——一条
  alloca 带两个元数据附件（`!dbg` + rustc `!annotation`）时先剥的被顶掉，
  fail-silent。改为保序累积 `metadata = format!(", {tail}{metadata}")`（Python
  镜像本就 `insert(0)` + `join(", ")` 正确）。多附件形态入单测。
- 新增 `debug_shaped_module_is_rewritten_with_dbg_tails_preserved` 单测：debug
  形态 IR（define `#0 !dbg !N` 尾、调用 `#0, !dbg !N` 尾、结果绑定调用、多附
  件 alloca）全链断言 + 幂等断言。1320 tests 全绿（含 toolkit env 的 libdevice
  组；漏 CUDA_TOOLKIT_PATH 时该组 5 例环境性失败，与代码无关）。

## 验证链

| 步骤 | 结果 |
| --- | --- |
| `CUDA_OXIDE_DEBUG=full cargo-oxide build vecadd` | llc 通过（此前 Cannot select: sreg.ntid.x） |
| 卡上运行（debug 插桩 code object） | 1024/1024 全对 |
| 同 dylib 非 debug vecadd | 1024/1024 全对（无回归） |
| hsaco 节表 | 8 个 `.debug_*`（probe 预期 7 节之上多 pubnames/pubtypes） |
| llvm-dwarfdump | CU + 9 subprogram + 形参/内联 DIE + 行表 105 行 |
| `cargo test -p cuda-oxide-codegen` | 1321 passed |
| pytest（test_gfx1030 + debug-tests） | 12 passed / 24 passed（cuda-gdb 组跳过=gfx 预期） |
| 批回归（105 例） | 修前 91/105（六例 `undefined %ntid_x` 回归）→ 两遍扫描修复后 **97/105 = 基线**（4 已知 flaky：dotprod/atomics/redux_sum/warp_reduce；3 排除），且调用点追加器为激活状态经全批验证 |

## 环境教训（dylib 新鲜度误诊一轮）

- cargo-oxide 后端 dylib 的正确构建方式 = 容器既定 INNER_ENV（`/host-cargo/bin`
  + `CARGO_HOME=/host-cargo` + `RUSTUP_HOME=/host-rustup` + RUSTC_BOOTSTRAP=1，
  rustc 1.99.1 SIMT pin）。容器内并存 `/home/ubuntu/.cargo`（另一套 cargo）——
  用它构建会解析到不同工具链状态，且 dylib 必须与加载它的 rustc ABI 一致。
- 「源码含改动 + 构建显示 Finished」不等于「dylib 含改动」：指纹/路径歧义时用
  `cargo build` 的 Compiling 行 + 符号/字符串取证，并以显式
  `CUDA_OXIDE_BACKEND=<path>` 隔离发现顺序变量。
- 批 runner scratch（`/home/zhuo/tmp-prof`，容器可写层）每轮累积 ~1GB/例：
  本轮 93GB 打满宿主根盘（Errno 28 级联假 build-fail）。清理后重跑。**批跑前
  先查 df**。

## 已知留档 wart（未动，保持已验证字节输出）

- `rewrite_calls` 重写行双缩进（`indent` + `line[..eq]` 各带一份前导空白）：
  llc 对空白不敏感，97/105 批一直带着它通过；修复属化妆品改动，会扰动全部产
  物字节，不在本轮范围。
- 容器 rustfmt（1.10.0-stable, 1.99.1 pin）与已提交代码存在 ~200 处风格漂移
  （`cargo oxide fmt --check` 报 collector.rs/printf 例/sreg.rs 等未触碰文件；
  `cargo clippy` 亦报既有 parens/unused 警告）。本轮只保证**自己改动区域**
  rustfmt 清洁；基线漂移留给 CI 仲裁，不做全局重排。
