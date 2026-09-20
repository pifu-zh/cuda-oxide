# vendor/cuda-bindings — [PORT gfx1030 Stage2] HIP 后端替身

crates.io `cuda-bindings` 0.3.1 的 **HIP 实现**：同包名同版本，供 gfx1030 的
example 工作区用 `[patch.crates-io]` 整体换入。上游真身（bindgen 对 cuda.h、
dlopen libcuda）在本机不可用（无 NVIDIA 驱动）；本 crate 把同一批 `cu*`
入口翻译到 `libamdhip64`（运行期 dlopen，**构建期零 SDK 依赖**——不再需要
CUDA_TOOLKIT_PATH/bindgen）。

## 设计契约

- **入口面**：`cuda-core`/`cuda-host`/`cuda-macros` 引用到的全部 `cu*` 平面
  函数、句柄类型、launch 结构体、扁平化枚举常量（`<Type>_enum_<NAME>` /
  `cudaError_enum_<NAME>`），签名保持 CUDA 形状 → 上层 crate 零改动编译。
- **dlopen 模式**：仿上游 `dyn_load`——`OnceLock` 缓存函数指针表，逐符号
  `Result`，老 ROCm 缺符号降级为 `CUDA_ERROR_NOT_SUPPORTED` 而非整库失败。
  库名候选：`CUDA_OXIDE_HIP_LIBRARY` 覆盖 → `libamdhip64.so.1/.so/.7/.6/.5`。
- **错误码**：HIP 对常见码镜像 CUDA 驱动枚举（success=0 等），直接透传。
- **显式不支持**：HIP 无对应物者（multicast 族、cluster launch 属性、
  `cuCtxSetFlags`/`cuCtxSetLimit`）返回 `CUDA_ERROR_NOT_SUPPORTED`——
  响亮 stub 优于静默错误（aiter 教训）。
- **属性映射**：`cuDeviceGetAttribute` 的 CUDA→HIP 枚举值映射经 gfx1030
  实机验证（CU=40、CC=10.3、L2=4MiB 等）。
- **finalize 竞态**：`cuModuleLoadData` 成功后 host 侧 `hipDeviceSynchronize`
  排空 ROCm 7.x 的异步 finalize（DeepGEMM 实证；每模块一次、微秒级）。

## 换入方式（gfx1030 example）

```toml
[patch.crates-io]
cuda-bindings = { path = "../../../host-hip/cuda-bindings" }
```

上游 CUDA 路径不受影响：未打 patch 的构建仍用 crates.io 真身。

## 脆弱点（升级注意）

1. `cuda-core` 升版 → 新引用的 `cu*`/类型/常量需在此补齐（缺者编译期
   直接报错，不会静默）。
2. HIP 枚举数值映射以 ROCm 7.14 头 + 实机探针为准；ROCm 大版本升级需重验
   `cuDeviceGetAttribute` 映射表。
3. 结构体布局：`CUmemLocation_st`/`CUmemAccessDesc_st` 与 HIP 字节同构
   （编译期 assert），`CUmemAllocationProp_st`/`CUmemPoolProps` 走
   handler 内翻译。
