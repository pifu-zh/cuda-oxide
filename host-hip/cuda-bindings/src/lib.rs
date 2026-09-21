//! [PORT gfx1030 Stage2] HIP-backed stand-in for the crates.io `cuda-bindings`
//! package.
//!
//! Contract: same package name + version (so `[patch.crates-io]` swaps it in
//! wholesale) and the same referenced surface of the upstream crate — the
//! `cu*` driver-API entry points, opaque handle types, launch structs, and
//! flattened enum constants that `cuda-core`/`cuda-host`/`cuda-macros` use —
//! but every driver call is translated to the HIP runtime
//! (`libamdhip64`, dlopen'd at first use, no build-time SDK).
//!
//! Design notes:
//! - Entry points keep their CUDA names/shapes so the crates above compile
//!   unmodified; the HIP ABI agrees at the C level (handles are pointers,
//!   codes are `int`, `cuLaunchKernel`'s per-parameter `kernelParams` array is
//!   exactly `hipModuleLaunchKernel`'s shape — Phase 1 verified on device).
//! - HIP error codes pass through numerically: HIP mirrors the CUDA driver
//!   enum for common codes (`hipSuccess` = 0 = `CUDA_SUCCESS`, invalid value
//!   = 1, out of memory = 2, ...), which is all `DriverError` needs.
//! - Where HIP has no equivalent (multicast objects, cluster attributes on
//!   RDNA2, `cuCtxSetFlags`), the entry point returns
//!   `CUDA_ERROR_NOT_SUPPORTED` explicitly instead of failing silently —
//!   aiter lesson: loud stubs over silent wrong answers.
//! - `cuModuleLoadData` runs a host-side device synchronize after a
//!   successful load: ROCm 7.x finalizes module code asynchronously and a
//!   launch racing that finalize can silently no-op (DeepGEMM port
//!   evidence; the tiny Stage-1 vecadd module happened not to hit it, the
//!   barrier costs microseconds and removes the race).

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]

#[allow(unused_imports)]
use std::ffi::c_void;
use std::os::raw::c_char;

pub mod hip;

pub use hip::DynLoadError;

use hip::{
    HipApi, HipCtx, HipEvent, HipFunction, HipMemAccessDesc, HipMemGenericAllocationHandle,
    HipMemLocation, HipMemPoolProps, HipMemoryPool, HipModule, HipStream, HipUuid,
};

// ---------------------------------------------------------------------------
// Loader (mirrors upstream dyn_load: one cached table, error introspection)
// ---------------------------------------------------------------------------

static HIP_API: std::sync::OnceLock<Result<HipApi, DynLoadError>> = std::sync::OnceLock::new();

fn init_hip() -> &'static Result<HipApi, DynLoadError> {
    HIP_API.get_or_init(|| unsafe { hip::load_api() })
}

fn hip() -> Result<&'static HipApi, CUresult> {
    match init_hip() {
        Ok(api) => Ok(api),
        // The driver-reported code for "library unavailable" upstream.
        Err(_) => Err(cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED),
    }
}

/// Returns the cached HIP load failure, if the runtime never loaded.
pub fn cuda_driver_load_error() -> Option<&'static DynLoadError> {
    match init_hip() {
        Err(error) => Some(error),
        Ok(_) => None,
    }
}

/// Whether the HIP runtime loaded successfully.
pub fn is_cuda_driver_available() -> bool {
    hip().is_ok()
}

macro_rules! hip {
    ($field:ident ( $($arg:expr),* $(,)? )) => {
        match hip() {
            Ok(api) => match api.$field {
                // HIP entry points return C `int` codes; our CUresult is u32.
                Ok(f) => unsafe { f($($arg),*) as u32 },
                // Symbol missing from the loaded runtime (older ROCm).
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_SUPPORTED,
            },
            Err(code) => code,
        }
    };
}

// ---------------------------------------------------------------------------
// Types (handles, result, structs)
// ---------------------------------------------------------------------------

/// Driver result codes are `u32` end to end (bindgen renders C enums as
/// unsigned); HIP mirrors the numbering. Handlers cast HIP's `int` codes.
pub type CUresult = u32;

pub type CUdevice = i32;
/// CUDA `CUdeviceptr` is an integer device address; `hipDeviceptr_t` is
/// `void*`. Handlers cast through `usize` at the boundary.
pub type CUdeviceptr = u64;

#[repr(C)]
pub struct CUctx_st {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct CUmodule_st {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct CUfunc_st {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct CUmemoryPool_st {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct CUgraph_st {
    _unused: [u8; 0],
}
// [PORT gfx1030 PhaseB.1] CUDA graph exec handle (opaque, HIP 1:1)
#[repr(C)]
pub struct CUgraphExec_st {
    _unused: [u8; 0],
}
#[repr(C)]
pub struct CUmemGenericAllocationHandle_st {
    _unused: [u8; 0],
}

pub type CUcontext = *mut CUctx_st;
pub type CUstream = *mut CUstream_st;
pub type CUmodule = *mut CUmodule_st;
pub type CUfunction = *mut CUfunc_st;
pub type CUevent = *mut CUevent_st;
pub type CUmemoryPool = *mut CUmemoryPool_st;
pub type CUgraph = *mut CUgraph_st;
// [PORT gfx1030 PhaseB.1] CUDA graph exec handle (opaque, HIP 1:1)
pub type CUgraphExec = *mut CUgraphExec_st;

// [PORT gfx1030 PhaseB.1] constants the cuda-async reactor references
// (values from cuda.h 13.x; HIP mirrors the pinned-memory flag numbering)
pub const CU_HOST_TASK_BLOCKING: u32 = 0x0;
pub const CU_HOST_TASK_SPINWAIT: u32 = 0x1;
pub const CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_NONE: u32 = 0;
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
pub type CUmemGenericAllocationHandle = *mut CUmemGenericAllocationHandle_st;

/// `CUuuid`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUuuid {
    pub bytes: [c_char; 16],
}

/// `CUmemLocation_st`: byte-identical to HIP's `hipMemLocation`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemLocation_st {
    pub type_: u32,
    pub id: i32,
}
pub type CUmemLocation = CUmemLocation_st;

/// `CUmemAllocationProp_st`. Field-for-field the CUDA shape; translated into
/// HIP's `hipMemAllocationProp` (different field order + trailing metadata)
/// inside the handlers that take it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAllocationProp_st {
    pub type_: u32,
    pub requestedHandleType: u32,
    pub location: CUmemLocation_st,
    pub alloc: CUmemAllocationProp_alloc,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAllocationProp_alloc {
    pub allocFlags: usize,
}

/// `CUmemAccessDesc_st`: byte-identical to HIP's `hipMemAccessDesc`
/// (asserted at compile time below).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAccessDesc_st {
    pub location: CUmemLocation_st,
    pub flags: u32,
}

/// `CUmemPoolProps` (CUDA shape; translated to `hipMemPoolProps` in-handler).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemPoolProps {
    pub allocType: u32,
    pub handleTypes: u32,
    pub location: CUmemLocation_st,
    pub usage: CUmemUsage_st,
    pub maxPoolSize: usize,
    pub reserved: [u8; 64],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemUsage_st {
    pub enabled: u8,
}

/// `CUmulticastObjectProp_st`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmulticastObjectProp_st {
    pub size: usize,
    pub numDevices: u32,
    pub numDevicesHandled: u32,
    pub appUuid: CUuuid,
    pub requestId: u64,
}

/// `CUlaunchAttribute_st`: upstream models it opaquely (CUDA 13.2+ layout
/// `{ id: u32 @0, pad, value: union @8 }`); consumers write it through raw
/// byte offsets, so a byte array of the documented 24-byte size is exact.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUlaunchAttribute_st {
    pub opaque: [u8; 24],
}

/// `CUlaunchConfig_st`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUlaunchConfig_st {
    pub gridDimX: u32,
    pub gridDimY: u32,
    pub gridDimZ: u32,
    pub blockDimX: u32,
    pub blockDimY: u32,
    pub blockDimZ: u32,
    pub sharedMemBytes: u32,
    pub hStream: CUstream,
    pub attrs: *mut CUlaunchAttribute_st,
    pub numAttrs: u32,
}

const _: () = assert!(
    std::mem::size_of::<CUmemAccessDesc_st>() == std::mem::size_of::<hip::HipMemAccessDesc>()
);

// Enum词汇表: aliases + flattened `<Type>_enum_<NAME>` constants, following
// the upstream generated naming convention. Values verified against
// $HOME/opt/cuda13/include/cuda.h (CUDA 13) — see repo docs for the
// extraction commands.

pub type CUctx_flags = u32;
pub type CUctx_flags_enum = u32;
pub type CUstream_flags = u32;
pub type CUevent_flags = u32;
pub type CUevent_wait_flags = u32;
pub type CUdevice_attribute = u32;
pub type CUfunction_attribute = u32;
pub type CUfunction_attribute_enum = u32;
pub type CUfunc_cache_enum = u32;
pub type CUlimit = u32;
pub type CUmemPool_attribute = u32;
pub type CUmemAllocationType = u32;
pub type CUmemAllocationHandleType = u32;
pub type CUmemLocationType = u32;
pub type CUmemAllocationGranularity_flags = u32;
pub type CUmemAccess_flags = u32;
pub type CUmemAttach_flags = u32;
pub type CUmem_advise = u32;
pub type CUmulticastGranularity_flags = u32;
pub type CUstreamCaptureMode = u32;
pub type CUstreamCaptureStatus = u32;

macro_rules! cuda_consts {
    ($($(#[$doc:meta])* $name:ident = $value:expr;)*) => {
        $($(#[$doc])* pub const $name: u32 = $value;)*
    };
}

cuda_consts! {
    cudaError_enum_CUDA_SUCCESS = 0;
    cudaError_enum_CUDA_ERROR_INVALID_VALUE = 1;
    cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY = 2;
    cudaError_enum_CUDA_ERROR_NOT_INITIALIZED = 3;
    cudaError_enum_CUDA_ERROR_DEINITIALIZED = 4;
    cudaError_enum_CUDA_ERROR_NO_DEVICE = 100;
    cudaError_enum_CUDA_ERROR_INVALID_DEVICE = 101;
    cudaError_enum_CUDA_ERROR_INVALID_IMAGE = 200;
    cudaError_enum_CUDA_ERROR_INVALID_CONTEXT = 201;
    cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303;
    cudaError_enum_CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222;
    cudaError_enum_CUDA_ERROR_INVALID_HANDLE = 400;
    cudaError_enum_CUDA_ERROR_NOT_FOUND = 500;
    cudaError_enum_CUDA_ERROR_NOT_READY = 600;
    cudaError_enum_CUDA_ERROR_ILLEGAL_ADDRESS = 700;
    cudaError_enum_CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704;
    cudaError_enum_CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705;
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED = 801;
    cudaError_enum_CUDA_ERROR_INVALID_CLUSTER_SIZE = 912;

    CUctx_flags_enum_CU_CTX_SCHED_AUTO = 0x00;
    CUctx_flags_enum_CU_CTX_SCHED_SPIN = 0x01;
    CUctx_flags_enum_CU_CTX_SCHED_YIELD = 0x02;
    CUctx_flags_enum_CU_CTX_SCHED_BLOCKING_SYNC = 0x04;
    CUctx_flags_enum_CU_CTX_SCHED_MASK = 0x07;
    CUstream_flags_enum_CU_STREAM_DEFAULT = 0x0;
    CUstream_flags_enum_CU_STREAM_NON_BLOCKING = 0x1;
    CUevent_flags_enum_CU_EVENT_DEFAULT = 0x0;
    CUevent_flags_enum_CU_EVENT_BLOCKING_SYNC = 0x1;
    CUevent_flags_enum_CU_EVENT_DISABLE_TIMING = 0x2;
    CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT = 0x0;
    CUevent_wait_flags_enum_CU_EVENT_WAIT_EXTERNAL = 0x1;

    // CUDA attr -> verified HIP attr mapping lives in cuDeviceGetAttribute.
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120;
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132;

    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0;
    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1;
    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2;
    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;
    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS = 4;
    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;
    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11;
    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12;
    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13;

    CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2;
    CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4;

    CUlimit_enum_CU_LIMIT_STACK_SIZE = 0x00;
    CUlimit_enum_CU_LIMIT_PRINTF_FIFO_SIZE = 0x01;
    CUlimit_enum_CU_LIMIT_MALLOC_HEAP_SIZE = 0x02;
    CUlimit_enum_CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03;
    CUlimit_enum_CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04;
    CUlimit_enum_CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05;
    CUlimit_enum_CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06;

    CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL = 0x1;
    CUmemAttach_flags_enum_CU_MEM_ATTACH_HOST = 0x2;
    CUmemAttach_flags_enum_CU_MEM_ATTACH_SINGLE = 0x4;

    CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_INVALID = 0x0;
    CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED = 0x1;
    CUmemAllocationHandleType_enum_CU_MEM_HANDLE_TYPE_NONE = 0x0;
    CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_INVALID = 0x0;
    CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE = 0x1;
    CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0;
    CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1;
    CUmemAccess_flags_enum_CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1;
    CUmemAccess_flags_enum_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3;
    CUmulticastGranularity_flags_enum_CU_MULTICAST_GRANULARITY_MINIMUM = 0x0;
    CUmulticastGranularity_flags_enum_CU_MULTICAST_GRANULARITY_RECOMMENDED = 0x1;

    CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1;
    CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2;
    CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3;
    CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4;
    CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5;
    CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6;
    CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7;
    CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8;

    CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2;
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4;
}

/// Writes the HIP device id into a `CUmemLocation_st`. `CUmemLocation_st` is
/// byte-identical to `hipMemLocation`, so this is a plain field store.
pub fn set_mem_location_id(loc: &mut CUmemLocation_st, id: i32) {
    loc.id = id;
}

// ---------------------------------------------------------------------------
// Dispatch: CUDA-named entry points -> HIP
// ---------------------------------------------------------------------------

/// Init / device / context

pub unsafe extern "C" fn cuInit(flags: u32) -> CUresult {
    hip!(hipInit(flags))
}

pub unsafe extern "C" fn cuDriverGetVersion(version: *mut i32) -> CUresult {
    // Report the loaded HIP runtime; no consumer keys behavior on it.
    hip!(hipInit(0));
    if !version.is_null() {
        // ROCm 7.x major.minor packed like CUDA (e.g. 7.14 -> 7014).
        unsafe { *version = 7_014 };
    }
    cudaError_enum_CUDA_SUCCESS
}

pub unsafe extern "C" fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult {
    hip!(hipDeviceGet(device as *mut hip::HipDevice, ordinal))
}

pub unsafe extern "C" fn cuDeviceGetCount(count: *mut i32) -> CUresult {
    hip!(hipGetDeviceCount(count))
}

pub unsafe extern "C" fn cuDeviceGetName(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult {
    hip!(hipDeviceGetName(name, len, dev))
}

pub unsafe extern "C" fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    hip!(hipDeviceTotalMem(bytes, dev))
}

pub unsafe extern "C" fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult {
    let r = hip!(hipDeviceGetUuid(uuid as *mut HipUuid, dev));
    r
}

pub unsafe extern "C" fn cuDeviceGetAttribute(
    value: *mut i32,
    attribute: CUdevice_attribute,
    dev: CUdevice,
) -> CUresult {
    // CUDA attr id -> HIP attr id. Verified live on gfx1030 (ROCm 7.14):
    // MultiprocessorCount -> 40 (RX 6950 XT), Compute Major/Minor -> 10/3,
    // MaxThreadsPerBlock -> 1024, L2 -> 4 MiB, MaxGridDimX -> 2^31-1.
    let hip_attr = match attribute {
        1 => 56,  // MAX_THREADS_PER_BLOCK -> hipDeviceAttributeMaxThreadsPerBlock
        2 => 26,  // MAX_BLOCK_DIM_X -> MaxBlockDimX
        3 => 27,  // MAX_BLOCK_DIM_Y
        4 => 28,  // MAX_BLOCK_DIM_Z
        5 => 29,  // MAX_GRID_DIM_X
        6 => 30,  // MAX_GRID_DIM_Y
        7 => 31,  // MAX_GRID_DIM_Z
        8 => 76,  // MAX_SHARED_MEMORY_PER_BLOCK -> MaxSharedMemoryPerBlock
        13 => 5,  // CLOCK_RATE -> ClockRate (verified: 2720000 kHz on gfx1030)
        16 => 63, // MULTIPROCESSOR_COUNT -> MultiprocessorCount
        38 => 19, // L2_CACHE_SIZE -> L2CacheSize
        39 => 57, // MAX_THREADS_PER_MULTIPROCESSOR -> MaxThreadsPerMultiProcessor
        75 => 23, // COMPUTE_CAPABILITY_MAJOR -> ComputeCapabilityMajor
        76 => 61, // COMPUTE_CAPABILITY_MINOR -> ComputeCapabilityMinor
        95 => 10, // COOPERATIVE_LAUNCH -> CooperativeLaunch
        97 => 77, // MAX_SHARED_MEMORY_PER_BLOCK_OPTIN -> SharedMemPerBlockOptin
        // RDNA2 has no thread-block clusters and no switch multicast; the
        // attrs are answered explicitly rather than with a wrong 0.
        120 | 132 => return cudaError_enum_CUDA_ERROR_NOT_SUPPORTED,
        _ => return cudaError_enum_CUDA_ERROR_NOT_SUPPORTED,
    };
    hip!(hipDeviceGetAttribute(value, hip_attr as i32, dev))
}

pub unsafe extern "C" fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    hip!(hipDevicePrimaryCtxRetain(pctx as *mut HipCtx, dev))
}

pub unsafe extern "C" fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    hip!(hipDevicePrimaryCtxRelease(dev))
}

pub unsafe extern "C" fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: CUctx_flags) -> CUresult {
    hip!(hipDevicePrimaryCtxSetFlags(dev, flags as u32))
}

pub unsafe extern "C" fn cuDevicePrimaryCtxGetState(
    dev: CUdevice,
    flags: *mut CUctx_flags,
    active: *mut i32,
) -> CUresult {
    hip!(hipDevicePrimaryCtxGetState(dev, flags as *mut u32, active))
}

pub unsafe extern "C" fn cuDeviceCanAccessPeer(
    canAccessPeer: *mut i32,
    dev: CUdevice,
    peerDev: CUdevice,
) -> CUresult {
    hip!(hipDeviceCanAccessPeer(canAccessPeer, dev, peerDev))
}

pub unsafe extern "C" fn cuDeviceGetDefaultMemPool(
    pool: *mut CUmemoryPool,
    dev: CUdevice,
) -> CUresult {
    hip!(hipDeviceGetDefaultMemPool(pool as *mut HipMemoryPool, dev))
}

pub unsafe extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    hip!(hipCtxGetCurrent(pctx as *mut HipCtx))
}

pub unsafe extern "C" fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    hip!(hipCtxSetCurrent(ctx as HipCtx))
}

pub unsafe extern "C" fn cuCtxSynchronize() -> CUresult {
    // HIP has no hipCtxSync; the primary-context equivalent is device sync.
    hip!(hipDeviceSynchronize())
}

pub unsafe extern "C" fn cuCtxSetFlags(_flags: CUctx_flags) -> CUresult {
    // No hipCtxSetFlags in ROCm 7.x.
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe extern "C" fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut u32) -> CUresult {
    hip!(hipCtxGetApiVersion(ctx as HipCtx, version))
}

pub unsafe extern "C" fn cuCtxSetLimit(_limit: CUlimit, _value: usize) -> CUresult {
    // HIP limit enum numbering differs from CU_LIMIT_*; unsupported rather
    // than misnumbered.
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe extern "C" fn cuCtxGetLimit(_pvalue: *mut usize, _limit: CUlimit) -> CUresult {
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe extern "C" fn cuCtxGetStreamPriorityRange(
    leastPriority: *mut i32,
    greatestPriority: *mut i32,
) -> CUresult {
    hip!(hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority))
}

pub unsafe extern "C" fn cuCtxEnablePeerAccess(peerContext: CUcontext, flags: u32) -> CUresult {
    hip!(hipCtxEnablePeerAccess(peerContext as HipCtx, flags))
}

pub unsafe extern "C" fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult {
    hip!(hipCtxDisablePeerAccess(peerContext as HipCtx))
}

/// Streams

pub unsafe extern "C" fn cuStreamCreate(
    pStream: *mut CUstream,
    flags: CUstream_flags,
) -> CUresult {
    hip!(hipStreamCreateWithFlags(pStream as *mut HipStream, flags as u32))
}

pub unsafe extern "C" fn cuStreamCreateWithPriority(
    pStream: *mut CUstream,
    flags: CUstream_flags,
    priority: i32,
) -> CUresult {
    hip!(hipStreamCreateWithPriority(
        pStream as *mut HipStream,
        flags as u32,
        priority,
    ))
}

pub unsafe extern "C" fn cuStreamGetPriority(hStream: CUstream, priority: *mut i32) -> CUresult {
    hip!(hipStreamGetPriority(hStream as HipStream, priority))
}

pub unsafe extern "C" fn cuStreamQuery(hStream: CUstream) -> CUresult {
    hip!(hipStreamQuery(hStream as HipStream))
}

pub unsafe extern "C" fn cuStreamSynchronize(hStream: CUstream) -> CUresult {
    hip!(hipStreamSynchronize(hStream as HipStream))
}

pub unsafe extern "C" fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult {
    hip!(hipStreamDestroy(hStream as HipStream))
}

pub unsafe extern "C" fn cuStreamWaitEvent(
    hStream: CUstream,
    hEvent: CUevent,
    flags: CUevent_wait_flags,
) -> CUresult {
    hip!(hipStreamWaitEvent(hStream as HipStream, hEvent as HipEvent, flags as u32))
}

pub unsafe extern "C" fn cuStreamIsCapturing(
    hStream: CUstream,
    status: *mut CUstreamCaptureStatus,
) -> CUresult {
    hip!(hipStreamIsCapturing(hStream as HipStream, status as *mut i32))
}

pub unsafe extern "C" fn cuStreamBeginCapture_v2(
    hStream: CUstream,
    mode: CUstreamCaptureMode,
) -> CUresult {
    hip!(hipStreamBeginCapture(hStream as HipStream, mode as i32))
}

pub unsafe extern "C" fn cuStreamEndCapture(hStream: CUstream, pGraph: *mut CUgraph) -> CUresult {
    hip!(hipStreamEndCapture(hStream as HipStream, pGraph as *mut *mut c_void))
}

pub unsafe extern "C" fn cuStreamAttachMemAsync(
    hStream: CUstream,
    dptr: CUdeviceptr,
    length: usize,
    flags: u32,
) -> CUresult {
    hip!(hipStreamAttachMemAsync(
        hStream as HipStream,
        dptr as usize as *mut c_void,
        length,
        flags,
    ))
}

/// Host functions

pub unsafe extern "C" fn cuLaunchHostFunc(
    hStream: CUstream,
    callback: Option<unsafe extern "C" fn(*mut c_void)>,
    userData: *mut c_void,
) -> CUresult {
    let Some(callback) = callback else {
        return cudaError_enum_CUDA_ERROR_INVALID_VALUE;
    };
    hip!(hipLaunchHostFunc(hStream as HipStream, callback, userData))
}

/// Newer 4-arg host-func variant (stream, callback, data, mode). No HIP
/// equivalent; unused by the gfx1030 example paths.
pub unsafe extern "C" fn cu_launch_host_func(
    _hStream: CUstream,
    _callback: Option<unsafe extern "C" fn(*mut c_void)>,
    _userData: *mut c_void,
    _mode: u32,
) -> CUresult {
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

/// Events

pub unsafe extern "C" fn cuEventCreate(phEvent: *mut CUevent, flags: CUevent_flags) -> CUresult {
    hip!(hipEventCreateWithFlags(phEvent as *mut HipEvent, flags as u32))
}

pub unsafe extern "C" fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult {
    hip!(hipEventRecord(hEvent as HipEvent, hStream as HipStream))
}

pub unsafe extern "C" fn cuEventQuery(hEvent: CUevent) -> CUresult {
    hip!(hipEventQuery(hEvent as HipEvent))
}

pub unsafe extern "C" fn cuEventSynchronize(hEvent: CUevent) -> CUresult {
    hip!(hipEventSynchronize(hEvent as HipEvent))
}

pub unsafe extern "C" fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult {
    hip!(hipEventDestroy(hEvent as HipEvent))
}

pub unsafe extern "C" fn cuEventElapsedTime(
    pMilliseconds: *mut f32,
    hStart: CUevent,
    hEnd: CUevent,
) -> CUresult {
    hip!(hipEventElapsedTime(pMilliseconds, hStart as HipEvent, hEnd as HipEvent))
}

pub unsafe extern "C" fn cuEventElapsedTime_v2(
    pMilliseconds: *mut f32,
    hStart: CUevent,
    hEnd: CUevent,
) -> CUresult {
    cuEventElapsedTime(pMilliseconds, hStart, hEnd)
}

/// Upstream helper wrapper (cuda-core `CudaEvent::elapsed_time`).
pub unsafe extern "C" fn cu_event_elapsed_time(
    ms: *mut f32,
    start: CUevent,
    end: CUevent,
) -> CUresult {
    cuEventElapsedTime(ms, start, end)
}

/// Memory: alloc / free / copy / set

pub unsafe extern "C" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    hip!(hipMalloc(dptr as *mut *mut c_void, bytesize))
}

pub unsafe extern "C" fn cuMemAllocAsync(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    hStream: CUstream,
) -> CUresult {
    hip!(hipMallocAsync(dptr as *mut *mut c_void, bytesize, hStream as HipStream))
}

pub unsafe extern "C" fn cuMemAllocFromPoolAsync(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    pool: CUmemoryPool,
    hStream: CUstream,
) -> CUresult {
    hip!(hipMallocFromPoolAsync(
        dptr as *mut *mut c_void,
        bytesize,
        pool as HipMemoryPool,
        hStream as HipStream,
    ))
}

pub unsafe extern "C" fn cuMemAllocManaged(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    flags: CUmemAttach_flags,
) -> CUresult {
    hip!(hipMallocManaged(dptr as *mut *mut c_void, bytesize, flags as u32))
}

pub unsafe extern "C" fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    hip!(hipHostAlloc(pp, bytesize, 0))
}

pub unsafe extern "C" fn cuMemHostAlloc(
    pp: *mut *mut c_void,
    bytesize: usize,
    flags: u32,
) -> CUresult {
    hip!(hipHostAlloc(pp, bytesize, flags))
}

pub unsafe extern "C" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    hip!(hipFree(dptr as usize as *mut c_void))
}

pub unsafe extern "C" fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult {
    hip!(hipFreeAsync(dptr as usize as *mut c_void, hStream as HipStream))
}

pub unsafe extern "C" fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    hip!(hipHostFree(p))
}

pub unsafe extern "C" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    hip!(hipMemGetInfo(free, total))
}

pub unsafe extern "C" fn cuMemcpyHtoD_v2(
    dst: CUdeviceptr,
    src: *const c_void,
    bytes: usize,
) -> CUresult {
    // hipMemcpyKind mirrors cudaMemcpyKind numbering (H2D = 1).
    hip!(hipMemcpy(dst as usize as *mut c_void, src, bytes, 1))
}

pub unsafe extern "C" fn cuMemcpyDtoH_v2(
    dst: *mut c_void,
    src: CUdeviceptr,
    bytes: usize,
) -> CUresult {
    hip!(hipMemcpy(dst, src as usize as *const c_void, bytes, 2))
}

pub unsafe extern "C" fn cuMemcpyDtoD_v2(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    bytes: usize,
) -> CUresult {
    hip!(hipMemcpy(
        dst as usize as *mut c_void,
        src as usize as *const c_void,
        bytes,
        3,
    ))
}

pub unsafe extern "C" fn cuMemcpyHtoDAsync_v2(
    dst: CUdeviceptr,
    src: *const c_void,
    bytes: usize,
    hStream: CUstream,
) -> CUresult {
    hip!(hipMemcpyAsync(
        dst as usize as *mut c_void,
        src,
        bytes,
        1,
        hStream as HipStream,
    ))
}

pub unsafe extern "C" fn cuMemcpyDtoHAsync_v2(
    dst: *mut c_void,
    src: CUdeviceptr,
    bytes: usize,
    hStream: CUstream,
) -> CUresult {
    hip!(hipMemcpyAsync(
        dst,
        src as usize as *const c_void,
        bytes,
        2,
        hStream as HipStream,
    ))
}

pub unsafe extern "C" fn cuMemcpyDtoDAsync_v2(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    bytes: usize,
    hStream: CUstream,
) -> CUresult {
    hip!(hipMemcpyAsync(
        dst as usize as *mut c_void,
        src as usize as *const c_void,
        bytes,
        3,
        hStream as HipStream,
    ))
}

pub unsafe extern "C" fn cuMemsetD8_v2(dptr: CUdeviceptr, value: u8, n: usize) -> CUresult {
    hip!(hipMemsetD8(dptr as usize as *mut c_void, value, n))
}

pub unsafe extern "C" fn cuMemsetD8Async(
    dptr: CUdeviceptr,
    value: u8,
    n: usize,
    hStream: CUstream,
) -> CUresult {
    hip!(hipMemsetD8Async(dptr as usize as *mut c_void, value, n, hStream as HipStream))
}

pub unsafe extern "C" fn cuMemPrefetchAsync_v2(
    dptr: CUdeviceptr,
    count: usize,
    location: CUmemLocation_st,
    flags: u32,
    hStream: CUstream,
) -> CUresult {
    hip!(hipMemPrefetchAsync(
        dptr as usize as *const c_void,
        count,
        location_to_device(location),
        flags,
        hStream as HipStream,
    ))
}

/// CUmemLocation_st{DEVICE, id} -> HIP device ordinal (the byte-identical
/// struct means id already is the device).
fn location_to_device(location: CUmemLocation_st) -> i32 {
    if location.type_ == CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE {
        location.id
    } else {
        -1
    }
}

pub unsafe extern "C" fn cuMemAdvise_v2(
    dptr: CUdeviceptr,
    count: usize,
    advice: CUmem_advise,
    location: CUmemLocation_st,
) -> CUresult {
    hip!(hipMemAdvise(
        dptr as usize as *const c_void,
        count,
        advice as i32,
        location_to_device(location),
    ))
}

/// Virtual memory management

pub unsafe extern "C" fn cuMemAddressReserve(
    ptr: *mut CUdeviceptr,
    size: usize,
    alignment: usize,
    addr: CUdeviceptr,
    flags: u64,
) -> CUresult {
    hip!(hipMemAddressReserve(
        ptr as *mut *mut c_void,
        size,
        alignment,
        addr as usize as *mut c_void,
        flags,
    ))
}

pub unsafe extern "C" fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult {
    hip!(hipMemAddressFree(ptr as usize as *mut c_void, size))
}

fn translate_allocation_prop(src: &CUmemAllocationProp_st) -> hip::HipMemAllocationProp {
    hip::HipMemAllocationProp {
        // CU_MEM_ALLOCATION_TYPE_PINNED == hipMemAllocationTypePinned == 1.
        type_: src.type_ as i32,
        // CU_MEM_HANDLE_TYPE_NONE == hipMemHandleTypeGeneric == 0.
        requestedHandleType: src.requestedHandleType as i32,
        // Byte-identical layout.
        location: HipMemLocation {
            type_: src.location.type_ as i32,
            id: src.location.id,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: hip::HipMemAllocationPropAllocFlags {
            compressionType: 0,
            gpuDirectRDMACapable: 0,
            usage: 0,
        },
    }
}

pub unsafe extern "C" fn cuMemCreate(
    handle: *mut CUmemGenericAllocationHandle,
    size: usize,
    prop: *const CUmemAllocationProp_st,
    flags: u64,
) -> CUresult {
    if prop.is_null() {
        return cudaError_enum_CUDA_ERROR_INVALID_VALUE;
    }
    let hip_prop = translate_allocation_prop(&*prop);
    hip!(hipMemCreate(handle as *mut HipMemGenericAllocationHandle, size, &hip_prop, flags))
}

pub unsafe extern "C" fn cuMemMap(
    ptr: CUdeviceptr,
    size: usize,
    offset: usize,
    handle: CUmemGenericAllocationHandle,
    flags: u64,
) -> CUresult {
    hip!(hipMemMap(
        ptr as usize as *mut c_void,
        size,
        offset,
        handle as HipMemGenericAllocationHandle,
        flags,
    ))
}

pub unsafe extern "C" fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult {
    hip!(hipMemUnmap(ptr as usize as *mut c_void, size))
}

pub unsafe extern "C" fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult {
    hip!(hipMemRelease(handle as HipMemGenericAllocationHandle))
}

pub unsafe extern "C" fn cuMemSetAccess(
    ptr: CUdeviceptr,
    size: usize,
    desc: *const CUmemAccessDesc_st,
    count: usize,
) -> CUresult {
    // CUmemAccessDesc_st is byte-identical to hipMemAccessDesc (compile-time
    // asserted above).
    hip!(hipMemSetAccess(
        ptr as usize as *mut c_void,
        size,
        desc as *const HipMemAccessDesc,
        count,
    ))
}

pub unsafe extern "C" fn cuMemGetAllocationGranularity(
    granularity: *mut usize,
    prop: *const CUmemAllocationProp_st,
    option: CUmemAllocationGranularity_flags,
) -> CUresult {
    if prop.is_null() {
        return cudaError_enum_CUDA_ERROR_INVALID_VALUE;
    }
    let hip_prop = translate_allocation_prop(&*prop);
    hip!(hipMemGetAllocationGranularity(granularity, &hip_prop, option as i32))
}

/// Memory pools

pub unsafe extern "C" fn cuMemPoolCreate(
    pool: *mut CUmemoryPool,
    poolProps: *const CUmemPoolProps,
) -> CUresult {
    if poolProps.is_null() {
        return cudaError_enum_CUDA_ERROR_INVALID_VALUE;
    }
    let src = &*poolProps;
    let hip_props = HipMemPoolProps {
        // CU_MEM_ALLOCATION_TYPE_* == hipMemAllocationType_* numbering.
        allocType: src.allocType as i32,
        handleTypes: src.handleTypes as i32,
        location: HipMemLocation {
            type_: src.location.type_ as i32,
            id: src.location.id,
        },
        win32SecurityAttributes: std::ptr::null_mut(),
        // HIP field is named `maxSize`.
        maxSize: src.maxPoolSize,
        reserved: [0; 56],
    };
    hip!(hipMemPoolCreate(pool as *mut HipMemoryPool, &hip_props))
}

pub unsafe extern "C" fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult {
    hip!(hipMemPoolDestroy(pool as HipMemoryPool))
}

pub unsafe extern "C" fn cuMemPoolSetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut c_void,
) -> CUresult {
    // CU_MEMPOOL_ATTR_* == hipMemPoolAttr numbering (1..8), value shapes match.
    hip!(hipMemPoolSetAttribute(pool as HipMemoryPool, attr as i32, value))
}

pub unsafe extern "C" fn cuMemPoolGetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut c_void,
) -> CUresult {
    hip!(hipMemPoolGetAttribute(pool as HipMemoryPool, attr as i32, value))
}

/// Multicast: no HIP surface on ROCm 7.x and no RDNA2 hardware. Explicit
/// unsupported (the upstream wrappers surface the error code as a Result).

pub unsafe extern "C" fn cuMulticastCreate(
    _mcHandle: *mut CUmemGenericAllocationHandle,
    _prop: *const CUmulticastObjectProp_st,
) -> CUresult {
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe extern "C" fn cuMulticastGetGranularity(
    _sizehint: *mut usize,
    _prop: *const CUmulticastObjectProp_st,
    _option: CUmulticastGranularity_flags,
) -> CUresult {
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe extern "C" fn cuMulticastAddDevice(
    _mcHandle: CUmemGenericAllocationHandle,
    _device: CUdevice,
) -> CUresult {
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe extern "C" fn cuMulticastBindMem(
    _mcHandle: CUmemGenericAllocationHandle,
    _mcOffset: usize,
    _memHandle: CUmemGenericAllocationHandle,
    _memOffset: usize,
    _size: usize,
    _flags: u64,
) -> CUresult {
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe extern "C" fn cuMulticastUnbind(
    _mcHandle: CUmemGenericAllocationHandle,
    _device: CUdevice,
    _mcOffset: usize,
    _size: usize,
) -> CUresult {
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

/// Modules & kernels

pub unsafe extern "C" fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult {
    hip!(hipModuleLoad(module as *mut HipModule, fname))
}

pub unsafe extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    let r = hip!(hipModuleLoadData(module as *mut HipModule, image));
    if r == cudaError_enum_CUDA_SUCCESS {
        // [PORT gfx1030 Stage2] ROCm 7.x finalizes freshly loaded module code
        // asynchronously; a launch racing the finalize can silently no-op
        // (DeepGEMM port evidence, Phase 1 notes). Drain it host-side before
        // any launch — microseconds on the small example modules.
        let _ = hip!(hipDeviceSynchronize());
    }
    r
}

// [PORT gfx1030 PhaseB.1] pinned-host device alias + stream write (the
// cuda-async reactor's slot-table path)
pub unsafe extern "C" fn cuMemHostGetDevicePointer_v2(
    dptr: *mut CUdeviceptr,
    host: *mut c_void,
    flags: u32,
) -> CUresult {
    hip!(hipHostGetDevicePointer(
        dptr as *mut *mut c_void,
        host,
        flags,
    ))
}

pub unsafe extern "C" fn cuStreamWriteValue32_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: u32,
    flags: u32,
) -> CUresult {
    hip!(hipStreamWriteValue32(
        stream as hip::HipStream,
        addr as *mut c_void,
        value as i32,
        flags,
    ))
}

// [PORT gfx1030 PhaseB.1] CUDA graph surface for cuda-async (cuda_graph.rs
// calls exactly these five; hipGraph* maps 1:1 in ROCm 7.x). The async
// example surface compiles and runs; graph capture/replay semantics mirror
// CUDA where HIP implements them and fail loudly where it does not.
pub unsafe extern "C" fn cuGraphInstantiateWithFlags(
    exec: *mut CUgraphExec,
    graph: CUgraph,
    flags: u64,
) -> CUresult {
    hip!(hipGraphInstantiateWithFlags(
        exec as *mut hip::HipGraphExec,
        graph as hip::HipGraph,
        flags,
    ))
}

pub unsafe extern "C" fn cuGraphLaunch(exec: CUgraphExec, stream: CUstream) -> CUresult {
    hip!(hipGraphLaunch(
        exec as hip::HipGraphExec,
        stream as hip::HipStream,
    ))
}

pub unsafe extern "C" fn cuGraphUpload(exec: CUgraphExec, stream: CUstream) -> CUresult {
    hip!(hipGraphUpload(
        exec as hip::HipGraphExec,
        stream as hip::HipStream,
    ))
}

pub unsafe extern "C" fn cuGraphDestroy(graph: CUgraph) -> CUresult {
    hip!(hipGraphDestroy(graph as hip::HipGraph))
}

pub unsafe extern "C" fn cuGraphExecDestroy(exec: CUgraphExec) -> CUresult {
    hip!(hipGraphExecDestroy(exec as hip::HipGraphExec))
}

pub unsafe extern "C" fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    hip!(hipModuleUnload(hmod as HipModule))
}

pub unsafe extern "C" fn cuModuleGetFunction(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    hip!(hipModuleGetFunction(hfunc as *mut HipFunction, hmod as HipModule, name))
}

pub unsafe extern "C" fn cuModuleGetGlobal_v2(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    // hipModuleGetGlobal returns a device pointer as void*.
    let mut global: *mut c_void = std::ptr::null_mut();
    let r = match hip() {
        Ok(api) => match api.hipModuleGetGlobal {
            Ok(f) => unsafe { f(&mut global, bytes, hmod as HipModule, name) as u32 },
            Err(_) => cudaError_enum_CUDA_ERROR_NOT_SUPPORTED,
        },
        Err(code) => code,
    };
    if r == cudaError_enum_CUDA_SUCCESS && !dptr.is_null() {
        unsafe { *dptr = global as usize as CUdeviceptr };
    }
    r
}

pub unsafe extern "C" fn cuFuncSetAttribute(
    hfunc: CUfunction,
    attr: CUfunction_attribute,
    value: i32,
) -> CUresult {
    // CU_FUNC_ATTRIBUTE_* == hipFuncAttribute numbering for the shared set.
    hip!(hipFuncSetAttribute(hfunc as HipFunction, attr as i32, value))
}

pub unsafe extern "C" fn cuFuncGetAttribute(
    pi: *mut i32,
    attribute: CUfunction_attribute,
    func: CUfunction,
) -> CUresult {
    hip!(hipFuncGetAttribute(pi, attribute as i32, func as HipFunction))
}

pub unsafe extern "C" fn cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache_enum) -> CUresult {
    hip!(hipFuncSetCacheConfig(hfunc as HipFunction, config as i32))
}

pub unsafe extern "C" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: u32,
    gridDimY: u32,
    gridDimZ: u32,
    blockDimX: u32,
    blockDimY: u32,
    blockDimZ: u32,
    sharedMemBytes: u32,
    hStream: CUstream,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    hip!(hipModuleLaunchKernel(
        f as HipFunction,
        gridDimX,
        gridDimY,
        gridDimZ,
        blockDimX,
        blockDimY,
        blockDimZ,
        sharedMemBytes,
        hStream as HipStream,
        kernelParams,
        extra,
    ))
}

pub unsafe extern "C" fn cuLaunchKernelEx(
    config: *const CUlaunchConfig_st,
    f: CUfunction,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    if config.is_null() {
        return cudaError_enum_CUDA_ERROR_INVALID_VALUE;
    }
    let cfg = unsafe { &*config };
    if cfg.numAttrs != 0 || !cfg.attrs.is_null() {
        // Cluster / cooperative attributes are not expressible through
        // hipModuleLaunchKernel and have no module-level HIP equivalent on
        // RDNA2. Explicit unsupported — the gfx1030 example set launches
        // plain kernels only.
        return cudaError_enum_CUDA_ERROR_NOT_SUPPORTED;
    }
    hip!(hipModuleLaunchKernel(
        f as HipFunction,
        cfg.gridDimX,
        cfg.gridDimY,
        cfg.gridDimZ,
        cfg.blockDimX,
        cfg.blockDimY,
        cfg.blockDimZ,
        cfg.sharedMemBytes,
        cfg.hStream as HipStream,
        kernelParams,
        extra,
    ))
}

/// Occupancy

pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
    pnumBlocks: *mut i32,
    func: CUfunction,
    blockSize: i32,
    dynamicSMemSize: usize,
) -> CUresult {
    hip!(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        pnumBlocks,
        func as HipFunction,
        blockSize,
        dynamicSMemSize,
    ))
}

pub unsafe extern "C" fn cuOccupancyMaxActiveClusters(
    _pnumClusters: *mut i32,
    _func: CUfunction,
    _config: *const CUlaunchConfig_st,
) -> CUresult {
    // Cluster occupancy: no RDNA2 equivalent.
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe extern "C" fn cuOccupancyMaxPotentialClusterSize(
    _pmaxClusterSize: *mut i32,
    _func: CUfunction,
    _config: *const CUlaunchConfig_st,
) -> CUresult {
    cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
}

/// Error strings

static ERROR_NAMES: &[(u32, &[u8])] = &[
    (cudaError_enum_CUDA_SUCCESS, b"CUDA_SUCCESS\0"),
    (cudaError_enum_CUDA_ERROR_INVALID_VALUE, b"CUDA_ERROR_INVALID_VALUE\0"),
    (cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY, b"CUDA_ERROR_OUT_OF_MEMORY\0"),
    (cudaError_enum_CUDA_ERROR_NOT_INITIALIZED, b"CUDA_ERROR_NOT_INITIALIZED\0"),
    (cudaError_enum_CUDA_ERROR_DEINITIALIZED, b"CUDA_ERROR_DEINITIALIZED\0"),
    (cudaError_enum_CUDA_ERROR_NO_DEVICE, b"CUDA_ERROR_NO_DEVICE\0"),
    (cudaError_enum_CUDA_ERROR_INVALID_DEVICE, b"CUDA_ERROR_INVALID_DEVICE\0"),
    (cudaError_enum_CUDA_ERROR_INVALID_IMAGE, b"CUDA_ERROR_INVALID_IMAGE\0"),
    (cudaError_enum_CUDA_ERROR_INVALID_CONTEXT, b"CUDA_ERROR_INVALID_CONTEXT\0"),
    (
        cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
        b"CUDA_ERROR_SHARED_OBJECT_INIT_FAILED\0",
    ),
    (
        cudaError_enum_CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
        b"CUDA_ERROR_UNSUPPORTED_PTX_VERSION\0",
    ),
    (cudaError_enum_CUDA_ERROR_INVALID_HANDLE, b"CUDA_ERROR_INVALID_HANDLE\0"),
    (cudaError_enum_CUDA_ERROR_NOT_FOUND, b"CUDA_ERROR_NOT_FOUND\0"),
    (cudaError_enum_CUDA_ERROR_NOT_READY, b"CUDA_ERROR_NOT_READY\0"),
    (cudaError_enum_CUDA_ERROR_ILLEGAL_ADDRESS, b"CUDA_ERROR_ILLEGAL_ADDRESS\0"),
    (
        cudaError_enum_CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
        b"CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED\0",
    ),
    (
        cudaError_enum_CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
        b"CUDA_ERROR_PEER_ACCESS_NOT_ENABLED\0",
    ),
    (cudaError_enum_CUDA_ERROR_NOT_SUPPORTED, b"CUDA_ERROR_NOT_SUPPORTED\0"),
    (
        cudaError_enum_CUDA_ERROR_INVALID_CLUSTER_SIZE,
        b"CUDA_ERROR_INVALID_CLUSTER_SIZE\0",
    ),
];

const UNKNOWN_ERROR_NAME: &[u8] = b"CUDA_ERROR_UNKNOWN_HIP_CODE\0";

fn error_name_bytes(code: CUresult) -> &'static [u8] {
    ERROR_NAMES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, name)| *name)
        .unwrap_or(UNKNOWN_ERROR_NAME)
}

pub unsafe extern "C" fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult {
    if pStr.is_null() {
        return cudaError_enum_CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        *pStr = error_name_bytes(error).as_ptr() as *const c_char;
    }
    cudaError_enum_CUDA_SUCCESS
}

pub unsafe extern "C" fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult {
    cuGetErrorName(error, pStr)
}

// ---------------------------------------------------------------------------
// cuRAND-named RNG entry points -> hipRAND
// ---------------------------------------------------------------------------

pub type curandGenerator_t = *mut hip::Opaque;
/// `curandRngType` enum, flattened bindgen-style. cuRAND/hipRAND number the
/// pseudo-default as 100.
pub const curandRngType_CURAND_RNG_PSEUDO_DEFAULT: u32 = 100;

macro_rules! curand {
    ($field:ident ( $($arg:expr),* $(,)? )) => {
        match hip::init_hiprand() {
            Ok(api) => match api.$field {
                Ok(f) => unsafe { f($($arg),*) as u32 },
                Err(_) => hip::CURAND_STATUS_NOT_INITIALIZED as u32,
            },
            Err(_) => hip::CURAND_STATUS_NOT_INITIALIZED as u32,
        }
    };
}

pub unsafe extern "C" fn curandCreateGenerator(
    generator: *mut curandGenerator_t,
    rng_type: u32,
) -> CUresult {
    curand!(hiprandCreateGenerator(generator as *mut hip::HipRandGenerator, rng_type as i32))
}

pub unsafe extern "C" fn curandDestroyGenerator(generator: curandGenerator_t) -> CUresult {
    curand!(hiprandDestroyGenerator(generator as hip::HipRandGenerator))
}

pub unsafe extern "C" fn curandSetStream(
    generator: curandGenerator_t,
    stream: CUstream,
) -> CUresult {
    curand!(hiprandSetStream(generator as hip::HipRandGenerator, stream as HipStream))
}

pub unsafe extern "C" fn curandSetPseudoRandomGeneratorSeed(
    generator: curandGenerator_t,
    seed: u64,
) -> CUresult {
    curand!(hiprandSetPseudoRandomGeneratorSeed(generator as hip::HipRandGenerator, seed))
}

pub unsafe extern "C" fn curandGenerateUniform(
    generator: curandGenerator_t,
    output: *mut f32,
    num: usize,
) -> CUresult {
    curand!(hiprandGenerateUniform(generator as hip::HipRandGenerator, output, num))
}

pub unsafe extern "C" fn curandGenerateUniformDouble(
    generator: curandGenerator_t,
    output: *mut f64,
    num: usize,
) -> CUresult {
    curand!(hiprandGenerateUniformDouble(generator as hip::HipRandGenerator, output, num))
}

pub unsafe extern "C" fn curandGenerateNormal(
    generator: curandGenerator_t,
    output: *mut f32,
    num: usize,
    mean: f32,
    stddev: f32,
) -> CUresult {
    curand!(hiprandGenerateNormal(
        generator as hip::HipRandGenerator,
        output,
        num,
        mean,
        stddev,
    ))
}

pub unsafe extern "C" fn curandGenerateNormalDouble(
    generator: curandGenerator_t,
    output: *mut f64,
    num: usize,
    mean: f64,
    stddev: f64,
) -> CUresult {
    curand!(hiprandGenerateNormalDouble(
        generator as hip::HipRandGenerator,
        output,
        num,
        mean,
        stddev,
    ))
}
