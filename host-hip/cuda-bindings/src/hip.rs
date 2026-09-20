//! dlopen FFI layer for `libamdhip64` (HIP runtime).
//!
//! Mirrors the cuda-bindings 0.3.1 `dyn_load` pattern: symbols resolve into
//! `Result<fn, libloading::Error>` fields once, at first use; individual
//! entry points degrade to an error code when the symbol is missing from the
//! loaded runtime (older ROCm) instead of failing to load the whole crate.
//!
//! Signatures are written against ROCm 7.14 headers
//! (`/opt/rocm/include/hip/hip_runtime_api.h`, `driver_types.h`); all enum
//! constants embedded in the dispatch layer were verified against those
//! headers and against a live gfx1030 device (attribute-value probe), see
//! lib.rs comments.

#![allow(non_camel_case_types, dead_code, non_snake_case)]

use libloading::Library;
use std::ffi::c_void;
use std::os::raw::c_char;

/// Every HIP opaque handle is a pointer to an incomplete struct
/// (`typedef struct ihipX_t* hipX_t`); one marker type models them all.
#[repr(C)]
pub struct Opaque {
    _unused: [u8; 0],
}

pub type HipCtx = *mut Opaque;
pub type HipStream = *mut Opaque;
pub type HipModule = *mut Opaque;
pub type HipFunction = *mut Opaque;
pub type HipEvent = *mut Opaque;
pub type HipDevice = i32;
/// `hipMemGenericAllocationHandle_t` is also an opaque pointer.
pub type HipMemGenericAllocationHandle = *mut Opaque;
pub type HipMemoryPool = *mut Opaque;

/// `hipUUID` (`driver_types.h`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HipUuid {
    pub bytes: [c_char; 16],
}

/// `hipMemLocation` (`driver_types.h`): `{ type; int id; }` — byte-identical
/// to the `CUmemLocation_st` this crate models.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HipMemLocation {
    pub type_: i32,
    pub id: i32,
}

/// `hipMemAllocationProp` (`hip_runtime_api.h`), field-for-field.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HipMemAllocationProp {
    pub type_: i32,
    pub requestedHandleType: i32,
    pub location: HipMemLocation,
    pub win32HandleMetaData: *mut c_void,
    pub allocFlags: HipMemAllocationPropAllocFlags,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HipMemAllocationPropAllocFlags {
    pub compressionType: u8,
    pub gpuDirectRDMACapable: u8,
    pub usage: u16,
}

/// `hipMemAccessDesc` (`driver_types.h`): `{ location; flags; }`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HipMemAccessDesc {
    pub location: HipMemLocation,
    pub flags: i32,
}

/// `hipMemPoolProps` (`hip_runtime_api.h`), field-for-field.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HipMemPoolProps {
    pub allocType: i32,
    pub handleTypes: i32,
    pub location: HipMemLocation,
    pub win32SecurityAttributes: *mut c_void,
    pub maxSize: usize,
    pub reserved: [u8; 56],
}

/// Candidate sonames, tried in order. The host ROCm install ships
/// `libamdhip64.so.7`; containers commonly expose `.so.1`. `dlopen` searches
/// the ldconfig cache, so plain sonames suffice.
pub const HIP_LIB_NAMES: &[&str] = &[
    "libamdhip64.so.1",
    "libamdhip64.so",
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
];

unsafe fn get<T: Copy>(lib: &Library, name: &str) -> Result<T, libloading::Error> {
    // SAFETY: caller guarantees `lib` is a live HIP runtime handle; symbol
    // types are declared to match the ROCm 7.14 prototypes.
    let sym: libloading::Symbol<T> = unsafe { lib.get(name.as_bytes())? };
    Ok(*sym)
}

macro_rules! hip_api {
    (
        $apiname:ident {
            $(
                $name:ident : fn ( $( $arg:ident : $argty:ty ),* $(,)? ) -> $ret:ty ;
            )*
        }
    ) => {
        /// Function-pointer table for the HIP entry points the dispatch
        /// layer forwards to. Each field carries the loader's per-symbol
        /// failure so a missing optional symbol degrades to an error code.
        pub struct $apiname {
            $(
                pub $name: Result<
                    unsafe extern "C" fn($($argty),*) -> $ret,
                    libloading::Error,
                >,
            )*
        }

        impl $apiname {
            unsafe fn from_library(lib: &Library) -> $apiname {
                $apiname {
                    $(
                        $name: unsafe {
                            get(lib, concat!(stringify!($name), "\0"))
                        },
                    )*
                }
            }
        }
    };
}

hip_api! {
    HipApi {
    hipInit: fn(flags: u32) -> i32;
    hipGetDeviceCount: fn(count: *mut i32) -> i32;
    hipDeviceGet: fn(device: *mut HipDevice, ordinal: i32) -> i32;
    hipDeviceGetName: fn(name: *mut c_char, len: i32, device: HipDevice) -> i32;
    hipDeviceTotalMem: fn(bytes: *mut usize, device: HipDevice) -> i32;
    hipDeviceGetUuid: fn(uuid: *mut HipUuid, device: HipDevice) -> i32;
    hipDeviceGetAttribute: fn(value: *mut i32, attribute: i32, device: HipDevice) -> i32;
    hipDevicePrimaryCtxRetain: fn(ctx: *mut HipCtx, device: HipDevice) -> i32;
    hipDevicePrimaryCtxRelease: fn(device: HipDevice) -> i32;
    hipDevicePrimaryCtxSetFlags: fn(device: HipDevice, flags: u32) -> i32;
    hipDevicePrimaryCtxGetState: fn(device: HipDevice, flags: *mut u32, active: *mut i32) -> i32;
    hipDeviceCanAccessPeer: fn(canAccessPeer: *mut i32, device: HipDevice, peerDevice: HipDevice) -> i32;
    hipDeviceSynchronize: fn() -> i32;
    hipDeviceGetDefaultMemPool: fn(pool: *mut HipMemoryPool, device: HipDevice) -> i32;
    hipDeviceGetStreamPriorityRange: fn(leastPriority: *mut i32, greatestPriority: *mut i32) -> i32;
    hipDeviceSetLimit: fn(limit: i32, value: usize) -> i32;
    hipDeviceGetLimit: fn(pValue: *mut usize, limit: i32) -> i32;
    hipCtxGetCurrent: fn(ctx: *mut HipCtx) -> i32;
    hipCtxSetCurrent: fn(ctx: HipCtx) -> i32;
    hipCtxGetApiVersion: fn(ctx: HipCtx, apiVersion: *mut u32) -> i32;
    hipCtxEnablePeerAccess: fn(peerCtx: HipCtx, flags: u32) -> i32;
    hipCtxDisablePeerAccess: fn(peerCtx: HipCtx) -> i32;
    hipStreamCreate: fn(stream: *mut HipStream) -> i32;
    hipStreamCreateWithFlags: fn(stream: *mut HipStream, flags: u32) -> i32;
    hipStreamCreateWithPriority: fn(stream: *mut HipStream, flags: u32, priority: i32) -> i32;
    hipStreamGetPriority: fn(stream: HipStream, priority: *mut i32) -> i32;
    hipStreamQuery: fn(stream: HipStream) -> i32;
    hipStreamSynchronize: fn(stream: HipStream) -> i32;
    hipStreamDestroy: fn(stream: HipStream) -> i32;
    hipStreamWaitEvent: fn(stream: HipStream, event: HipEvent, flags: u32) -> i32;
    hipStreamIsCapturing: fn(stream: HipStream, pCaptureStatus: *mut i32) -> i32;
    hipStreamBeginCapture: fn(stream: HipStream, captureMode: i32) -> i32;
    hipStreamEndCapture: fn(stream: HipStream, pGraph: *mut *mut c_void) -> i32;
    hipStreamAttachMemAsync: fn(stream: HipStream, dptr: *mut c_void, length: usize, flags: u32) -> i32;
    hipLaunchHostFunc: fn(stream: HipStream, f: unsafe extern "C" fn(*mut c_void), userData: *mut c_void) -> i32;
    hipEventCreate: fn(event: *mut HipEvent) -> i32;
    hipEventCreateWithFlags: fn(event: *mut HipEvent, flags: u32) -> i32;
    hipEventRecord: fn(event: HipEvent, stream: HipStream) -> i32;
    hipEventQuery: fn(event: HipEvent) -> i32;
    hipEventSynchronize: fn(event: HipEvent) -> i32;
    hipEventDestroy: fn(event: HipEvent) -> i32;
    hipEventElapsedTime: fn(ms: *mut f32, start: HipEvent, end: HipEvent) -> i32;
    hipMalloc: fn(ptr: *mut *mut c_void, size: usize) -> i32;
    hipFree: fn(ptr: *mut c_void) -> i32;
    hipMallocAsync: fn(ptr: *mut *mut c_void, size: usize, stream: HipStream) -> i32;
    hipFreeAsync: fn(ptr: *mut c_void, stream: HipStream) -> i32;
    hipMallocFromPoolAsync: fn(ptr: *mut *mut c_void, size: usize, pool: HipMemoryPool, stream: HipStream) -> i32;
    hipMallocManaged: fn(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    hipHostAlloc: fn(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    hipHostFree: fn(ptr: *mut c_void) -> i32;
    hipMemGetInfo: fn(free: *mut usize, total: *mut usize) -> i32;
    hipMemcpy: fn(dst: *mut c_void, src: *const c_void, sizeBytes: usize, kind: i32) -> i32;
    hipMemcpyAsync: fn(dst: *mut c_void, src: *const c_void, sizeBytes: usize, kind: i32, stream: HipStream) -> i32;
    hipMemsetD8: fn(dst: *mut c_void, value: u8, count: usize) -> i32;
    hipMemsetD8Async: fn(dst: *mut c_void, value: u8, count: usize, stream: HipStream) -> i32;
    hipMemPrefetchAsync: fn(dst: *const c_void, count: usize, dstDevice: i32, flags: u32, stream: HipStream) -> i32;
    hipMemAdvise: fn(devPtr: *const c_void, count: usize, advice: i32, device: i32) -> i32;
    hipMemAddressReserve: fn(ptr: *mut *mut c_void, size: usize, alignment: usize, addr: *mut c_void, flags: u64) -> i32;
    hipMemAddressFree: fn(ptr: *mut c_void, size: usize) -> i32;
    hipMemCreate: fn(handle: *mut HipMemGenericAllocationHandle, size: usize, prop: *const HipMemAllocationProp, flags: u64) -> i32;
    hipMemMap: fn(ptr: *mut c_void, size: usize, offset: usize, handle: HipMemGenericAllocationHandle, flags: u64) -> i32;
    hipMemUnmap: fn(ptr: *mut c_void, size: usize) -> i32;
    hipMemRelease: fn(handle: HipMemGenericAllocationHandle) -> i32;
    hipMemSetAccess: fn(ptr: *mut c_void, size: usize, desc: *const HipMemAccessDesc, count: usize) -> i32;
    hipMemGetAllocationGranularity: fn(granularity: *mut usize, prop: *const HipMemAllocationProp, option: i32) -> i32;
    hipMemPoolCreate: fn(pool: *mut HipMemoryPool, props: *const HipMemPoolProps) -> i32;
    hipMemPoolDestroy: fn(pool: HipMemoryPool) -> i32;
    hipMemPoolSetAttribute: fn(pool: HipMemoryPool, attr: i32, value: *mut c_void) -> i32;
    hipMemPoolGetAttribute: fn(pool: HipMemoryPool, attr: i32, value: *mut c_void) -> i32;
    hipModuleLoad: fn(module: *mut HipModule, fname: *const c_char) -> i32;
    hipModuleLoadData: fn(module: *mut HipModule, image: *const c_void) -> i32;
    hipModuleUnload: fn(module: HipModule) -> i32;
    hipModuleGetFunction: fn(hfunc: *mut HipFunction, hmod: HipModule, name: *const c_char) -> i32;
    hipModuleGetGlobal: fn(dptr: *mut *mut c_void, bytes: *mut usize, hmod: HipModule, name: *const c_char) -> i32;
    hipFuncSetAttribute: fn(hfunc: HipFunction, attr: i32, value: i32) -> i32;
    hipFuncGetAttribute: fn(value: *mut i32, attr: i32, hfunc: HipFunction) -> i32;
    hipFuncSetCacheConfig: fn(hfunc: HipFunction, config: i32) -> i32;
    hipModuleLaunchKernel: fn(
        f: HipFunction,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32,
        stream: HipStream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    hipOccupancyMaxActiveBlocksPerMultiprocessor: fn(
        numBlocks: *mut i32,
        f: HipFunction,
        blockSize: i32,
        dynamicSMemSize: usize,
    ) -> i32;
    }
}

/// Load the HIP runtime and resolve every symbol table entry.
///
/// `CUDA_OXIDE_HIP_LIBRARY` overrides the candidate list with a single
/// explicit path or soname (host/container soname differences).
///
/// The `Library` handle is intentionally leaked: the function pointers outlive
/// every caller and the runtime must stay mapped for the process lifetime
/// (same bargain as the upstream OnceLock-cached API struct).
pub(crate) unsafe fn load_api() -> Result<HipApi, DynLoadError> {
    static OVERRIDE: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    let override_name: Option<&'static str> = *OVERRIDE.get_or_init(|| {
        std::env::var("CUDA_OXIDE_HIP_LIBRARY")
            .ok()
            .filter(|p| !p.is_empty())
            .map(|p| Box::leak(p.into_boxed_str()) as &'static str)
    });
    let candidates: &'static [&'static str] = match override_name {
        // Leak the one-element list: DynLoadError.names needs 'static.
        Some(path) => Box::leak(Box::new([path])),
        None => HIP_LIB_NAMES,
    };
    let mut last_error: Option<libloading::Error> = None;
    for &name in candidates {
        match unsafe { Library::new(name) } {
            Ok(lib) => {
                let api = unsafe { HipApi::from_library(&lib) };
                std::mem::forget(lib);
                return Ok(api);
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(DynLoadError::LoadFailed {
        names: candidates,
        source: last_error.expect("HIP candidate list is non-empty"),
    })
}

/// Loader failure, shaped like the upstream cuda-bindings `DynLoadError` so
/// `cuda-core`'s error formatting keeps working.
#[derive(Debug)]
pub enum DynLoadError {
    LoadFailed {
        names: &'static [&'static str],
        source: libloading::Error,
    },
    RuntimeTooOld {
        compile_version: u32,
        runtime_version: u32,
    },
}

impl std::fmt::Display for DynLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynLoadError::LoadFailed { names, source } => {
                write!(f, "failed to load any of {names:?}: {source}")
            }
            DynLoadError::RuntimeTooOld {
                compile_version,
                runtime_version,
            } => write!(
                f,
                "HIP runtime too old: built against {}.{} but runtime is {}.{}",
                compile_version / 1000,
                (compile_version % 1000) / 10,
                runtime_version / 1000,
                (runtime_version % 1000) / 10,
            ),
        }
    }
}

impl std::error::Error for DynLoadError {}

// ---------------------------------------------------------------------------
// hipRAND (cuRAND-compatible RNG layer). ROCm ships the generator API as
// `hiprand*` inside libhiprand; the shim maps cuRAND-named calls onto it.
// ---------------------------------------------------------------------------

pub type HipRandGenerator = *mut Opaque;

hip_api! {
    HipRandApi {
    hiprandCreateGenerator: fn(generator: *mut HipRandGenerator, rng_type: i32) -> i32;
    hiprandDestroyGenerator: fn(generator: HipRandGenerator) -> i32;
    hiprandSetStream: fn(generator: HipRandGenerator, stream: HipStream) -> i32;
    hiprandSetPseudoRandomGeneratorSeed: fn(generator: HipRandGenerator, seed: u64) -> i32;
    hiprandGenerateUniform: fn(generator: HipRandGenerator, output: *mut f32, num: usize) -> i32;
    hiprandGenerateUniformDouble: fn(generator: HipRandGenerator, output: *mut f64, num: usize) -> i32;
    hiprandGenerateNormal: fn(generator: HipRandGenerator, output: *mut f32, num: usize, mean: f32, stddev: f32) -> i32;
    hiprandGenerateNormalDouble: fn(generator: HipRandGenerator, output: *mut f64, num: usize, mean: f64, stddev: f64) -> i32;
    }
}

pub const HIPRAND_LIB_NAMES: &[&str] = &["libhiprand.so.1", "libhiprand.so"];

/// `curandStatus_t`/`hiprandStatus_t` value for "not initialized": what a
/// caller sees if the RNG library never loaded (any nonzero fails loudly).
pub const CURAND_STATUS_NOT_INITIALIZED: i32 = 105;

static HIPRAND_API: std::sync::OnceLock<Result<HipRandApi, DynLoadError>> =
    std::sync::OnceLock::new();

pub(crate) fn init_hiprand() -> &'static Result<HipRandApi, DynLoadError> {
    HIPRAND_API.get_or_init(|| unsafe { load_hiprand_api() })
}

unsafe fn load_hiprand_api() -> Result<HipRandApi, DynLoadError> {
    let mut last_error: Option<libloading::Error> = None;
    for &name in HIPRAND_LIB_NAMES {
        match unsafe { Library::new(name) } {
            Ok(lib) => {
                let api = unsafe { HipRandApi::from_library(&lib) };
                std::mem::forget(lib);
                return Ok(api);
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(DynLoadError::LoadFailed {
        names: HIPRAND_LIB_NAMES,
        source: last_error.expect("hipRAND candidate list is non-empty"),
    })
}
