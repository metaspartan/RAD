use std::{env, path::{Path, PathBuf}};

fn main() {
    let rocm = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    println!("cargo:rustc-link-search=native={}/lib", rocm);
    let hip_rt = if Path::new(&format!("{rocm}/lib/libhipamd.so")).exists() {
        "hipamd"
    } else {
        "amdhip64"
    };
    println!("cargo:rustc-link-lib=dylib={hip_rt}");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=rocblas");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(
        out.join("fallback.rs"),
        r#"
pub type HipError = i32;
pub const HIP_SUCCESS: i32 = 0;
pub type HipStream = *mut core::ffi::c_void;
pub type HipDevicePtr = u64;

extern "C" {
    pub fn hipMalloc(ptr: *mut *mut core::ffi::c_void, size: usize) -> HipError;
    pub fn hipFree(ptr: *mut core::ffi::c_void) -> HipError;
    pub fn hipMemcpy(dst: *mut core::ffi::c_void, src: *const core::ffi::c_void, size: usize, kind: i32) -> HipError;
    pub fn hipMemcpyAsync(dst: *mut core::ffi::c_void, src: *const core::ffi::c_void, size: usize, kind: i32, stream: HipStream) -> HipError;
    pub fn hipStreamCreate(s: *mut HipStream) -> HipError;
    pub fn hipStreamSynchronize(s: HipStream) -> HipError;
    pub fn hipMemsetD32Async(dst: *mut core::ffi::c_void, value: i32, count: usize, stream: HipStream) -> HipError;
    pub fn hipStreamBeginCapture(s: HipStream, mode: i32) -> HipError;
    pub fn hipStreamEndCapture(s: HipStream, g: *mut *mut core::ffi::c_void) -> HipError;
    pub fn hipGraphInstantiate(e: *mut *mut core::ffi::c_void, g: *mut core::ffi::c_void, flags: u64) -> HipError;
    pub fn hipGraphLaunch(e: *mut core::ffi::c_void, s: HipStream) -> HipError;
    pub fn hipSetDevice(d: i32) -> HipError;
    pub fn hipGetLastError() -> HipError;
}

pub const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: i32 = 0;
"#,
    )
    .unwrap();
    println!("cargo:rerun-if-changed=build.rs");
}
