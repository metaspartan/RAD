use std::{env, path::{Path, PathBuf}};

fn main() {
    let cuda = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search=native={}/lib64", cuda);
    println!("cargo:rustc-link-search=native={}/targets/sbsa-linux/lib", cuda);
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest.join("../../kernels/cuda");
    let so_path = kernel_dir.join("librad_cuda_ops.so");
    if Path::new(&so_path).exists() {
        println!("cargo:rustc-link-search=native={}", kernel_dir.display());
        println!("cargo:rustc-link-lib=dylib=rad_cuda_ops");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", kernel_dir.display());
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(
        out.join("cuda_ffi.rs"),
        r#"
pub type CudaError = i32;
pub const CUDA_SUCCESS: i32 = 0;
pub type CudaStream = *mut core::ffi::c_void;
pub type CudaEvent = *mut core::ffi::c_void;

extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut core::ffi::c_void, size: usize) -> CudaError;
    pub fn cudaMallocManaged(ptr: *mut *mut core::ffi::c_void, size: usize, flags: u32) -> CudaError;
    pub fn cudaFree(ptr: *mut core::ffi::c_void) -> CudaError;
    pub fn cudaHostAlloc(ptr: *mut *mut core::ffi::c_void, size: usize, flags: u32) -> CudaError;
    pub fn cudaFreeHost(ptr: *mut core::ffi::c_void) -> CudaError;
    pub fn cudaMemcpy(dst: *mut core::ffi::c_void, src: *const core::ffi::c_void, size: usize, kind: i32) -> CudaError;
    pub fn cudaMemcpyAsync(dst: *mut core::ffi::c_void, src: *const core::ffi::c_void, size: usize, kind: i32, stream: CudaStream) -> CudaError;
    pub fn cudaMemsetAsync(dst: *mut core::ffi::c_void, value: i32, count: usize, stream: CudaStream) -> CudaError;
    pub fn cudaStreamCreateWithFlags(s: *mut CudaStream, flags: u32) -> CudaError;
    pub fn cudaStreamDestroy(s: CudaStream) -> CudaError;
    pub fn cudaStreamSynchronize(s: CudaStream) -> CudaError;
    pub fn cudaDeviceSynchronize() -> CudaError;
    pub fn cudaEventCreateWithFlags(ev: *mut CudaEvent, flags: u32) -> CudaError;
    pub fn cudaEventDestroy(ev: CudaEvent) -> CudaError;
    pub fn cudaEventRecord(ev: CudaEvent, s: CudaStream) -> CudaError;
    pub fn cudaStreamWaitEvent(s: CudaStream, ev: CudaEvent, flags: u32) -> CudaError;
    pub fn cudaStreamBeginCapture(s: CudaStream, mode: i32) -> CudaError;
    pub fn cudaStreamEndCapture(s: CudaStream, g: *mut *mut core::ffi::c_void) -> CudaError;
    pub fn cudaGraphInstantiateWithFlags(e: *mut *mut core::ffi::c_void, g: *mut core::ffi::c_void, flags: u64) -> CudaError;
    pub fn cudaGraphDestroy(g: *mut core::ffi::c_void) -> CudaError;
    pub fn cudaGraphExecDestroy(e: *mut core::ffi::c_void) -> CudaError;
    pub fn cudaGraphLaunch(e: *mut core::ffi::c_void, s: CudaStream) -> CudaError;
    pub fn cudaSetDevice(d: i32) -> CudaError;
    pub fn cudaGetDevice(d: *mut i32) -> CudaError;
    pub fn cudaGetDeviceCount(count: *mut i32) -> CudaError;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;
    pub fn cudaGetLastError() -> CudaError;
}

pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
pub const CUDA_STREAM_NON_BLOCKING: u32 = 1;
pub const CUDA_EVENT_DISABLE_TIMING: u32 = 2;
pub const CUDA_MEM_ATTACH_GLOBAL: u32 = 1;
pub const CUDA_HOST_ALLOC_MAPPED: u32 = 2;
pub const CUDA_STREAM_CAPTURE_MODE_GLOBAL: i32 = 0;
"#,
    )
    .unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../kernels/cuda/librad_cuda_ops.so");
}
