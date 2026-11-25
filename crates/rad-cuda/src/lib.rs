#![allow(non_camel_case_types)]

pub mod comm;

include!(concat!(env!("OUT_DIR"), "/cuda_ffi.rs"));

#[derive(Debug)]
pub struct CudaFail(pub CudaError);

impl std::fmt::Display for CudaFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda error rc={}", self.0)
    }
}

impl std::error::Error for CudaFail {}

impl From<CudaError> for CudaFail {
    fn from(e: CudaError) -> Self {
        CudaFail(e)
    }
}

pub struct CudaBuf {
    pub ptr: *mut core::ffi::c_void,
    pub len: usize,
}
unsafe impl Send for CudaBuf {}
unsafe impl Sync for CudaBuf {}

impl CudaBuf {
    pub fn new(bytes: usize) -> Result<Self, CudaError> {
        let alloc_bytes = bytes.max(16);
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr, alloc_bytes) };
        if rc != CUDA_SUCCESS {
            return Err(rc);
        }
        Ok(Self { ptr, len: alloc_bytes })
    }

    pub fn new_managed(bytes: usize) -> Result<Self, CudaError> {
        let alloc_bytes = bytes.max(16);
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { cudaMallocManaged(&mut ptr, alloc_bytes, CUDA_MEM_ATTACH_GLOBAL) };
        if rc != CUDA_SUCCESS {
            return Err(rc);
        }
        Ok(Self { ptr, len: alloc_bytes })
    }

    pub fn copy_from(&mut self, host: &[u8], stream: Option<CudaStream>) -> Result<(), CudaError> {
        assert!(host.len() <= self.len);
        let rc = match stream {
            Some(s) => unsafe {
                cudaMemcpyAsync(self.ptr, host.as_ptr() as _, host.len(), CUDA_MEMCPY_HOST_TO_DEVICE, s)
            },
            None => unsafe {
                cudaMemcpy(self.ptr, host.as_ptr() as _, host.len(), CUDA_MEMCPY_HOST_TO_DEVICE)
            },
        };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub fn copy_to_host(&self, out: &mut [u8]) -> Result<(), CudaError> {
        assert!(out.len() <= self.len);
        let rc = unsafe {
            cudaMemcpy(out.as_mut_ptr() as _, self.ptr, out.len(), CUDA_MEMCPY_DEVICE_TO_HOST)
        };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub fn zero_async(&mut self, stream: CudaStream) -> Result<(), CudaError> {
        let rc = unsafe { cudaMemsetAsync(self.ptr, 0, self.len, stream) };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }
}

impl Drop for CudaBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr) };
        }
    }
}

pub struct CudaPinnedBuf {
    pub ptr: *mut core::ffi::c_void,
    pub len: usize,
}
unsafe impl Send for CudaPinnedBuf {}
unsafe impl Sync for CudaPinnedBuf {}

impl CudaPinnedBuf {
    pub fn new(bytes: usize) -> Result<Self, CudaError> {
        let alloc_bytes = bytes.max(16);
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { cudaHostAlloc(&mut ptr, alloc_bytes, CUDA_HOST_ALLOC_MAPPED) };
        if rc != CUDA_SUCCESS {
            return Err(rc);
        }
        Ok(Self { ptr, len: alloc_bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.len / 4) }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.len / 4) }
    }
}

impl Drop for CudaPinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFreeHost(self.ptr) };
        }
    }
}

pub struct CudaCapturedGraph {
    pub exec: *mut core::ffi::c_void,
    pub stream: CudaStream,
}
unsafe impl Send for CudaCapturedGraph {}

impl CudaCapturedGraph {
    pub unsafe fn capture(f: impl FnOnce(CudaStream)) -> Result<Self, CudaError> {
        let mut s = std::ptr::null_mut();
        let rc = cudaStreamCreateWithFlags(&mut s, CUDA_STREAM_NON_BLOCKING);
        if rc != CUDA_SUCCESS {
            return Err(rc);
        }
        let rc = cudaStreamBeginCapture(s, CUDA_STREAM_CAPTURE_MODE_GLOBAL);
        if rc != CUDA_SUCCESS {
            cudaStreamDestroy(s);
            return Err(rc);
        }
        f(s);
        let mut g = std::ptr::null_mut();
        let rc = cudaStreamEndCapture(s, &mut g);
        if rc != CUDA_SUCCESS {
            cudaStreamDestroy(s);
            return Err(rc);
        }
        let mut exec = std::ptr::null_mut();
        let rc = cudaGraphInstantiateWithFlags(&mut exec, g, 0);
        cudaGraphDestroy(g);
        if rc != CUDA_SUCCESS {
            cudaStreamDestroy(s);
            return Err(rc);
        }
        Ok(Self { exec, stream: s })
    }

    pub fn launch(&self) -> Result<(), CudaError> {
        let rc = unsafe { cudaGraphLaunch(self.exec, self.stream) };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub fn sync(&self) -> Result<(), CudaError> {
        let rc = unsafe { cudaStreamSynchronize(self.stream) };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }
}

impl Drop for CudaCapturedGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                cudaGraphExecDestroy(self.exec);
            }
            if !self.stream.is_null() {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

pub fn device_count() -> Result<i32, CudaError> {
    let mut n = 0i32;
    let rc = unsafe { cudaGetDeviceCount(&mut n) };
    if rc == CUDA_SUCCESS { Ok(n) } else { Err(rc) }
}

pub fn set_device(dev: i32) -> Result<(), CudaError> {
    let rc = unsafe { cudaSetDevice(dev) };
    if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
}

pub fn sync_device() -> Result<(), CudaError> {
    let rc = unsafe { cudaDeviceSynchronize() };
    if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
}

pub fn mem_info() -> Result<(usize, usize), CudaError> {
    let mut free = 0usize;
    let mut total = 0usize;
    let rc = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if rc == CUDA_SUCCESS { Ok((free, total)) } else { Err(rc) }
}
