#![allow(non_camel_case_types)]

include!(concat!(env!("OUT_DIR"), "/fallback.rs"));

pub struct DevBuf {
    pub ptr: *mut core::ffi::c_void,
    pub len: usize,
}
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

#[derive(Debug)]
pub struct HipFail(pub HipError);
impl std::fmt::Display for HipFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hip error rc={}", self.0)
    }
}
impl std::error::Error for HipFail {}
impl From<HipError> for HipFail {
    fn from(e: HipError) -> Self { HipFail(e) }
}

impl DevBuf {
    pub fn new(bytes: usize) -> Result<Self, HipError> {
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { hipMalloc(&mut ptr, bytes) };
        if rc != HIP_SUCCESS {
            return Err(rc);
        }
        Ok(Self { ptr, len: bytes })
    }

    pub fn copy_from(&mut self, host: &[u8], stream: Option<HipStream>) -> Result<(), HipError> {
        assert!(host.len() <= self.len);
        let rc = match stream {
            Some(s) => unsafe {
                hipMemcpyAsync(self.ptr, host.as_ptr() as _, host.len(), HIP_MEMCPY_HOST_TO_DEVICE, s)
            },
            None => unsafe { hipMemcpy(self.ptr, host.as_ptr() as _, host.len(), HIP_MEMCPY_HOST_TO_DEVICE) },
        };
        if rc == HIP_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub fn zero(&mut self) -> Result<(), HipError> {
        let rc = unsafe { hipMemsetD32Async(self.ptr, 0, self.len / 4, std::ptr::null_mut()) };
        if rc == HIP_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub fn copy_to_host(&self, out: &mut [u8]) -> Result<(), HipError> {
        assert!(out.len() <= self.len);
        let rc = unsafe { hipMemcpy(out.as_mut_ptr() as _, self.ptr, out.len(), HIP_MEMCPY_DEVICE_TO_HOST) };
        if rc == HIP_SUCCESS { Ok(()) } else { Err(rc) }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { hipFree(self.ptr) };
    }
}

pub struct CapturedGraph {
    pub exec: *mut core::ffi::c_void,
    pub stream: HipStream,
}
unsafe impl Send for CapturedGraph {}

impl CapturedGraph {
    pub unsafe fn capture(f: impl FnOnce(HipStream)) -> Result<Self, HipError> {
        let mut s = std::ptr::null_mut();
        if hipStreamCreate(&mut s) != HIP_SUCCESS {
            return Err(hipGetLastError());
        }
        if hipStreamBeginCapture(s, HIP_STREAM_CAPTURE_MODE_GLOBAL) != HIP_SUCCESS {
            return Err(hipGetLastError());
        }
        f(s);
        let mut g = std::ptr::null_mut();
        if hipStreamEndCapture(s, &mut g) != HIP_SUCCESS {
            return Err(hipGetLastError());
        }
        let mut exec = std::ptr::null_mut();
        if hipGraphInstantiate(&mut exec, g, 0) != HIP_SUCCESS {
            return Err(hipGetLastError());
        }
        Ok(Self { exec, stream: s })
    }

    pub fn launch(&self) -> Result<(), HipError> {
        let rc = unsafe { hipGraphLaunch(self.exec, self.stream) };
        if rc == HIP_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub fn sync(&self) -> Result<(), HipError> {
        let rc = unsafe { hipStreamSynchronize(self.stream) };
        if rc == HIP_SUCCESS { Ok(()) } else { Err(rc) }
    }
}
