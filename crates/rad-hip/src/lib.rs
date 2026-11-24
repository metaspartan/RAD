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

    pub fn launch_on(&self, s: HipStream) -> Result<(), HipError> {
        let rc = unsafe { hipGraphLaunch(self.exec, s) };
        if rc == HIP_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub fn sync(&self) -> Result<(), HipError> {
        let rc = unsafe { hipStreamSynchronize(self.stream) };
        if rc == HIP_SUCCESS { Ok(()) } else { Err(rc) }
    }
}

pub struct PinnedBuf {
    pub ptr: *mut core::ffi::c_void,
    pub len: usize,
}
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    pub fn new(bytes: usize) -> Result<Self, HipError> {
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { hipHostMalloc(&mut ptr, bytes, 0) };
        if rc != HIP_SUCCESS {
            return Err(rc);
        }
        Ok(Self { ptr, len: bytes })
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.len / 4) }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.len / 4) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        unsafe { hipHostFree(self.ptr) };
    }
}

pub fn device_count() -> Result<i32, HipError> {
    let mut n = 0i32;
    let rc = unsafe { hipGetDeviceCount(&mut n) };
    if rc == HIP_SUCCESS { Ok(n) } else { Err(rc) }
}

pub fn get_device() -> Result<i32, HipError> {
    let mut d = 0i32;
    let rc = unsafe { hipGetDevice(&mut d) };
    if rc == HIP_SUCCESS { Ok(d) } else { Err(rc) }
}

pub fn set_device(dev: i32) -> Result<(), HipError> {
    let rc = unsafe { hipSetDevice(dev) };
    if rc == HIP_SUCCESS { Ok(()) } else { Err(rc) }
}

pub fn mem_info(dev: i32) -> Result<(usize, usize), HipError> {
    let prev = get_device().unwrap_or(0);
    set_device(dev)?;
    let mut free = 0usize;
    let mut total = 0usize;
    let rc = unsafe { hipMemGetInfo(&mut free, &mut total) };
    let _ = set_device(prev);
    if rc == HIP_SUCCESS { Ok((free, total)) } else { Err(rc) }
}

pub fn can_access_peer(dev: i32, peer: i32) -> bool {
    if dev == peer {
        return true;
    }
    let mut can = 0i32;
    let rc = unsafe { hipDeviceCanAccessPeer(&mut can, dev, peer) };
    rc == HIP_SUCCESS && can != 0
}

pub fn enable_p2p_matrix(devs: &[i32]) -> Vec<Vec<bool>> {
    let n = devs.len();
    let prev = get_device().unwrap_or(0);
    let mut mat = vec![vec![false; n]; n];
    for i in 0..n {
        mat[i][i] = true;
        if set_device(devs[i]).is_err() {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            if can_access_peer(devs[i], devs[j]) {
                let rc = unsafe { hipDeviceEnablePeerAccess(devs[j], 0) };
                if rc == HIP_SUCCESS || rc == 704 {
                    mat[i][j] = true;
                }
            }
        }
    }
    let _ = set_device(prev);
    mat
}
