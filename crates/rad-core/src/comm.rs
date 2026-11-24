use crate::ops;
use anyhow::Result;
use rad_hip::{DevBuf, HipError, HipStream, PinnedBuf};

fn he(e: HipError) -> anyhow::Error {
    anyhow::anyhow!("hip error rc={e}")
}

pub struct GpuDeviceInfo {
    pub id: i32,
    pub free_bytes: usize,
    pub total_bytes: usize,
}

pub struct GpuTopology {
    pub devices: Vec<GpuDeviceInfo>,
    pub p2p: Vec<Vec<bool>>,
}

impl GpuTopology {
    pub fn probe(requested: Option<&[i32]>) -> Result<Self> {
        let count = rad_hip::device_count().map_err(he)?;
        let ids: Vec<i32> = match requested {
            Some(r) if !r.is_empty() => r.to_vec(),
            _ => (0..count).collect(),
        };
        let mut devices = Vec::with_capacity(ids.len());
        for &id in &ids {
            if id < 0 || id >= count {
                anyhow::bail!("GPU id {id} out of range (detected {count} HIP devices)");
            }
            let (free_bytes, total_bytes) = rad_hip::mem_info(id).unwrap_or((0, 0));
            devices.push(GpuDeviceInfo { id, free_bytes, total_bytes });
        }
        let p2p = rad_hip::enable_p2p_matrix(&ids);
        Ok(Self { devices, p2p })
    }

    pub fn p2p_pairs(&self) -> usize {
        let n = self.p2p.len();
        let mut c = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.p2p[i][j] && self.p2p[j][i] {
                    c += 1;
                }
            }
        }
        c
    }
}

pub struct MultiGpuComm {
    pub devices: Vec<i32>,
    pub p2p: Vec<Vec<bool>>,
    stage_dev: Vec<DevBuf>,
    stage_host: Vec<PinnedBuf>,
    cap_elems: usize,
}

impl MultiGpuComm {
    pub fn new(devices: Vec<i32>, max_elems: usize) -> Result<Self> {
        let p2p = rad_hip::enable_p2p_matrix(&devices);
        let bytes = max_elems.max(1) * 4;
        let prev = rad_hip::get_device().unwrap_or(0);
        let mut stage_dev = Vec::with_capacity(devices.len());
        let mut stage_host = Vec::with_capacity(devices.len());
        for &d in &devices {
            rad_hip::set_device(d).map_err(he)?;
            stage_dev.push(DevBuf::new(bytes).map_err(he)?);
            stage_host.push(PinnedBuf::new(bytes).map_err(he)?);
        }
        let _ = rad_hip::set_device(prev);
        Ok(Self {
            devices,
            p2p,
            stage_dev,
            stage_host,
            cap_elems: max_elems.max(1),
        })
    }

    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    pub fn all_p2p(&self) -> bool {
        let n = self.devices.len();
        for i in 0..n {
            for j in 0..n {
                if i != j && !self.p2p[i][j] {
                    return false;
                }
            }
        }
        n > 1
    }

    pub unsafe fn allreduce_sum_f32(&mut self, bufs: &[*mut f32], streams: &[HipStream], n: usize) -> Result<()> {
        let w = self.devices.len();
        if w <= 1 || n == 0 {
            return Ok(());
        }
        assert!(n <= self.cap_elems);
        assert_eq!(bufs.len(), w);
        assert_eq!(streams.len(), w);
        let bytes = n * 4;
        let prev = rad_hip::get_device().unwrap_or(0);

        if w == 2 && self.p2p[0][1] && self.p2p[1][0] {
            rad_hip::set_device(self.devices[0]).map_err(he)?;
            let rc0 = rad_hip::hipMemcpyPeerAsync(
                self.stage_dev[0].ptr,
                self.devices[0],
                bufs[1] as *const core::ffi::c_void,
                self.devices[1],
                bytes,
                streams[0],
            );
            if rc0 != rad_hip::HIP_SUCCESS {
                let _ = rad_hip::set_device(prev);
                return Err(he(rc0));
            }
            rad_hip::set_device(self.devices[1]).map_err(he)?;
            let rc1 = rad_hip::hipMemcpyPeerAsync(
                self.stage_dev[1].ptr,
                self.devices[1],
                bufs[0] as *const core::ffi::c_void,
                self.devices[0],
                bytes,
                streams[1],
            );
            if rc1 != rad_hip::HIP_SUCCESS {
                let _ = rad_hip::set_device(prev);
                return Err(he(rc1));
            }
            rad_hip::set_device(self.devices[0]).map_err(he)?;
            ops::ink_vec_add(streams[0], bufs[0] as *const f32, self.stage_dev[0].ptr as *const f32, bufs[0], n as i32);
            rad_hip::set_device(self.devices[1]).map_err(he)?;
            ops::ink_vec_add(streams[1], bufs[1] as *const f32, self.stage_dev[1].ptr as *const f32, bufs[1], n as i32);
            let _ = rad_hip::set_device(prev);
            return Ok(());
        }

        for r in 0..w {
            rad_hip::set_device(self.devices[r]).map_err(he)?;
            let rc = rad_hip::hipMemcpyAsync(
                self.stage_host[r].ptr,
                bufs[r] as *const core::ffi::c_void,
                bytes,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
                streams[r],
            );
            if rc != rad_hip::HIP_SUCCESS {
                let _ = rad_hip::set_device(prev);
                return Err(he(rc));
            }
        }
        for r in 0..w {
            rad_hip::set_device(self.devices[r]).map_err(he)?;
            let rc = rad_hip::hipStreamSynchronize(streams[r]);
            if rc != rad_hip::HIP_SUCCESS {
                let _ = rad_hip::set_device(prev);
                return Err(he(rc));
            }
        }
        let (first, rest) = self.stage_host.split_at_mut(1);
        let acc = &mut first[0].as_f32_slice_mut()[..n];
        for hbuf in rest {
            let src = &hbuf.as_f32_slice()[..n];
            for i in 0..n {
                acc[i] += src[i];
            }
        }
        let sum_ptr = first[0].ptr as *const core::ffi::c_void;
        for r in 0..w {
            rad_hip::set_device(self.devices[r]).map_err(he)?;
            let rc = rad_hip::hipMemcpyAsync(
                bufs[r] as *mut core::ffi::c_void,
                sum_ptr,
                bytes,
                rad_hip::HIP_MEMCPY_HOST_TO_DEVICE,
                streams[r],
            );
            if rc != rad_hip::HIP_SUCCESS {
                let _ = rad_hip::set_device(prev);
                return Err(he(rc));
            }
        }
        let _ = rad_hip::set_device(prev);
        Ok(())
    }

    pub unsafe fn broadcast_f32(&mut self, src_rank: usize, bufs: &[*mut f32], streams: &[HipStream], n: usize) -> Result<()> {
        let w = self.devices.len();
        if w <= 1 || n == 0 {
            return Ok(());
        }
        assert!(n <= self.cap_elems);
        let bytes = n * 4;
        let prev = rad_hip::get_device().unwrap_or(0);
        let mut need_host = false;
        for r in 0..w {
            if r != src_rank && !self.p2p[r][src_rank] {
                need_host = true;
                break;
            }
        }
        if !need_host {
            for r in 0..w {
                if r == src_rank {
                    continue;
                }
                rad_hip::set_device(self.devices[r]).map_err(he)?;
                let rc = rad_hip::hipMemcpyPeerAsync(
                    bufs[r] as *mut core::ffi::c_void,
                    self.devices[r],
                    bufs[src_rank] as *const core::ffi::c_void,
                    self.devices[src_rank],
                    bytes,
                    streams[r],
                );
                if rc != rad_hip::HIP_SUCCESS {
                    let _ = rad_hip::set_device(prev);
                    return Err(he(rc));
                }
            }
        } else {
            rad_hip::set_device(self.devices[src_rank]).map_err(he)?;
            let rc = rad_hip::hipMemcpyAsync(
                self.stage_host[0].ptr,
                bufs[src_rank] as *const core::ffi::c_void,
                bytes,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
                streams[src_rank],
            );
            if rc != rad_hip::HIP_SUCCESS {
                let _ = rad_hip::set_device(prev);
                return Err(he(rc));
            }
            let _ = rad_hip::hipStreamSynchronize(streams[src_rank]);
            let src_ptr = self.stage_host[0].ptr as *const core::ffi::c_void;
            for r in 0..w {
                if r == src_rank {
                    continue;
                }
                rad_hip::set_device(self.devices[r]).map_err(he)?;
                let rc = rad_hip::hipMemcpyAsync(
                    bufs[r] as *mut core::ffi::c_void,
                    src_ptr,
                    bytes,
                    rad_hip::HIP_MEMCPY_HOST_TO_DEVICE,
                    streams[r],
                );
                if rc != rad_hip::HIP_SUCCESS {
                    let _ = rad_hip::set_device(prev);
                    return Err(he(rc));
                }
            }
        }
        let _ = rad_hip::set_device(prev);
        Ok(())
    }
}
