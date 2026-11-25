use core::ffi::{c_char, c_void};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RadRdmaInfo {
    qpn: u32,
    psn: u32,
    rkey: u32,
    addr: u64,
    gid: [u8; 16],
}

extern "C" {
    fn rad_rdma_init(dev_name: *const c_char, buf_bytes: usize, local_info: *mut RadRdmaInfo) -> *mut c_void;
    fn rad_rdma_connect(handle: *mut c_void, remote_info: *const RadRdmaInfo) -> i32;
    fn rad_rdma_send_dptr(handle: *mut c_void) -> *mut c_void;
    fn rad_rdma_recv_dptr(handle: *mut c_void) -> *mut c_void;
    fn rad_rdma_exchange_mapped(handle: *mut c_void, n_bytes: i32) -> i32;
    fn rad_rdma_close(handle: *mut c_void);
}

pub struct Tp2Comm {
    pub rank: usize,
    stream: Option<TcpStream>,
    peer_buf: Vec<f32>,
    rdma: *mut c_void,
}

unsafe impl Send for Tp2Comm {}

impl Drop for Tp2Comm {
    fn drop(&mut self) {
        if !self.rdma.is_null() {
            unsafe { rad_rdma_close(self.rdma) };
            self.rdma = std::ptr::null_mut();
        }
    }
}

impl Tp2Comm {
    pub fn single() -> Self {
        Self {
            rank: 0,
            stream: None,
            peer_buf: Vec::new(),
            rdma: std::ptr::null_mut(),
        }
    }

    pub fn connect(rank: usize, addr: &str) -> std::io::Result<Self> {
        let mut s = if rank == 0 {
            let listener = TcpListener::bind(addr)?;
            let (conn, _) = listener.accept()?;
            conn
        } else {
            let mut last_err = None;
            let mut connected = None;
            for _ in 0..400 {
                match TcpStream::connect(addr) {
                    Ok(c) => {
                        connected = Some(c);
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            }
            match connected {
                Some(c) => c,
                None => return Err(last_err.unwrap()),
            }
        };
        s.set_nodelay(true)?;

        let mut rdma_handle: *mut c_void = std::ptr::null_mut();
        let dev = b"rocep1s0f1\0";
        let mut local_info = RadRdmaInfo::default();
        let h = unsafe { rad_rdma_init(dev.as_ptr() as *const c_char, 2 * 1024 * 1024, &mut local_info) };
        let local_ok: u8 = if h.is_null() { 0 } else { 1 };
        let mut peer_ok = [0u8; 1];
        if rank == 0 {
            s.write_all(&[local_ok])?;
            s.read_exact(&mut peer_ok)?;
        } else {
            s.read_exact(&mut peer_ok)?;
            s.write_all(&[local_ok])?;
        }
        if local_ok == 1 && peer_ok[0] == 1 {
            let info_sz = std::mem::size_of::<RadRdmaInfo>();
            let local_bytes = unsafe { std::slice::from_raw_parts(&local_info as *const RadRdmaInfo as *const u8, info_sz) };
            let mut remote_info = RadRdmaInfo::default();
            let remote_bytes = unsafe { std::slice::from_raw_parts_mut(&mut remote_info as *mut RadRdmaInfo as *mut u8, info_sz) };
            if rank == 0 {
                s.write_all(local_bytes)?;
                s.read_exact(remote_bytes)?;
            } else {
                s.read_exact(remote_bytes)?;
                s.write_all(local_bytes)?;
            }
            let rc = unsafe { rad_rdma_connect(h, &remote_info) };
            let ready = [if rc == 0 { 1u8 } else { 0u8 }];
            let mut peer_ready = [0u8; 1];
            if rank == 0 {
                s.write_all(&ready)?;
                s.read_exact(&mut peer_ready)?;
            } else {
                s.read_exact(&mut peer_ready)?;
                s.write_all(&ready)?;
            }
            if ready[0] == 1 && peer_ready[0] == 1 {
                rdma_handle = h;
                if rank == 0 {
                    println!("[rdma] ConnectX-7 RoCE v2 connected (rocep1s0f1)");
                }
            } else {
                unsafe { rad_rdma_close(h) };
            }
        } else if !h.is_null() {
            unsafe { rad_rdma_close(h) };
        }

        Ok(Self {
            rank,
            stream: Some(s),
            peer_buf: Vec::with_capacity(65536),
            rdma: rdma_handle,
        })
    }

    pub fn world_size(&self) -> usize {
        if self.stream.is_some() { 2 } else { 1 }
    }

    pub fn set_nodelay(&self, nodelay: bool) {
        if let Some(ref s) = self.stream {
            let _ = s.set_nodelay(nodelay);
        }
    }

    pub fn send_bytes(&mut self, data: &[u8]) -> std::io::Result<()> {
        let Some(ref mut s) = self.stream else {
            return Ok(());
        };
        let len = (data.len() as u64).to_le_bytes();
        s.write_all(&len)?;
        s.write_all(data)?;
        Ok(())
    }

    pub fn recv_bytes(&mut self) -> std::io::Result<Vec<u8>> {
        let Some(ref mut s) = self.stream else {
            return Ok(Vec::new());
        };
        let mut lbuf = [0u8; 8];
        s.read_exact(&mut lbuf)?;
        let len = u64::from_le_bytes(lbuf) as usize;
        let mut out = vec![0u8; len];
        s.read_exact(&mut out)?;
        Ok(out)
    }

    pub fn send_u64_slice(&mut self, vals: &[u64]) -> std::io::Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 8) };
        self.send_bytes(bytes)
    }

    pub fn recv_u64_slice(&mut self) -> std::io::Result<Vec<u64>> {
        let bytes = self.recv_bytes()?;
        let n = bytes.len() / 8;
        let mut out = vec![0u64; n];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, n * 8);
        }
        Ok(out)
    }

    pub fn rdma_ptrs(&self) -> Option<(*mut c_void, *mut c_void)> {
        if self.rdma.is_null() {
            return None;
        }
        let s = unsafe { rad_rdma_send_dptr(self.rdma) };
        let r = unsafe { rad_rdma_recv_dptr(self.rdma) };
        if s.is_null() || r.is_null() {
            None
        } else {
            Some((s, r))
        }
    }

    pub fn rdma_exchange_mapped(&mut self, n_bytes: usize) -> std::io::Result<bool> {
        if self.rdma.is_null() {
            return Ok(false);
        }
        let rc = unsafe { rad_rdma_exchange_mapped(self.rdma, n_bytes as i32) };
        if rc == 0 {
            Ok(true)
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::Other, format!("rdma exchange failed: {rc}")))
        }
    }

    pub fn all_reduce_sum_f16(&mut self, data: &mut [u16]) -> std::io::Result<()> {
        let Some(ref mut s) = self.stream else {
            return Ok(());
        };
        let n = data.len();
        let n_bytes = n * 2;
        if self.peer_buf.len() * 4 < n_bytes {
            self.peer_buf.resize((n_bytes + 3) / 4, 0.0);
        }
        let send_bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, n_bytes) };
        let recv_bytes = unsafe { std::slice::from_raw_parts_mut(self.peer_buf.as_mut_ptr() as *mut u8, n_bytes) };
        if self.rank == 0 {
            s.write_all(send_bytes)?;
            s.read_exact(recv_bytes)?;
        } else {
            s.read_exact(recv_bytes)?;
            s.write_all(send_bytes)?;
        }
        let peer_u16 = unsafe { std::slice::from_raw_parts(self.peer_buf.as_ptr() as *const u16, n) };
        for i in 0..n {
            data[i] = f32_to_f16_fast(f16_to_f32_fast(data[i]) + f16_to_f32_fast(peer_u16[i]));
        }
        Ok(())
    }

    pub fn all_reduce_argmax(&mut self, vals: &mut [f32], ids: &mut [i32]) -> std::io::Result<()> {
        let Some(ref mut s) = self.stream else {
            return Ok(());
        };
        let m = vals.len();
        let mut send_buf = vec![0u8; m * 8];
        let mut recv_buf = vec![0u8; m * 8];
        for i in 0..m {
            send_buf[i * 8..i * 8 + 4].copy_from_slice(&vals[i].to_le_bytes());
            send_buf[i * 8 + 4..i * 8 + 8].copy_from_slice(&ids[i].to_le_bytes());
        }
        if self.rank == 0 {
            s.write_all(&send_buf)?;
            s.read_exact(&mut recv_buf)?;
        } else {
            s.read_exact(&mut recv_buf)?;
            s.write_all(&send_buf)?;
        }
        for i in 0..m {
            let pv = f32::from_le_bytes(recv_buf[i * 8..i * 8 + 4].try_into().unwrap());
            let pi = i32::from_le_bytes(recv_buf[i * 8 + 4..i * 8 + 8].try_into().unwrap());
            if pv > vals[i] {
                vals[i] = pv;
                ids[i] = pi;
            }
        }
        Ok(())
    }

    pub fn broadcast_u32(&mut self, val: &mut u32) -> std::io::Result<()> {
        let Some(ref mut s) = self.stream else {
            return Ok(());
        };
        let mut buf = val.to_le_bytes();
        if self.rank == 0 {
            s.write_all(&buf)?;
        } else {
            s.read_exact(&mut buf)?;
            *val = u32::from_le_bytes(buf);
        }
        Ok(())
    }

    pub fn broadcast_i32_vec(&mut self, data: &mut Vec<i32>) -> std::io::Result<()> {
        let Some(ref mut s) = self.stream else {
            return Ok(());
        };
        if self.rank == 0 {
            let len = (data.len() as u32).to_le_bytes();
            s.write_all(&len)?;
            let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            s.write_all(bytes)?;
        } else {
            let mut lbuf = [0u8; 4];
            s.read_exact(&mut lbuf)?;
            let n = u32::from_le_bytes(lbuf) as usize;
            data.resize(n, 0);
            let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) };
            s.read_exact(bytes)?;
        }
        Ok(())
    }
}

fn f16_to_f32_fast(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1f) as u32;
    let f = (bits & 0x3ff) as u32;
    let out = if e == 0 {
        if f == 0 {
            s << 31
        } else {
            let mut m = f;
            let mut exp = 127 - 14;
            while (m & 0x400) == 0 {
                m <<= 1;
                exp -= 1;
            }
            (s << 31) | (exp << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | 0x7f800000 | (f << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (f << 13)
    };
    f32::from_bits(out)
}

fn f32_to_f16_fast(val: f32) -> u16 {
    let b = val.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let f = b & 0x7fffff;
    if e <= 0 {
        s
    } else if e >= 31 {
        s | 0x7c00
    } else {
        s | ((e as u16) << 10) | ((f >> 13) as u16)
    }
}
