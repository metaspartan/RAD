use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

pub struct Tp2Comm {
    pub rank: usize,
    stream: Option<TcpStream>,
    peer_buf: Vec<f32>,
}

impl Tp2Comm {
    pub fn single() -> Self {
        Self {
            rank: 0,
            stream: None,
            peer_buf: Vec::new(),
        }
    }

    pub fn connect(rank: usize, addr: &str) -> std::io::Result<Self> {
        let s = if rank == 0 {
            let listener = TcpListener::bind(addr)?;
            let (conn, _) = listener.accept()?;
            conn
        } else {
            let mut last_err = None;
            let mut connected = None;
            for _ in 0..200 {
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
        Ok(Self {
            rank,
            stream: Some(s),
            peer_buf: Vec::with_capacity(65536),
        })
    }

    pub fn world_size(&self) -> usize {
        if self.stream.is_some() { 2 } else { 1 }
    }

    pub fn all_reduce_sum_f32(&mut self, data: &mut [f32]) -> std::io::Result<()> {
        let Some(ref mut s) = self.stream else {
            return Ok(());
        };
        let n = data.len();
        if self.peer_buf.len() < n {
            self.peer_buf.resize(n, 0.0);
        }
        let send_bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, n * 4) };
        let recv_bytes = unsafe { std::slice::from_raw_parts_mut(self.peer_buf.as_mut_ptr() as *mut u8, n * 4) };
        if self.rank == 0 {
            s.write_all(send_bytes)?;
            s.read_exact(recv_bytes)?;
        } else {
            s.read_exact(recv_bytes)?;
            s.write_all(send_bytes)?;
        }
        for i in 0..n {
            data[i] += self.peer_buf[i];
        }
        Ok(())
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
