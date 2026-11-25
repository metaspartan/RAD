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
