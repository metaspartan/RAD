use crate::client::{fetch_health, fetch_models, stream_chat, ChatMessage, StreamEvent, StreamParams};
use anyhow::Result;
use reqwest::Client;
use serde::Serialize;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Clone, Serialize)]
pub struct PlainLine {
    pub kind: &'static str,
    pub text: String,
}

#[derive(Serialize)]
struct PlainFrame {
    t_ms: u64,
    lines: Vec<PlainLine>,
}

pub struct PlainConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub display_name: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub enable_thinking: bool,
    pub reasoning_effort: String,
}

pub async fn run_plain_cli(client: &Client, mut cfg: PlainConfig) -> Result<()> {
    println!("\x1b[1mrad-chat\x1b[0m (CLI)  |  \x1b[36m{}\x1b[0m  |  {}", cfg.display_name, cfg.base_url);
    println!("\x1b[90mCommands: /clear  /model [name]  /url <http://...>  /think  /temp <f>  /exit\x1b[0m\n");

    let mut history: Vec<ChatMessage> = Vec::new();
    let stdin = io::stdin();
    let mut lines_iter = stdin.lock().lines();

    loop {
        print!("\x1b[32m> \x1b[0m");
        let _ = io::stdout().flush();
        let Some(Ok(raw)) = lines_iter.next() else {
            break;
        };
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('/') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts[0] {
                "/exit" | "/quit" => break,
                "/clear" => {
                    history.clear();
                    println!("\x1b[90m[cleared]\x1b[0m");
                }
                "/think" => {
                    cfg.enable_thinking = !cfg.enable_thinking;
                    println!("\x1b[90m[thinking = {}]\x1b[0m", if cfg.enable_thinking { "on" } else { "off" });
                }
                "/temp" => {
                    if let Some(v) = parts.get(1).and_then(|s| s.parse::<f32>().ok()) {
                        cfg.temperature = v.clamp(0.0, 2.0);
                        println!("\x1b[90m[temp = {:.2}]\x1b[0m", cfg.temperature);
                    }
                }
                "/model" => {
                    if let Some(m) = parts.get(1) {
                        cfg.model = (*m).to_string();
                        cfg.display_name = (*m).to_string();
                        println!("\x1b[90m[model = {}]\x1b[0m", cfg.model);
                    } else {
                        let ms = fetch_models(client, &cfg.base_url, cfg.api_key.as_deref()).await;
                        println!("\x1b[90m[models: {}]\x1b[0m", ms.join(", "));
                    }
                }
                "/url" => {
                    if let Some(u) = parts.get(1) {
                        cfg.base_url = (*u).to_string();
                        let h = fetch_health(client, &cfg.base_url, cfg.api_key.as_deref()).await;
                        if !h.model.is_empty() {
                            cfg.model = h.model.clone();
                            cfg.display_name = h.model;
                        }
                        println!("\x1b[90m[connected: {} ({})]\x1b[0m", cfg.base_url, cfg.display_name);
                    }
                }
                _ => {
                    println!("\x1b[90mCommands: /clear  /model [name]  /url <http://...>  /think  /temp <f>  /exit\x1b[0m");
                }
            }
            continue;
        }

        history.push(ChatMessage { role: "user".to_string(), content: line.to_string() });
        let (tx, mut rx) = mpsc::unbounded_channel();
        let params = StreamParams {
            base_url: cfg.base_url.clone(),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            messages: history.clone(),
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            enable_thinking: cfg.enable_thinking,
            reasoning_effort: cfg.reasoning_effort.clone(),
        };
        tokio::spawn(stream_chat(client.clone(), params, tx));

        let mut in_think = false;
        let mut reply = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Reasoning(s) => {
                    if !in_think {
                        print!("\x1b[90m<think>\n");
                        in_think = true;
                    }
                    print!("{s}");
                    let _ = io::stdout().flush();
                }
                StreamEvent::Content(s) => {
                    if in_think {
                        print!("\n</think>\x1b[0m\n");
                        in_think = false;
                    }
                    reply.push_str(&s);
                    print!("{s}");
                    let _ = io::stdout().flush();
                }
                StreamEvent::Done(st) => {
                    if in_think {
                        print!("\x1b[0m");
                    }
                    println!(
                        "\n\x1b[90m[{:.1} tok/s | TTFT {:.0} ms | {} tok | {:.2}s]\x1b[0m\n",
                        st.tok_per_sec, st.ttft_ms, st.gen_tokens, st.decode_s
                    );
                }
                StreamEvent::Error(e) => {
                    eprintln!("\n\x1b[31m[error] {e}\x1b[0m\n");
                }
            }
        }
        if !reply.is_empty() {
            history.push(ChatMessage { role: "assistant".to_string(), content: reply });
        }
    }
    Ok(())
}

fn push_stream_text(lines: &mut Vec<PlainLine>, kind: &'static str, text: &str) {
    for (idx, part) in text.split('\n').enumerate() {
        if idx == 0 {
            if let Some(last) = lines.last_mut() {
                if last.kind == kind {
                    last.text.push_str(part);
                    continue;
                }
            }
        }
        lines.push(PlainLine { kind, text: part.to_string() });
    }
}

pub async fn run_plain_scripted_capture(
    client: &Client,
    mut cfg: PlainConfig,
    script_path: PathBuf,
    capture_path: PathBuf,
) -> Result<()> {
    let raw = std::fs::read_to_string(&script_path)?;
    let steps: Vec<serde_json::Value> = serde_json::from_str(&raw)?;
    let mut out_file = std::fs::File::create(&capture_path)?;
    let t_start = Instant::now();
    let mut last_snap = Instant::now() - Duration::from_secs(1);
    let mut lines: Vec<PlainLine> = Vec::new();
    let mut history: Vec<ChatMessage> = Vec::new();

    let mut emit = |lines: &[PlainLine], force: bool| -> Result<()> {
        if !force && last_snap.elapsed() < Duration::from_millis(33) {
            return Ok(());
        }
        let frame = PlainFrame {
            t_ms: t_start.elapsed().as_millis() as u64,
            lines: lines.to_vec(),
        };
        writeln!(out_file, "{}", serde_json::to_string(&frame)?)?;
        last_snap = Instant::now();
        Ok(())
    };

    for step in steps {
        if let Some(new_url) = step.get("url").and_then(|v| v.as_str()) {
            cfg.base_url = new_url.to_string();
            cfg.api_key = step.get("api_key").and_then(|v| v.as_str()).map(|s| s.to_string());
            let h = fetch_health(client, &cfg.base_url, cfg.api_key.as_deref()).await;
            if let Some(m) = step.get("model").and_then(|v| v.as_str()) {
                cfg.model = m.to_string();
            } else if !h.model.is_empty() {
                cfg.model = h.model.clone();
            }
            cfg.display_name = step
                .get("display_name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| cfg.model.clone());
            if step.get("clear").and_then(|v| v.as_bool()).unwrap_or(false) {
                history.clear();
                lines.clear();
            }
            if let Some(cmd_line) = step.get("cmd").and_then(|v| v.as_str()) {
                lines.push(PlainLine { kind: "cmd", text: cmd_line.to_string() });
            }
            lines.push(PlainLine {
                kind: "banner",
                text: format!("rad-chat (CLI)  |  {}  |  {}", cfg.display_name, cfg.base_url),
            });
            lines.push(PlainLine { kind: "dim", text: String::new() });
            emit(&lines, true)?;
            tokio::time::sleep(Duration::from_millis(450)).await;
        }
        if let Some(think) = step.get("think").and_then(|v| v.as_bool()) {
            cfg.enable_thinking = think;
        }
        if let Some(mt) = step.get("max_tokens").and_then(|v| v.as_u64()) {
            cfg.max_tokens = mt as usize;
        }
        let Some(prompt) = step.get("prompt").and_then(|v| v.as_str()) else {
            continue;
        };

        lines.push(PlainLine { kind: "prompt", text: "> ".to_string() });
        for ch in prompt.chars() {
            if let Some(last) = lines.last_mut() {
                last.text.push(ch);
            }
            emit(&lines, false)?;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        emit(&lines, true)?;
        tokio::time::sleep(Duration::from_millis(200)).await;

        history.push(ChatMessage { role: "user".to_string(), content: prompt.to_string() });
        let (tx, mut rx) = mpsc::unbounded_channel();
        let params = StreamParams {
            base_url: cfg.base_url.clone(),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            messages: history.clone(),
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            enable_thinking: cfg.enable_thinking,
            reasoning_effort: cfg.reasoning_effort.clone(),
        };
        tokio::spawn(stream_chat(client.clone(), params, tx));

        let mut reply = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Reasoning(s) => {
                    push_stream_text(&mut lines, "think", &s);
                    emit(&lines, false)?;
                }
                StreamEvent::Content(s) => {
                    reply.push_str(&s);
                    push_stream_text(&mut lines, "text", &s);
                    emit(&lines, false)?;
                }
                StreamEvent::Done(st) => {
                    lines.push(PlainLine {
                        kind: "stats",
                        text: format!(
                            "[{:.1} tok/s | TTFT {:.0} ms | {} tok | {:.2}s]",
                            st.tok_per_sec, st.ttft_ms, st.gen_tokens, st.decode_s
                        ),
                    });
                    lines.push(PlainLine { kind: "dim", text: String::new() });
                    emit(&lines, true)?;
                }
                StreamEvent::Error(e) => {
                    lines.push(PlainLine { kind: "err", text: format!("[error] {e}") });
                    emit(&lines, true)?;
                }
            }
        }
        if !reply.is_empty() {
            history.push(ChatMessage { role: "assistant".to_string(), content: reply });
        }
        let hold_ms = step.get("hold_ms").and_then(|v| v.as_u64()).unwrap_or(1000);
        tokio::time::sleep(Duration::from_millis(hold_ms)).await;
        emit(&lines, true)?;
    }
    Ok(())
}
