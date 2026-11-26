mod client;
mod plain;
mod ui;

use anyhow::Result;
use clap::Parser;
use client::{fetch_health, fetch_metrics, fetch_models, stream_chat, ChatMessage, StreamEvent, StreamParams};
use plain::{run_plain_cli, run_plain_scripted_capture, PlainConfig};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::{CrosstermBackend, TestBackend};
use ratatui::style::Color;
use ratatui::Terminal;
use reqwest::Client;
use serde::Serialize;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use ui::{draw_ui, AppState, DisplayTurn};

#[derive(Parser, Debug)]
#[command(name = "rad-chat", about = "RAD interactive CLI and TUI chat with live streaming metrics")]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:6733")]
    url: String,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    display_name: Option<String>,
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
    #[arg(long, default_value_t = false)]
    think: bool,
    #[arg(long, default_value = "low")]
    effort: String,
    #[arg(long, default_value_t = false)]
    cli: bool,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    script: Option<PathBuf>,
    #[arg(long)]
    capture_jsonl: Option<PathBuf>,
    #[arg(long, default_value_t = 120)]
    cols: u16,
    #[arg(long, default_value_t = 34)]
    rows: u16,
}

#[derive(Serialize)]
struct CapturedCell {
    s: String,
    fg: [u8; 3],
    bg: [u8; 3],
    b: bool,
    i: bool,
}

#[derive(Serialize)]
struct CapturedFrame {
    t_ms: u64,
    cols: u16,
    rows: u16,
    cells: Vec<CapturedCell>,
}

fn color_rgb(c: Color, is_fg: bool) -> [u8; 3] {
    match c {
        Color::Rgb(r, g, b) => [r, g, b],
        Color::Black => [11, 15, 25],
        Color::White => [241, 245, 249],
        Color::Gray => [148, 163, 184],
        Color::DarkGray => [71, 85, 105],
        Color::Cyan => [56, 189, 248],
        Color::Green => [74, 222, 128],
        Color::Yellow => [251, 191, 36],
        Color::Red => [248, 113, 113],
        Color::Magenta => [192, 132, 252],
        _ => {
            if is_fg { [226, 232, 240] } else { [11, 15, 25] }
        }
    }
}

fn snapshot_frame(term: &Terminal<TestBackend>, t_ms: u64) -> CapturedFrame {
    let buf = term.backend().buffer();
    let cols = buf.area.width;
    let rows = buf.area.height;
    let mut cells = Vec::with_capacity((cols as usize) * (rows as usize));
    for y in 0..rows {
        for x in 0..cols {
            let c = &buf[(x, y)];
            cells.push(CapturedCell {
                s: c.symbol().to_string(),
                fg: color_rgb(c.fg, true),
                bg: color_rgb(c.bg, false),
                b: c.modifier.contains(ratatui::style::Modifier::BOLD),
                i: c.modifier.contains(ratatui::style::Modifier::ITALIC),
            });
        }
    }
    CapturedFrame { t_ms, cols, rows, cells }
}

async fn run_one_shot(client: &Client, cli: &Cli, model: String, prompt: String) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let params = StreamParams {
        base_url: cli.url.clone(),
        api_key: cli.api_key.clone(),
        model: model.clone(),
        messages: vec![ChatMessage { role: "user".to_string(), content: prompt }],
        max_tokens: cli.max_tokens,
        temperature: cli.temp,
        enable_thinking: cli.think,
        reasoning_effort: cli.effort.clone(),
    };
    tokio::spawn(stream_chat(client.clone(), params, tx));
    let mut in_think = false;
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
                print!("{s}");
                let _ = io::stdout().flush();
            }
            StreamEvent::Done(st) => {
                if in_think {
                    print!("\x1b[0m");
                }
                println!(
                    "\n\x1b[36m[RAD {}]\x1b[0m \x1b[32m{:.1} tok/s\x1b[0m | TTFT {:.0} ms | {} tok in {:.2}s | Cache {:.1}% | Spec {:.2}x",
                    model, st.tok_per_sec, st.ttft_ms, st.gen_tokens, st.decode_s, st.cache_hit_pct, st.avg_emit
                );
            }
            StreamEvent::Error(e) => {
                eprintln!("\n[rad-chat error] {e}");
            }
        }
    }
    Ok(())
}

async fn run_scripted_capture(
    client: &Client,
    cli: &Cli,
    mut app: AppState,
    script_path: PathBuf,
    capture_path: PathBuf,
) -> Result<()> {
    let raw = std::fs::read_to_string(&script_path)?;
    let steps: Vec<serde_json::Value> = serde_json::from_str(&raw)?;
    let backend = TestBackend::new(cli.cols, cli.rows);
    let mut term = Terminal::new(backend)?;
    let mut out_file = std::fs::File::create(&capture_path)?;
    let t_start = Instant::now();
    let mut last_snap = Instant::now() - Duration::from_secs(1);

    let mut emit_frame = |term: &mut Terminal<TestBackend>, app: &mut AppState, force: bool| -> Result<()> {
        if !force && last_snap.elapsed() < Duration::from_millis(33) {
            return Ok(());
        }
        term.draw(|f| draw_ui(f, app))?;
        let frame = snapshot_frame(term, t_start.elapsed().as_millis() as u64);
        writeln!(out_file, "{}", serde_json::to_string(&frame)?)?;
        last_snap = Instant::now();
        Ok(())
    };

    emit_frame(&mut term, &mut app, true)?;
    tokio::time::sleep(Duration::from_millis(600)).await;
    emit_frame(&mut term, &mut app, true)?;

    let mut history: Vec<ChatMessage> = Vec::new();
    let mut cur_api_key = cli.api_key.clone();

    for step in steps {
        if let Some(new_url) = step.get("url").and_then(|v| v.as_str()) {
            app.base_url = new_url.to_string();
            cur_api_key = step.get("api_key").and_then(|v| v.as_str()).map(|s| s.to_string());
            app.health = fetch_health(client, &app.base_url, cur_api_key.as_deref()).await;
            if let Some(m) = step.get("model").and_then(|v| v.as_str()) {
                app.model = m.to_string();
            } else if !app.health.model.is_empty() {
                app.model = app.health.model.clone();
            }
            if let Some(dn) = step.get("display_name").and_then(|v| v.as_str()) {
                app.display_name = dn.to_string();
            } else {
                app.display_name = app.model.clone();
            }
            if let Some(m) = fetch_metrics(client, &app.base_url, cur_api_key.as_deref()).await {
                app.metrics = m;
            }
            if step.get("clear").and_then(|v| v.as_bool()).unwrap_or(false) {
                history.clear();
                app.turns.clear();
            }
            app.status_msg = format!("Connected to {}", app.base_url);
            emit_frame(&mut term, &mut app, true)?;
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        if let Some(think) = step.get("think").and_then(|v| v.as_bool()) {
            app.enable_thinking = think;
        }
        if let Some(mt) = step.get("max_tokens").and_then(|v| v.as_u64()) {
            app.max_tokens = mt as usize;
        }
        let Some(prompt) = step.get("prompt").and_then(|v| v.as_str()) else {
            continue;
        };

        for ch in prompt.chars() {
            app.input.push(ch);
            emit_frame(&mut term, &mut app, false)?;
            tokio::time::sleep(Duration::from_millis(12)).await;
        }
        emit_frame(&mut term, &mut app, true)?;
        tokio::time::sleep(Duration::from_millis(250)).await;

        let user_msg = std::mem::take(&mut app.input);
        history.push(ChatMessage { role: "user".to_string(), content: user_msg.clone() });
        app.turns.push(DisplayTurn { role: "user".to_string(), reasoning: String::new(), content: user_msg, stats: None });
        app.turns.push(DisplayTurn { role: "assistant".to_string(), reasoning: String::new(), content: String::new(), stats: None });
        app.streaming = true;
        app.live_ttft_ms = None;
        app.live_chunks = 0;
        app.live_tok_s = 0.0;
        app.live_elapsed_s = 0.0;
        app.auto_scroll = true;
        emit_frame(&mut term, &mut app, true)?;

        let (tx, mut rx) = mpsc::unbounded_channel();
        let params = StreamParams {
            base_url: app.base_url.clone(),
            api_key: cur_api_key.clone(),
            model: app.model.clone(),
            messages: history.clone(),
            max_tokens: app.max_tokens,
            temperature: app.temperature,
            enable_thinking: app.enable_thinking,
            reasoning_effort: app.reasoning_effort.clone(),
        };
        tokio::spawn(stream_chat(client.clone(), params, tx));

        let t0 = Instant::now();
        let mut ttft_at: Option<Instant> = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Reasoning(s) => {
                    if ttft_at.is_none() {
                        ttft_at = Some(Instant::now());
                        app.live_ttft_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    app.live_chunks += 1;
                    app.live_elapsed_s = t0.elapsed().as_secs_f64();
                    let dec = ttft_at.unwrap().elapsed().as_secs_f64().max(0.01);
                    app.live_tok_s = app.live_chunks as f64 / dec;
                    app.push_tps_sample(app.live_tok_s);
                    if let Some(last) = app.turns.last_mut() {
                        last.reasoning.push_str(&s);
                    }
                    emit_frame(&mut term, &mut app, false)?;
                }
                StreamEvent::Content(s) => {
                    if ttft_at.is_none() {
                        ttft_at = Some(Instant::now());
                        app.live_ttft_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    app.live_chunks += 1;
                    app.live_elapsed_s = t0.elapsed().as_secs_f64();
                    let dec = ttft_at.unwrap().elapsed().as_secs_f64().max(0.01);
                    app.live_tok_s = app.live_chunks as f64 / dec;
                    app.push_tps_sample(app.live_tok_s);
                    if let Some(last) = app.turns.last_mut() {
                        last.content.push_str(&s);
                    }
                    emit_frame(&mut term, &mut app, false)?;
                }
                StreamEvent::Done(st) => {
                    app.streaming = false;
                    app.live_tok_s = st.tok_per_sec;
                    app.live_chunks = st.gen_tokens;
                    app.live_elapsed_s = st.total_s;
                    app.push_tps_sample(st.tok_per_sec);
                    if let Some(m) = fetch_metrics(client, &app.base_url, cur_api_key.as_deref()).await {
                        app.metrics = m;
                    }
                    if let Some(last) = app.turns.last_mut() {
                        history.push(ChatMessage { role: "assistant".to_string(), content: last.content.clone() });
                        last.stats = Some(st);
                    }
                    emit_frame(&mut term, &mut app, true)?;
                }
                StreamEvent::Error(e) => {
                    app.streaming = false;
                    app.status_msg = format!("Error: {e}");
                    emit_frame(&mut term, &mut app, true)?;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(1200)).await;
        emit_frame(&mut term, &mut app, true)?;
    }
    Ok(())
}

async fn handle_slash_command(
    cmd: &str,
    client: &Client,
    app: &mut AppState,
    history: &mut Vec<ChatMessage>,
    api_key: &Option<String>,
) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts.first().copied().unwrap_or("") {
        "/exit" | "/quit" => return true,
        "/clear" => {
            history.clear();
            app.turns.clear();
            app.status_msg = "Cleared conversation".to_string();
        }
        "/think" => {
            app.enable_thinking = !app.enable_thinking;
            app.show_thinking = app.enable_thinking;
            app.status_msg = format!("Thinking {}", if app.enable_thinking { "enabled" } else { "disabled" });
        }
        "/effort" => {
            if let Some(v) = parts.get(1) {
                app.reasoning_effort = v.to_string();
                app.status_msg = format!("Reasoning effort = {v}");
            }
        }
        "/temp" => {
            if let Some(v) = parts.get(1).and_then(|s| s.parse::<f32>().ok()) {
                app.temperature = v;
                app.status_msg = format!("Temperature = {v:.2}");
            }
        }
        "/model" => {
            if let Some(m) = parts.get(1) {
                app.model = m.to_string();
                app.display_name = m.to_string();
                app.status_msg = format!("Model = {m}");
            } else {
                let models = fetch_models(client, &app.base_url, api_key.as_deref()).await;
                app.turns.push(DisplayTurn {
                    role: "system".to_string(),
                    reasoning: String::new(),
                    content: format!("Available models: {}", models.join(", ")),
                    stats: None,
                });
            }
        }
        "/url" => {
            if let Some(u) = parts.get(1) {
                app.base_url = u.to_string();
                app.health = fetch_health(client, &app.base_url, api_key.as_deref()).await;
                if !app.health.model.is_empty() {
                    app.model = app.health.model.clone();
                    app.display_name = app.model.clone();
                }
                if let Some(m) = fetch_metrics(client, &app.base_url, api_key.as_deref()).await {
                    app.metrics = m;
                }
                app.status_msg = format!("Switched to {}", app.base_url);
            }
        }
        "/stats" => {
            if let Some(m) = fetch_metrics(client, &app.base_url, api_key.as_deref()).await {
                app.metrics = m;
            }
            app.turns.push(DisplayTurn {
                role: "system".to_string(),
                reasoning: String::new(),
                content: format!(
                    "Endpoint: {} ({}) | Generated: {} tok | Prompt: {} tok | Prefix Cache: {:.1}% ({} saved) | Spec: {:.2} tok/rnd | KV: {:.1}%",
                    app.base_url,
                    app.model,
                    app.metrics.generated_tokens_total,
                    app.metrics.prompt_tokens_total,
                    app.metrics.cache_hit_pct(),
                    app.metrics.prefix_cache_tokens_saved,
                    app.metrics.avg_emit(),
                    app.metrics.kv_cache_ratio * 100.0
                ),
                stats: None,
            });
        }
        _ => {
            app.turns.push(DisplayTurn {
                role: "system".to_string(),
                reasoning: String::new(),
                content: "Commands: /clear  /model [name]  /url <http://...>  /think  /effort <low|med|high>  /temp <f>  /stats  /exit".to_string(),
                stats: None,
            });
        }
    }
    false
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = Client::builder().build()?;
    let health = fetch_health(&client, &cli.url, cli.api_key.as_deref()).await;
    let model = cli
        .model
        .clone()
        .or_else(|| if !health.model.is_empty() { Some(health.model.clone()) } else { None })
        .unwrap_or_else(|| "rad".to_string());

    if let Some(prompt) = cli.prompt.clone() {
        return run_one_shot(&client, &cli, model, prompt).await;
    }

    let display_name = cli.display_name.clone().unwrap_or_else(|| model.clone());
    if cli.cli {
        let pcfg = PlainConfig {
            base_url: cli.url.clone(),
            api_key: cli.api_key.clone(),
            model,
            display_name,
            max_tokens: cli.max_tokens,
            temperature: cli.temp,
            enable_thinking: cli.think,
            reasoning_effort: cli.effort.clone(),
        };
        if let (Some(script), Some(cap)) = (cli.script.clone(), cli.capture_jsonl.clone()) {
            return run_plain_scripted_capture(&client, pcfg, script, cap).await;
        }
        return run_plain_cli(&client, pcfg).await;
    }
    let mut app = AppState::new(
        cli.url.clone(),
        model,
        display_name,
        cli.max_tokens,
        cli.temp,
        cli.think,
        cli.effort.clone(),
    );
    app.health = health;
    if let Some(m) = fetch_metrics(&client, &cli.url, cli.api_key.as_deref()).await {
        app.metrics = m;
    }

    if let (Some(script), Some(cap)) = (cli.script.clone(), cli.capture_jsonl.clone()) {
        return run_scripted_capture(&client, &cli, app, script, cap).await;
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut history: Vec<ChatMessage> = Vec::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();
    let mut stream_t0 = Instant::now();
    let mut ttft_at: Option<Instant> = None;
    let mut last_poll = Instant::now();

    let res = loop {
        terminal.draw(|f| draw_ui(f, &mut app))?;

        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Reasoning(s) => {
                    if ttft_at.is_none() {
                        ttft_at = Some(Instant::now());
                        app.live_ttft_ms = Some(stream_t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    app.live_chunks += 1;
                    app.live_elapsed_s = stream_t0.elapsed().as_secs_f64();
                    let dec = ttft_at.unwrap().elapsed().as_secs_f64().max(0.01);
                    app.live_tok_s = app.live_chunks as f64 / dec;
                    app.push_tps_sample(app.live_tok_s);
                    if let Some(last) = app.turns.last_mut() {
                        last.reasoning.push_str(&s);
                    }
                }
                StreamEvent::Content(s) => {
                    if ttft_at.is_none() {
                        ttft_at = Some(Instant::now());
                        app.live_ttft_ms = Some(stream_t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    app.live_chunks += 1;
                    app.live_elapsed_s = stream_t0.elapsed().as_secs_f64();
                    let dec = ttft_at.unwrap().elapsed().as_secs_f64().max(0.01);
                    app.live_tok_s = app.live_chunks as f64 / dec;
                    app.push_tps_sample(app.live_tok_s);
                    if let Some(last) = app.turns.last_mut() {
                        last.content.push_str(&s);
                    }
                }
                StreamEvent::Done(st) => {
                    app.streaming = false;
                    app.live_tok_s = st.tok_per_sec;
                    app.live_chunks = st.gen_tokens;
                    app.live_elapsed_s = st.total_s;
                    app.push_tps_sample(st.tok_per_sec);
                    if let Some(m) = fetch_metrics(&client, &app.base_url, cli.api_key.as_deref()).await {
                        app.metrics = m;
                    }
                    if let Some(last) = app.turns.last_mut() {
                        history.push(ChatMessage { role: "assistant".to_string(), content: last.content.clone() });
                        last.stats = Some(st);
                    }
                    app.status_msg = "Ready  |  /help for commands".to_string();
                }
                StreamEvent::Error(e) => {
                    app.streaming = false;
                    app.status_msg = format!("Error: {e}");
                }
            }
        }

        if last_poll.elapsed() >= Duration::from_secs(2) {
            if let Some(m) = fetch_metrics(&client, &app.base_url, cli.api_key.as_deref()).await {
                app.metrics = m;
            }
            last_poll = Instant::now();
        }

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(k) = event::read()? {
                if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
                    break Ok(());
                }
                match k.code {
                    KeyCode::Esc => break Ok(()),
                    KeyCode::Up => {
                        app.auto_scroll = false;
                        app.scroll = app.scroll.saturating_sub(2);
                    }
                    KeyCode::Down => {
                        app.scroll = app.scroll.saturating_add(2);
                    }
                    KeyCode::PageUp => {
                        app.auto_scroll = false;
                        app.scroll = app.scroll.saturating_sub(12);
                    }
                    KeyCode::PageDown => {
                        app.scroll = app.scroll.saturating_add(12);
                    }
                    KeyCode::Backspace if !app.streaming => {
                        app.input.pop();
                    }
                    KeyCode::Char(c) if !app.streaming => {
                        app.input.push(c);
                    }
                    KeyCode::Enter if !app.streaming => {
                        let line = app.input.trim().to_string();
                        app.input.clear();
                        if line.is_empty() {
                            continue;
                        }
                        if line.starts_with('/') {
                            if handle_slash_command(&line, &client, &mut app, &mut history, &cli.api_key).await {
                                break Ok(());
                            }
                            continue;
                        }
                        history.push(ChatMessage { role: "user".to_string(), content: line.clone() });
                        app.turns.push(DisplayTurn { role: "user".to_string(), reasoning: String::new(), content: line, stats: None });
                        app.turns.push(DisplayTurn { role: "assistant".to_string(), reasoning: String::new(), content: String::new(), stats: None });
                        app.streaming = true;
                        app.auto_scroll = true;
                        app.live_ttft_ms = None;
                        app.live_chunks = 0;
                        app.live_tok_s = 0.0;
                        app.live_elapsed_s = 0.0;
                        app.status_msg = "Streaming...".to_string();
                        stream_t0 = Instant::now();
                        ttft_at = None;
                        let params = StreamParams {
                            base_url: app.base_url.clone(),
                            api_key: cli.api_key.clone(),
                            model: app.model.clone(),
                            messages: history.clone(),
                            max_tokens: app.max_tokens,
                            temperature: app.temperature,
                            enable_thinking: app.enable_thinking,
                            reasoning_effort: app.reasoning_effort.clone(),
                        };
                        tokio::spawn(stream_chat(client.clone(), params, tx.clone()));
                    }
                    _ => {}
                }
            }
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}
