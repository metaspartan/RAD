use crate::client::{ServerHealth, ServerMetrics, TurnStats};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Sparkline, Wrap};
use ratatui::Frame;

#[derive(Clone, Debug)]
pub struct DisplayTurn {
    pub role: String,
    pub reasoning: String,
    pub content: String,
    pub stats: Option<TurnStats>,
}

pub struct AppState {
    pub base_url: String,
    pub model: String,
    pub display_name: String,
    pub health: ServerHealth,
    pub metrics: ServerMetrics,
    pub turns: Vec<DisplayTurn>,
    pub input: String,
    pub scroll: u16,
    pub auto_scroll: bool,
    pub streaming: bool,
    pub show_thinking: bool,
    pub enable_thinking: bool,
    pub reasoning_effort: String,
    pub temperature: f32,
    pub max_tokens: usize,
    pub live_ttft_ms: Option<f64>,
    pub live_chunks: usize,
    pub live_tok_s: f64,
    pub live_elapsed_s: f64,
    pub tps_history: Vec<u64>,
    pub status_msg: String,
}

impl AppState {
    pub fn new(
        base_url: String,
        model: String,
        display_name: String,
        max_tokens: usize,
        temperature: f32,
        enable_thinking: bool,
        reasoning_effort: String,
    ) -> Self {
        Self {
            base_url,
            model,
            display_name,
            health: ServerHealth::default(),
            metrics: ServerMetrics::default(),
            turns: Vec::new(),
            input: String::new(),
            scroll: 0,
            auto_scroll: true,
            streaming: false,
            show_thinking: true,
            enable_thinking,
            reasoning_effort,
            temperature,
            max_tokens,
            live_ttft_ms: None,
            live_chunks: 0,
            live_tok_s: 0.0,
            live_elapsed_s: 0.0,
            tps_history: vec![0; 48],
            status_msg: "Ready  |  /help for commands  |  Enter to send  |  Ctrl+C to quit".to_string(),
        }
    }

    pub fn push_tps_sample(&mut self, tps: f64) {
        if self.tps_history.len() >= 48 {
            self.tps_history.remove(0);
        }
        self.tps_history.push(tps.round().max(0.0) as u64);
    }
}

pub fn draw_ui(f: &mut Frame, app: &mut AppState) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(4),
            Constraint::Length(3),
        ])
        .split(area);

    draw_header(f, app, chunks[0]);
    draw_chat(f, app, chunks[1]);
    draw_telemetry(f, app, chunks[2]);
    draw_input(f, app, chunks[3]);
}

fn draw_header(f: &mut Frame, app: &AppState, area: Rect) {
    let dot_color = if app.health.ok { Color::Rgb(74, 222, 128) } else { Color::Rgb(248, 113, 113) };
    let engine = if app.health.engine.is_empty() { "rad" } else { &app.health.engine };
    let ctx_str = if app.health.max_ctx > 0 {
        format!("{}K ctx", app.health.max_ctx / 1024)
    } else {
        "live".to_string()
    };
    let tp_str = if app.health.world_size > 1 {
        format!("TP={}", app.health.world_size)
    } else {
        "1x GPU".to_string()
    };
    let think_str = if app.enable_thinking {
        format!("think:{}", app.reasoning_effort)
    } else {
        "think:off".to_string()
    };

    let line = Line::from(vec![
        Span::styled(
            " RAD ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Rgb(56, 189, 248))
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled("●", Style::default().fg(dot_color)),
        Span::styled(
            format!(" {} ", app.display_name),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(engine, Style::default().fg(Color::Rgb(167, 139, 250))),
        Span::styled(format!(" ({tp_str}, {ctx_str}) "), Style::default().fg(Color::Gray)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("temp={:.2} ", app.temperature), Style::default().fg(Color::Rgb(251, 191, 36))),
        Span::styled(format!("{think_str} "), Style::default().fg(Color::Rgb(148, 163, 184))),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(&app.base_url, Style::default().fg(Color::DarkGray)),
    ]);

    let header = Paragraph::new(line).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(51, 65, 85))),
    );
    f.render_widget(header, area);
}

fn draw_chat(f: &mut Frame, app: &mut AppState, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();
    if app.turns.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::styled(
                "  Welcome to RAD Chat TUI",
                Style::default().fg(Color::Rgb(56, 189, 248)).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "  (OpenAI SSE + Live Prometheus Telemetry)",
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        lines.push(Line::from(Span::styled(
            "  Type a prompt below and press Enter. Use /help for slash commands.",
            Style::default().fg(Color::Gray),
        )));
    }

    for (idx, turn) in app.turns.iter().enumerate() {
        let is_last = idx + 1 == app.turns.len();
        if turn.role == "user" {
            lines.push(Line::from(vec![
                Span::styled(
                    "▶ YOU ",
                    Style::default().fg(Color::Rgb(56, 189, 248)).add_modifier(Modifier::BOLD),
                ),
            ]));
            for l in turn.content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {l}"),
                    Style::default().fg(Color::White),
                )));
            }
            lines.push(Line::raw(""));
        } else if turn.role == "system" {
            for l in turn.content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  [rad] {l}"),
                    Style::default().fg(Color::Rgb(251, 191, 36)),
                )));
            }
            lines.push(Line::raw(""));
        } else {
            let badge = if app.streaming && is_last {
                Span::styled(
                    format!("⚡ {}  ({:.1} tok/s) ", app.display_name, app.live_tok_s),
                    Style::default().fg(Color::Rgb(74, 222, 128)).add_modifier(Modifier::BOLD),
                )
            } else {
                Span::styled(
                    format!("◆ {} ", app.display_name),
                    Style::default().fg(Color::Rgb(167, 139, 250)).add_modifier(Modifier::BOLD),
                )
            };
            lines.push(Line::from(vec![badge]));

            if app.show_thinking && !turn.reasoning.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  ┌─ thinking",
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                )));
                for l in turn.reasoning.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  │ {l}"),
                        Style::default().fg(Color::Rgb(148, 163, 184)).add_modifier(Modifier::ITALIC),
                    )));
                }
                lines.push(Line::from(Span::styled(
                    "  └─",
                    Style::default().fg(Color::DarkGray),
                )));
            }

            if !turn.content.is_empty() {
                for l in turn.content.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {l}"),
                        Style::default().fg(Color::Rgb(241, 245, 249)),
                    )));
                }
            }
            if app.streaming && is_last {
                lines.push(Line::from(Span::styled(
                    "  ▌",
                    Style::default().fg(Color::Rgb(74, 222, 128)),
                )));
            }

            if let Some(ref st) = turn.stats {
                lines.push(Line::from(vec![
                    Span::styled("  ╰─ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:.1} tok/s", st.tok_per_sec),
                        Style::default().fg(Color::Rgb(74, 222, 128)).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("TTFT {:.0} ms", st.ttft_ms), Style::default().fg(Color::Rgb(56, 189, 248))),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("{} tok in {:.2}s", st.gen_tokens, st.decode_s), Style::default().fg(Color::Gray)),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("Cache {:.1}%", st.cache_hit_pct), Style::default().fg(Color::Rgb(251, 191, 36))),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("Spec {:.2}x", st.avg_emit), Style::default().fg(Color::Rgb(192, 132, 252))),
                ]));
            }
            lines.push(Line::raw(""));
        }
    }

    let inner_width = area.width.saturating_sub(2).max(1) as usize;
    let total_visual_lines: usize = lines
        .iter()
        .map(|l| {
            let w = l.width();
            if w == 0 { 1 } else { (w + inner_width - 1) / inner_width }
        })
        .sum();
    let visible = area.height.saturating_sub(2) as usize;
    let max_scroll = total_visual_lines.saturating_sub(visible) as u16;
    if app.auto_scroll {
        app.scroll = max_scroll;
    } else if app.scroll > max_scroll {
        app.scroll = max_scroll;
    }

    let chat = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll, 0))
        .block(
            Block::default()
                .title(Span::styled(" Conversation ", Style::default().fg(Color::Rgb(148, 163, 184))))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(51, 65, 85))),
        );
    f.render_widget(chat, area);
}

fn draw_telemetry(f: &mut Frame, app: &AppState, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(66), Constraint::Percentage(34)])
        .split(area);

    let ttft_txt = app
        .live_ttft_ms
        .map(|v| format!("{v:.0} ms"))
        .unwrap_or_else(|| "-".to_string());
    let state_span = if app.streaming {
        Span::styled(" STREAMING ", Style::default().fg(Color::Black).bg(Color::Rgb(74, 222, 128)).add_modifier(Modifier::BOLD))
    } else {
        Span::styled(" IDLE ", Style::default().fg(Color::Black).bg(Color::Rgb(148, 163, 184)).add_modifier(Modifier::BOLD))
    };

    let line1 = Line::from(vec![
        state_span,
        Span::raw("  "),
        Span::styled(
            format!("{:>6.1} tok/s", app.live_tok_s),
            Style::default().fg(Color::Rgb(74, 222, 128)).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  │  TTFT ", Style::default().fg(Color::DarkGray)),
        Span::styled(ttft_txt, Style::default().fg(Color::Rgb(56, 189, 248))),
        Span::styled("  │  Out ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{} tok", app.live_chunks), Style::default().fg(Color::White)),
        Span::styled(format!(" ({:.2}s)", app.live_elapsed_s), Style::default().fg(Color::Gray)),
    ]);

    let line2 = Line::from(vec![
        Span::styled("Server: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("Cache {:.1}%", app.metrics.cache_hit_pct()),
            Style::default().fg(Color::Rgb(251, 191, 36)),
        ),
        Span::styled("  │  Spec ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.2} tok/rnd", app.metrics.avg_emit()),
            Style::default().fg(Color::Rgb(192, 132, 252)),
        ),
        Span::styled("  │  KV ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.1}%", app.metrics.kv_cache_ratio * 100.0),
            Style::default().fg(Color::Rgb(56, 189, 248)),
        ),
        Span::styled("  │  Total ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{} gen", app.metrics.generated_tokens_total),
            Style::default().fg(Color::Gray),
        ),
    ]);

    let stats_box = Paragraph::new(vec![line1, line2]).block(
        Block::default()
            .title(Span::styled(" Live Metrics ", Style::default().fg(Color::Rgb(148, 163, 184))))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(51, 65, 85))),
    );
    f.render_widget(stats_box, cols[0]);

    let spark = Sparkline::default()
        .block(
            Block::default()
                .title(Span::styled(" Throughput (tok/s) ", Style::default().fg(Color::Rgb(148, 163, 184))))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(51, 65, 85))),
        )
        .data(&app.tps_history)
        .style(Style::default().fg(Color::Rgb(56, 189, 248)));
    f.render_widget(spark, cols[1]);
}

fn draw_input(f: &mut Frame, app: &AppState, area: Rect) {
    let title = format!(" Prompt  ({}) ", app.status_msg);
    let prompt_text = format!("> {}█", app.input);
    let input = Paragraph::new(prompt_text)
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .title(Span::styled(title, Style::default().fg(Color::Rgb(148, 163, 184))))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(if app.streaming {
                    Color::Rgb(74, 222, 128)
                } else {
                    Color::Rgb(56, 189, 248)
                })),
        );
    f.render_widget(input, area);
}
