use serde_json::json;
use crate::grammar;
use crate::types::{AnthropicMessagesRequest, ChatMessage};

pub(crate) fn msg_text_and_images(val: &serde_json::Value, images: &mut Vec<String>) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            let mut out = String::new();
            for item in arr {
                let ty = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if ty == "image_url" || ty == "input_image" {
                    if let Some(url) = item.get("image_url").and_then(|u| u.get("url").or(Some(u))).and_then(|v| v.as_str()) {
                        images.push(url.to_string());
                        out.push_str("<|content_image|>");
                    }
                } else if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                    out.push_str(t);
                } else if let Some(s) = item.as_str() {
                    out.push_str(s);
                }
            }
            out
        }
        _ => String::new(),
    }
}

pub(crate) fn normalize_tool_specs(tools: &[serde_json::Value]) -> (Vec<serde_json::Value>, Vec<String>) {
    let mut specs = Vec::new();
    let mut names = Vec::new();
    for t in tools {
        let fn_obj = t.get("function").unwrap_or(t);
        if let Some(name) = fn_obj.get("name").and_then(|v| v.as_str()) {
            names.push(name.to_string());
            let desc = fn_obj.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let params = fn_obj.get("parameters").cloned().unwrap_or_else(|| json!({}));
            let ty = t.get("type").and_then(|v| v.as_str()).unwrap_or("function");
            specs.push(json!({
                "description": desc,
                "name": name,
                "parameters": params,
                "type": ty
            }));
        }
    }
    (specs, names)
}

pub(crate) fn canonical_tool_sig(name: &str, raw_args: &str) -> String {
    let norm_args = match serde_json::from_str::<serde_json::Value>(raw_args.trim()) {
        Ok(v) => serde_json::to_string(&v).unwrap_or_else(|_| raw_args.trim().to_string()),
        Err(_) => raw_args.trim().to_string(),
    };
    format!("{}:{}", name.trim(), norm_args)
}

pub(crate) fn detect_repeated_tool_loop(messages: &[ChatMessage]) -> (usize, Vec<String>) {
    let mut recent_sigs: Vec<String> = Vec::new();
    for m in messages.iter().rev() {
        match m.role.as_str() {
            "tool" => continue,
            "assistant" => {
                if let Some(ref tcs) = m.tool_calls {
                    if !tcs.is_empty() {
                        let mut parts = Vec::new();
                        for tc in tcs {
                            let fn_obj = tc.get("function").unwrap_or(tc);
                            let name = fn_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let args = fn_obj
                                .get("arguments")
                                .or_else(|| fn_obj.get("args"))
                                .map(|v| match v {
                                    serde_json::Value::String(s) => s.clone(),
                                    other => serde_json::to_string(other).unwrap_or_default(),
                                })
                                .unwrap_or_default();
                            parts.push(canonical_tool_sig(name, &args));
                        }
                        recent_sigs.push(parts.join("|"));
                        if recent_sigs.len() >= 10 {
                            break;
                        }
                        continue;
                    }
                }
                break;
            }
            "user" => break,
            _ => continue,
        }
    }
    if recent_sigs.len() < 2 {
        return (0, Vec::new());
    }
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for sig in &recent_sigs {
        *counts.entry(sig.as_str()).or_insert(0) += 1;
    }
    let mut max_repeats = 0usize;
    let mut blocked = Vec::new();
    for (sig, cnt) in counts {
        if cnt >= 2 {
            let reps = cnt - 1;
            if reps > max_repeats {
                max_repeats = reps;
            }
            blocked.push(sig.to_string());
        }
    }
    (max_repeats, blocked)
}

pub(crate) fn apply_tool_loop_guard(
    prompt: &mut String,
    sampling: &mut grammar::SamplingParams,
    repeat_count: usize,
) {
    if repeat_count == 0 {
        return;
    }
    if repeat_count == 1 {
        sampling.temperature = sampling.temperature.max(0.5);
        sampling.repetition_penalty = sampling.repetition_penalty.max(1.08);
        let nudge = "<|message_system|><|content_text|>You already ran that exact tool call and received its output above. Do NOT repeat the same command again. Take a different action or answer the user directly.<|end_message|>";
        if let Some(pos) = prompt.rfind("<|message_model|>") {
            prompt.insert_str(pos, nudge);
        }
    } else {
        sampling.temperature = sampling.temperature.max(0.6);
        sampling.repetition_penalty = sampling.repetition_penalty.max(1.12);
        let nudge = "<|message_system|><|content_text|>STOP: You have repeated the exact same tool call multiple times in a row. That identical tool call is now blocked. Do NOT run it again. Either run a completely different command to make progress or reply directly to the user in plain text now.<|end_message|>";
        if let Some(pos) = prompt.rfind("<|message_model|>") {
            prompt.insert_str(pos, nudge);
        }
    }
}

pub(crate) fn format_prompt_with_tools(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    #[allow(dead_code)]
    enable_thinking: bool,
) -> (String, Vec<String>, Vec<String>) {
    let mut s = String::from("<|begin_of_text|>");
    let mut images = Vec::new();
    let mut allowed_tools = Vec::new();
    if let Some(ts) = tools {
        if !ts.is_empty() {
            let (specs, names) = normalize_tool_specs(ts);
            allowed_tools = names;
            if !specs.is_empty() {
                let spec_json = serde_json::to_string(&specs).unwrap_or_else(|_| "[]".into());
                s.push_str("<|message_system|>tool_declare<|content_xml|>");
                s.push_str(&spec_json);
                s.push_str("<|end_message|>");
            }
        }
    }
    let mut effort_emitted = false;
    let effort_str = if enable_thinking { "0.7" } else { "0" };
    for m in messages {
        if !effort_emitted && !matches!(m.role.as_str(), "system" | "developer") {
            s.push_str("<|message_system|><|content_text|>Thinking effort level: ");
            s.push_str(effort_str);
            s.push_str("<|end_message|>");
            effort_emitted = true;
        }
        let txt = msg_text_and_images(&m.content, &mut images);
        match m.role.as_str() {
            "system" | "developer" => {
                s.push_str("<|message_system|><|content_text|>");
                s.push_str(&txt);
                s.push_str("<|end_message|>");
            }
            "user" => {
                s.push_str("<|message_user|><|content_text|>");
                s.push_str(&txt);
                s.push_str("<|end_message|>");
            }
            "assistant" => {
                if let Some(ref rc) = m.reasoning_content {
                    if !rc.is_empty() {
                        s.push_str("<|message_model|><|content_thinking|>");
                        s.push_str(rc);
                        s.push_str("<|end_message|>");
                    }
                }
                let has_tcs = m.tool_calls.as_ref().map(|v| !v.is_empty()).unwrap_or(false);
                let first_cached_tc = m
                    .tool_calls
                    .as_ref()
                    .and_then(|v| v.first())
                    .and_then(|tc| tc.get("id"))
                    .and_then(|v| v.as_str())
                    .and_then(grammar::lookup_raw_tool_call);
                let first_inline_tc = first_cached_tc.as_ref().filter(|r| r.inline_after_text);
                if first_inline_tc.is_none() {
                    let eff_txt = if txt.is_empty() {
                        first_cached_tc.as_ref().map(|r| r.raw_prefix_text.as_str()).unwrap_or("")
                    } else {
                        txt.as_str()
                    };
                    if !eff_txt.is_empty() || !has_tcs || !enable_thinking {
                        s.push_str("<|message_model|><|content_text|>");
                        s.push_str(eff_txt);
                        s.push_str("<|end_message|>");
                    }
                }
                if let Some(ref tcs) = m.tool_calls {
                    for (tc_idx, tc) in tcs.iter().enumerate() {
                        let fn_obj = tc.get("function").unwrap_or(tc);
                        if let Some(name) = fn_obj.get("name").and_then(|v| v.as_str()) {
                            let cached_raw = tc
                                .get("id")
                                .and_then(|v| v.as_str())
                                .and_then(grammar::lookup_raw_tool_call);
                            let call_json = cached_raw
                                .as_ref()
                                .map(|r| r.raw_payload.clone())
                                .unwrap_or_else(|| {
                                    let args_json = match fn_obj.get("arguments").or_else(|| fn_obj.get("args")) {
                                        Some(serde_json::Value::String(st)) => {
                                            let trimmed = st.trim();
                                            if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
                                                trimmed.to_string()
                                            } else {
                                                "{}".into()
                                            }
                                        }
                                        Some(v) if v.is_object() => serde_json::to_string(v).unwrap_or_else(|_| "{}".into()),
                                        _ => "{}".into(),
                                    };
                                    let name_json = serde_json::to_string(name).unwrap_or_else(|_| format!("\"{name}\""));
                                    format!("{{\"name\":{name_json},\"args\":{args_json}}}")
                                });
                            if tc_idx == 0 {
                                if let Some(ref r) = first_inline_tc {
                                    s.push_str("<|message_model|><|content_text|>");
                                    s.push_str(&r.raw_prefix_text);
                                    s.push_str("<|content_invoke_tool_json|>");
                                    s.push_str(&call_json);
                                    s.push_str("<|end_message|>");
                                    continue;
                                }
                            }
                            s.push_str("<|message_model|>");
                            s.push_str(name);
                            s.push_str("<|content_invoke_tool_json|>");
                            s.push_str(&call_json);
                            s.push_str("<|end_message|>");
                        }
                    }
                }
                s.push_str("<|content_model_end_sampling|>");
            }
            "tool" => {
                let mut tname = m.name.clone().unwrap_or_default();
                if tname.is_empty() {
                    if let Some(ref cid) = m.tool_call_id {
                        'outer: for prev in messages {
                            if prev.role == "assistant" {
                                if let Some(ref tcs) = prev.tool_calls {
                                    for tc in tcs {
                                        if tc.get("id").and_then(|v| v.as_str()) == Some(cid.as_str()) {
                                            if let Some(n) = tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()) {
                                                tname = n.to_string();
                                                break 'outer;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                s.push_str("<|message_tool|>");
                s.push_str(&tname);
                s.push_str("<|content_text|>");
                s.push_str(&txt);
                s.push_str("<|end_message|>");
            }
            _ => {}
        }
    }
    if !effort_emitted {
        s.push_str("<|message_system|><|content_text|>Thinking effort level: ");
        s.push_str(effort_str);
        s.push_str("<|end_message|>");
    }
    if enable_thinking {
        s.push_str("<|message_model|><|content_thinking|>");
    } else {
        s.push_str("<|message_model|><|content_text|>");
    }
    (s, images, allowed_tools)
}

pub(crate) fn format_prompt(messages: &[ChatMessage], enable_thinking: bool) -> String {
    format_prompt_with_tools(messages, None, enable_thinking).0
}

pub(crate) fn format_responses_input_with_tools(
    input: &serde_json::Value,
    instructions: Option<&str>,
    tools: Option<&[serde_json::Value]>,
    #[allow(dead_code)]
    enable_thinking: bool,
) -> (String, Vec<String>, Vec<String>) {
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(inst) = instructions {
        if !inst.is_empty() {
            msgs.push(ChatMessage {
                role: "system".into(),
                content: json!(inst),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
    }
    match input {
        serde_json::Value::String(s) => {
            msgs.push(ChatMessage {
                role: "user".into(),
                content: json!(s),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if item_type == "function_call" {
                    let call_id = item.get("call_id").or_else(|| item.get("id")).and_then(|v| v.as_str()).unwrap_or("call_0").to_string();
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let arguments = item.get("arguments").cloned().unwrap_or_else(|| json!("{}"));
                    let tc_json = json!({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments}
                    });
                    if let Some(last) = msgs.last_mut() {
                        if last.role == "assistant" {
                            last.tool_calls.get_or_insert_with(Vec::new).push(tc_json);
                            continue;
                        }
                    }
                    msgs.push(ChatMessage {
                        role: "assistant".into(),
                        content: json!(""),
                        tool_calls: Some(vec![tc_json]),
                        tool_call_id: None,
                        name: None,
                        reasoning_content: None,
                    });
                } else if item_type == "function_call_output" {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).map(|s| s.to_string());
                    let output = item.get("output").cloned().unwrap_or_else(|| json!(""));
                    msgs.push(ChatMessage {
                        role: "tool".into(),
                        content: output,
                        tool_calls: None,
                        tool_call_id: call_id,
                        name: None,
                        reasoning_content: None,
                    });
                } else {
                    let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string();
                    let content = item.get("content").cloned().unwrap_or_else(|| json!(""));
                    msgs.push(ChatMessage {
                        role,
                        content,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                        reasoning_content: None,
                    });
                }
            }
        }
        _ => {
            msgs.push(ChatMessage {
                role: "user".into(),
                content: json!(""),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
    }
    format_prompt_with_tools(&msgs, tools, enable_thinking)
}

#[allow(dead_code)]
pub(crate) fn split_thinking(raw: &str, enable_thinking: bool) -> (Option<String>, String) {
    let cleaned = raw
        .replace("<|content_model_end_sampling|>", "")
        .replace("<|end_message|>", "");
    if !enable_thinking {
        if let Some((_, rest)) = cleaned.rsplit_once("<|content_text|>") {
            return (None, rest.trim().to_string());
        }
        return (None, cleaned.trim().to_string());
    }
    if let Some((think, ans)) = cleaned.split_once("<|content_text|>") {
        let t = think.replace("<|message_model|>", "").replace("<|content_thinking|>", "").trim().to_string();
        let a = ans.trim().to_string();
        (if t.is_empty() { None } else { Some(t) }, a)
    } else {
        (None, cleaned.trim().to_string())
    }
}


pub(crate) fn convert_anthropic_request(
    req: &AnthropicMessagesRequest,
) -> (Vec<ChatMessage>, Option<Vec<serde_json::Value>>, bool) {
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(ref sys) = req.system {
        let sys_txt = match sys {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !sys_txt.is_empty() {
            msgs.push(ChatMessage {
                role: "system".into(),
                content: json!(sys_txt),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
    }
    for m in &req.messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user").to_string();
        let content = m.get("content").cloned().unwrap_or(serde_json::Value::Null);
        if let Some(arr) = content.as_array() {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tcs: Vec<serde_json::Value> = Vec::new();
            let mut tool_results: Vec<(String, String)> = Vec::new();
            let mut thinking_parts: Vec<String> = Vec::new();
            for blk in arr {
                let btype = blk.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match btype {
                    "text" => {
                        if let Some(t) = blk.get("text").and_then(|v| v.as_str()) {
                            text_parts.push(t.to_string());
                        }
                    }
                    "thinking" => {
                        if let Some(t) = blk.get("thinking").and_then(|v| v.as_str()) {
                            thinking_parts.push(t.to_string());
                        }
                    }
                    "tool_use" => {
                        let id = blk.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let name = blk.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let input = blk.get("input").cloned().unwrap_or_else(|| json!({}));
                        let args_str = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                        tcs.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": name, "arguments": args_str}
                        }));
                    }
                    "tool_result" => {
                        let tid = blk.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let body = match blk.get("content") {
                            Some(serde_json::Value::String(s)) => s.clone(),
                            Some(serde_json::Value::Array(sub)) => sub
                                .iter()
                                .filter_map(|x| x.get("text").and_then(|t| t.as_str()))
                                .collect::<Vec<_>>()
                                .join("\n"),
                            Some(v) => v.to_string(),
                            None => String::new(),
                        };
                        tool_results.push((tid, body));
                    }
                    _ => {}
                }
            }
            if !tool_results.is_empty() {
                for (tid, body) in tool_results {
                    msgs.push(ChatMessage {
                        role: "tool".into(),
                        content: json!(body),
                        tool_calls: None,
                        tool_call_id: Some(tid),
                        name: None,
                        reasoning_content: None,
                    });
                }
                if !text_parts.is_empty() {
                    msgs.push(ChatMessage {
                        role,
                        content: json!(text_parts.join("\n")),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                        reasoning_content: None,
                    });
                }
            } else {
                msgs.push(ChatMessage {
                    role,
                    content: if text_parts.is_empty() { serde_json::Value::Null } else { json!(text_parts.join("\n")) },
                    tool_calls: if tcs.is_empty() { None } else { Some(tcs) },
                    tool_call_id: None,
                    name: None,
                    reasoning_content: if thinking_parts.is_empty() { None } else { Some(thinking_parts.join("\n")) },
                });
            }
        } else {
            msgs.push(ChatMessage {
                role,
                content,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
    }
    let oai_tools = req.tools.as_ref().map(|ts| {
        ts.iter()
            .map(|t| {
                let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let desc = t.get("description").and_then(|v| v.as_str()).unwrap_or("");
                let params = t.get("input_schema").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}}));
                json!({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
            })
            .collect::<Vec<_>>()
    });
    let enable_thinking = req
        .thinking
        .as_ref()
        .and_then(|t| t.get("type"))
        .and_then(|v| v.as_str())
        .map(|s| s == "enabled")
        .unwrap_or(false);
    (msgs, oai_tools, enable_thinking)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_prompt_recovers_dropped_preamble_from_tool_call_cache() {
        let cid = "call_unit_preamble_0";
        grammar::remember_raw_tool_call(
            cid,
            grammar::RawToolCall {
                inline_after_text: false,
                raw_prefix_text: "I will check that now.".into(),
                raw_payload: r#"{"name":"exec","args":{"command":"ls"}}"#.into(),
            },
        );
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: json!("list files"),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: serde_json::Value::Null,
                tool_calls: Some(vec![json!({
                    "id": cid,
                    "type": "function",
                    "function": {"name": "exec", "arguments": "{\"command\": \"ls\"}"}
                })]),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            },
        ];
        let (prompt, _, _) = format_prompt_with_tools(&msgs, None, false);
        assert!(prompt.contains("<|message_model|><|content_text|>I will check that now.<|end_message|>"));
        assert!(prompt.contains(r#"<|content_invoke_tool_json|>{"name":"exec","args":{"command":"ls"}}<|end_message|>"#));
    }

    #[test]
    fn split_thinking_extracts_reasoning_and_content() {
        let raw = "step 1<|end_message|><|message_model|><|content_text|>final answer";
        let (r, c) = split_thinking(raw, true);
        assert_eq!(r.as_deref(), Some("step 1"));
        assert_eq!(c, "final answer");
    }

    #[test]
    fn repeated_tool_loop_normalizes_json_and_escalates() {
        let make_turn = |args: &str| ChatMessage {
            role: "assistant".into(),
            content: json!("checking"),
            tool_calls: Some(vec![json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "exec", "arguments": args}
            })]),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        };
        let msgs2 = vec![
            make_turn(r#"{"command": "ls -la", "cwd": "/tmp"}"#),
            make_turn(r#"{"cwd":"/tmp","command":"ls -la"}"#),
        ];
        let (cnt2, sigs2) = detect_repeated_tool_loop(&msgs2);
        assert_eq!(cnt2, 1);
        assert_eq!(sigs2.len(), 1);

        let msgs3 = vec![
            make_turn(r#"{"command": "ls -la"}"#),
            make_turn(r#"{"command":"ls -la"}"#),
            make_turn(r#"{"command": "ls -la"}"#),
        ];
        let (cnt3, sigs3) = detect_repeated_tool_loop(&msgs3);
        assert_eq!(cnt3, 2);
        assert!(sigs3.contains(&canonical_tool_sig("exec", r#"{"command":"ls -la"}"#)));

        let msgs_alt = vec![
            make_turn(r#"{"command": "print_ok"}"#),
            make_turn(r#"{"command": "read_file"}"#),
            make_turn(r#"{"command": "print_ok"}"#),
            make_turn(r#"{"command": "read_file"}"#),
        ];
        let (cnt_alt, sigs_alt) = detect_repeated_tool_loop(&msgs_alt);
        assert_eq!(cnt_alt, 1);
        assert!(sigs_alt.contains(&canonical_tool_sig("exec", r#"{"command":"print_ok"}"#)));
        assert!(sigs_alt.contains(&canonical_tool_sig("exec", r#"{"command":"read_file"}"#)));

        let mut prompt = String::from("<|begin_of_text|><|message_model|><|content_text|>");
        let mut samp = grammar::SamplingParams::greedy();
        apply_tool_loop_guard(&mut prompt, &mut samp, cnt3);
        assert!(samp.temperature >= 0.6);
        assert!(samp.repetition_penalty >= 1.12 && samp.repetition_penalty <= 1.15);
        assert!(prompt.contains("blocked"));

        let mut after_plain = vec![
            make_turn(r#"{"command": "ls -la"}"#),
            make_turn(r#"{"command":"ls -la"}"#),
            make_turn(r#"{"command": "ls -la"}"#),
            ChatMessage {
                role: "assistant".into(),
                content: json!("Here is the final story."),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            },
        ];
        assert_eq!(detect_repeated_tool_loop(&after_plain), (0, Vec::new()));
        after_plain.push(ChatMessage {
            role: "user".into(),
            content: json!("Continue with chapter 2."),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        });
        assert_eq!(detect_repeated_tool_loop(&after_plain), (0, Vec::new()));
    }
}
