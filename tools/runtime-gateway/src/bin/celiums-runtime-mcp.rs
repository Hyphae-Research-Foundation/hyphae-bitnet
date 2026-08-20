// SPDX-License-Identifier: Apache-2.0

use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

use celiums_runtime_gateway::read_secret_file;
use clap::Parser;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::Mutex,
};

#[derive(Debug, Parser)]
#[command(name = "celiums-runtime-mcp", version, about)]
struct Args {
    #[arg(
        long,
        env = "CELIUMS_GATEWAY_URL",
        default_value = "http://127.0.0.1:8090"
    )]
    gateway_url: String,
    #[arg(long, env = "CELIUMS_GATEWAY_API_KEY_FILE")]
    api_key_file: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    #[serde(default)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

struct McpState {
    client: reqwest::Client,
    base_url: Url,
    api_key: Option<String>,
    initialized: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut base_url = Url::parse(&args.gateway_url)?;
    if base_url.scheme() != "http"
        || !matches!(base_url.host_str(), Some("127.0.0.1" | "::1" | "localhost"))
        || !base_url.username().is_empty()
        || base_url.password().is_some()
    {
        return Err("gateway URL must use HTTP loopback".into());
    }
    if !base_url.path().ends_with('/') {
        base_url.set_path(&format!("{}/", base_url.path()));
    }
    let api_key = args
        .api_key_file
        .as_deref()
        .map(read_secret_file)
        .transpose()?;
    let state = Arc::new(Mutex::new(McpState {
        client: reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_mins(15))
            .build()?,
        base_url,
        api_key,
        initialized: false,
    }));
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<RpcRequest>(&line) {
            Ok(request) => dispatch(&state, request).await,
            Err(error) => Some(RpcResponse {
                jsonrpc: "2.0",
                id: Value::Null,
                result: None,
                error: Some(json!({"code": -32700, "message": error.to_string()})),
            }),
        };
        if let Some(response) = response {
            stdout
                .write_all(serde_json::to_string(&response)?.as_bytes())
                .await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

async fn dispatch(state: &Arc<Mutex<McpState>>, request: RpcRequest) -> Option<RpcResponse> {
    if request.jsonrpc != "2.0" {
        return response_error(request.id, -32600, "jsonrpc must be 2.0");
    }
    let id = request.id.clone();
    match request.method.as_str() {
        "initialize" => {
            state.lock().await.initialized = true;
            response_ok(
                id,
                json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "celiums-runtime-mcp", "version": env!("CARGO_PKG_VERSION")}
                }),
            )
        }
        "notifications/initialized" | "notifications/cancelled" => None,
        "ping" => response_ok(id, json!({})),
        "tools/list" => {
            if !state.lock().await.initialized {
                return response_error(id, -32002, "server not initialized");
            }
            response_ok(id, json!({"tools": tools()}))
        }
        "tools/call" => {
            if !state.lock().await.initialized {
                return response_error(id, -32002, "server not initialized");
            }
            let name = request
                .params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arguments = request
                .params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match call_tool(state, name, arguments).await {
                Ok(value) => response_ok(
                    id,
                    json!({"content": [{"type": "text", "text": serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string())}], "structuredContent": value, "isError": false}),
                ),
                Err(error) => response_ok(
                    id,
                    json!({"content": [{"type": "text", "text": error}], "isError": true}),
                ),
            }
        }
        _ => response_error(id, -32601, "method not found"),
    }
}

async fn call_tool(
    state: &Arc<Mutex<McpState>>,
    name: &str,
    arguments: Value,
) -> Result<Value, String> {
    let (method, path, body) = match name {
        "celiums_retrieve_context" => {
            (reqwest::Method::POST, "/v1/retrieval".to_owned(), arguments)
        }
        "celiums_rag_answer" => {
            let query = arguments
                .get("query")
                .and_then(Value::as_str)
                .ok_or("query is required")?;
            let rag = arguments.get("rag").cloned().unwrap_or_else(|| json!({}));
            (
                reqwest::Method::POST,
                "/v1/chat/completions".to_owned(),
                json!({"messages": [{"role": "user", "content": query}], "temperature": arguments.get("temperature").cloned().unwrap_or(json!(0.0)), "max_tokens": arguments.get("max_tokens").cloned().unwrap_or(json!(256)), "rag": rag}),
            )
        }
        "celiums_memory_write" => {
            let mut body = arguments;
            body["kind"] = json!("memory");
            (reqwest::Method::PUT, "/v1/knowledge".to_owned(), body)
        }
        "celiums_memory_search" => {
            let mut body = arguments;
            body["kind"] = json!("memory");
            (reqwest::Method::POST, "/v1/retrieval".to_owned(), body)
        }
        "celiums_answer_verify" => {
            let receipt = arguments
                .get("receipt_id")
                .and_then(Value::as_str)
                .ok_or("receipt_id is required")?;
            (
                reqwest::Method::POST,
                format!("/v1/receipts/{}/verify", encode_path_segment(receipt)),
                Value::Null,
            )
        }
        _ => return Err(format!("unknown tool: {name}")),
    };
    let state = state.lock().await;
    let url = state
        .base_url
        .join(path.trim_start_matches('/'))
        .map_err(|error| error.to_string())?;
    let mut request = state.client.request(method, url);
    if let Some(api_key) = &state.api_key {
        request = request.bearer_auth(api_key);
    }
    if !body.is_null() {
        request = request.json(&body);
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    let status = response.status();
    let value: Value = response.json().await.map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(value.to_string());
    }
    Ok(value)
}

fn tools() -> Vec<Value> {
    let mut common = BTreeMap::new();
    common.insert("query", json!({"type": "string", "minLength": 1}));
    common.insert("scope", json!({"type": "string", "default": "default"}));
    common.insert(
        "top_k",
        json!({"type": "integer", "minimum": 1, "maximum": 64, "default": 5}),
    );
    common.insert("kind", json!({"type": "string", "default": "document"}));
    common.insert(
        "candidate_limit",
        json!({"type": "integer", "minimum": 1, "maximum": 10000, "default": 64}),
    );
    common.insert("mode", json!({"type": "string", "enum": ["lexical", "exact", "ann", "hybrid_exact", "hybrid_ann"], "default": "lexical"}));
    common.insert("proof", json!({"type": "boolean", "default": false}));
    vec![
        tool(
            "celiums_retrieve_context",
            "Retrieve snapshot-bound local evidence without generation.",
            common.clone(),
            &["query"],
        ),
        tool(
            "celiums_rag_answer",
            "Answer with Celiums BitNet using local Hyphae retrieval.",
            BTreeMap::from([
                ("query", json!({"type": "string", "minLength": 1})),
                ("rag", json!({"type": "object"})),
                (
                    "max_tokens",
                    json!({"type": "integer", "minimum": 1, "maximum": 4096}),
                ),
            ]),
            &["query"],
        ),
        tool(
            "celiums_memory_write",
            "Persist one scoped memory item.",
            BTreeMap::from([
                ("external_id", json!({"type": "string"})),
                ("scope", json!({"type": "string"})),
                ("title", json!({"type": "string"})),
                ("content", json!({"type": "string", "minLength": 1})),
                ("metadata", json!({"type": "object"})),
                (
                    "kind",
                    json!({"type": "string", "const": "memory", "default": "memory"}),
                ),
            ]),
            &["content"],
        ),
        tool(
            "celiums_memory_search",
            "Search scoped persistent memory.",
            common,
            &["query"],
        ),
        tool(
            "celiums_answer_verify",
            "Read the durable receipt for one generated answer.",
            BTreeMap::from([("receipt_id", json!({"type": "string", "minLength": 1}))]),
            &["receipt_id"],
        ),
    ]
}

fn tool(
    name: &str,
    description: &str,
    properties: BTreeMap<&str, Value>,
    required: &[&str],
) -> Value {
    json!({"name": name, "description": description, "inputSchema": {"type": "object", "properties": properties, "required": required, "additionalProperties": false}})
}

fn encode_path_segment(value: &str) -> String {
    value
        .bytes()
        .flat_map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
                vec![char::from(byte)].into_iter()
            } else {
                format!("%{byte:02X}")
                    .chars()
                    .collect::<Vec<_>>()
                    .into_iter()
            }
        })
        .collect()
}

fn response_ok(id: Option<Value>, result: Value) -> Option<RpcResponse> {
    id.map(|id| RpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    })
}
fn response_error(id: Option<Value>, code: i64, message: &str) -> Option<RpcResponse> {
    Some(RpcResponse {
        jsonrpc: "2.0",
        id: id.unwrap_or(Value::Null),
        result: None,
        error: Some(json!({"code": code, "message": message})),
    })
}
