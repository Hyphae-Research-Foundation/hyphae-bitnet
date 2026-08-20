// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use futures_util::StreamExt;
use reqwest::{Response, Url};
use serde_json::Value;

use crate::error::GatewayError;

#[derive(Debug)]
pub struct UpstreamJson {
    pub status: reqwest::StatusCode,
    pub body: Value,
    pub runtime: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BitnetClient {
    client: reqwest::Client,
    base_url: Url,
    api_key: Option<String>,
}

impl BitnetClient {
    pub fn new(
        base_url: &str,
        api_key: Option<String>,
        timeout: Duration,
    ) -> Result<Self, GatewayError> {
        let mut base_url = Url::parse(base_url)
            .map_err(|error| GatewayError::bad_request(format!("invalid BitNet URL: {error}")))?;
        if !matches!(base_url.host_str(), Some("127.0.0.1" | "::1" | "localhost"))
            || base_url.scheme() != "http"
            || !base_url.username().is_empty()
            || base_url.password().is_some()
        {
            return Err(GatewayError::bad_request(
                "BitNet URL must be plain HTTP on a literal loopback address",
            ));
        }
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path()));
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(3))
            .timeout(timeout)
            .build()
            .map_err(GatewayError::internal)?;
        Ok(Self {
            client,
            base_url,
            api_key,
        })
    }

    pub async fn health(&self) -> Result<(), GatewayError> {
        self.request(reqwest::Method::GET, "health", None)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn models(&self) -> Result<Value, GatewayError> {
        Ok(self
            .request(reqwest::Method::GET, "v1/models", None)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn complete(&self, body: Value, chat: bool) -> Result<UpstreamJson, GatewayError> {
        let path = if chat {
            "v1/chat/completions"
        } else {
            "v1/completions"
        };
        let response = self
            .request(reqwest::Method::POST, path, Some(body))
            .send()
            .await?;
        let status = response.status();
        let runtime = response
            .headers()
            .get("x-celiums-bitnet-runtime")
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        Ok(UpstreamJson {
            status,
            body: response.json().await?,
            runtime,
        })
    }

    pub async fn stream(&self, body: Value, chat: bool) -> Result<Response, GatewayError> {
        let path = if chat {
            "v1/chat/completions"
        } else {
            "v1/completions"
        };
        Ok(self
            .request(reqwest::Method::POST, path, Some(body))
            .send()
            .await?
            .error_for_status()?)
    }

    pub async fn completion_text(response: Response) -> Result<String, GatewayError> {
        let mut stream = response.bytes_stream();
        let mut pending = String::new();
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            pending.push_str(&String::from_utf8_lossy(&chunk?));
            while let Some(end) = pending.find("\n\n") {
                let frame = pending.drain(..end + 2).collect::<String>();
                for line in frame.lines() {
                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };
                    if data == "[DONE]" {
                        continue;
                    }
                    let value: Value = serde_json::from_str(data).map_err(|error| {
                        GatewayError::Unavailable(format!("invalid BitNet SSE: {error}"))
                    })?;
                    if let Some(piece) = value["choices"][0]["delta"]["content"]
                        .as_str()
                        .or_else(|| value["choices"][0]["text"].as_str())
                    {
                        text.push_str(piece);
                    }
                }
            }
        }
        Ok(text)
    }

    fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> reqwest::RequestBuilder {
        let url = self
            .base_url
            .join(path)
            .unwrap_or_else(|_| self.base_url.clone());
        let mut request = self.client.request(method, url);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        request
    }
}
