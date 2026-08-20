// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::error::GatewayError;

#[derive(Clone, Debug)]
pub enum EmbeddingProvider {
    Disabled,
    Http(HttpEmbeddingProvider),
}

#[derive(Clone, Debug)]
pub struct HttpEmbeddingProvider {
    client: reqwest::Client,
    endpoint: Url,
    api_key: Option<String>,
    model: String,
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    input: &'a str,
    model: &'a str,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingProvider {
    pub fn disabled() -> Self {
        Self::Disabled
    }

    pub fn http(
        endpoint: &str,
        api_key: Option<String>,
        model: String,
        dimension: usize,
        timeout: Duration,
    ) -> Result<Self, GatewayError> {
        let endpoint = Url::parse(endpoint).map_err(|error| {
            GatewayError::bad_request(format!("invalid embedding URL: {error}"))
        })?;
        if !is_loopback_url(&endpoint) {
            return Err(GatewayError::bad_request(
                "embedding URL must resolve to a literal loopback address",
            ));
        }
        if dimension == 0 || dimension > u16::MAX.into() {
            return Err(GatewayError::bad_request(
                "embedding dimension must be between 1 and 65535",
            ));
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(timeout)
            .build()
            .map_err(GatewayError::internal)?;
        Ok(Self::Http(HttpEmbeddingProvider {
            client,
            endpoint,
            api_key,
            model,
            dimension,
        }))
    }

    pub const fn dimension(&self) -> Option<usize> {
        match self {
            Self::Disabled => None,
            Self::Http(provider) => Some(provider.dimension),
        }
    }

    pub fn model(&self) -> Option<&str> {
        match self {
            Self::Disabled => None,
            Self::Http(provider) => Some(&provider.model),
        }
    }

    pub async fn embed(&self, input: &str) -> Result<Option<Vec<f32>>, GatewayError> {
        let Self::Http(provider) = self else {
            return Ok(None);
        };
        if input.is_empty() {
            return Err(GatewayError::bad_request("embedding input is empty"));
        }
        let mut request = provider
            .client
            .post(provider.endpoint.clone())
            .json(&EmbeddingRequest {
                input,
                model: &provider.model,
            });
        if let Some(key) = &provider.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await?.error_for_status()?;
        let payload: EmbeddingResponse = response.json().await?;
        let embedding = payload
            .data
            .into_iter()
            .next()
            .ok_or_else(|| {
                GatewayError::Unavailable("embedding response contained no data".into())
            })?
            .embedding;
        if embedding.len() != provider.dimension {
            return Err(GatewayError::Unavailable(format!(
                "embedding dimension mismatch: expected {}, received {}",
                provider.dimension,
                embedding.len()
            )));
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(GatewayError::Unavailable(
                "embedding contains a non-finite value".into(),
            ));
        }
        Ok(Some(embedding))
    }
}

fn is_loopback_url(url: &Url) -> bool {
    matches!(url.host_str(), Some("127.0.0.1" | "::1" | "localhost"))
        && matches!(url.scheme(), "http" | "https")
        && url.username().is_empty()
        && url.password().is_none()
}

#[cfg(test)]
mod tests {
    use super::is_loopback_url;

    #[test]
    fn only_loopback_embedding_urls_are_accepted() {
        let accepted_ipv4 = "http://127.0.0.1:8081/v1/embeddings".parse();
        let accepted_name = "http://localhost:8081/v1/embeddings".parse();
        let rejected_spoof = "http://127.invalid/v1/embeddings".parse();
        let rejected_remote = "https://example.com/v1/embeddings".parse();
        assert!(accepted_ipv4.is_ok_and(|url| is_loopback_url(&url)));
        assert!(accepted_name.is_ok_and(|url| is_loopback_url(&url)));
        assert!(rejected_spoof.is_ok_and(|url| !is_loopback_url(&url)));
        assert!(rejected_remote.is_ok_and(|url| !is_loopback_url(&url)));
    }
}
