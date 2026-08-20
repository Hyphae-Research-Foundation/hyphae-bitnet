// SPDX-License-Identifier: Apache-2.0

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    Unauthorized(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Unavailable(String),
    #[error("{0}")]
    Internal(String),
}

impl GatewayError {
    pub fn internal(error: impl std::fmt::Display) -> Self {
        Self::Internal(error.to_string())
    }

    pub fn unavailable(error: impl std::fmt::Display) -> Self {
        Self::Unavailable(error.to_string())
    }

    pub fn bad_request(error: impl std::fmt::Display) -> Self {
        Self::BadRequest(error.to_string())
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, kind) = match self {
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
            Self::NotFound(_) => (StatusCode::NOT_FOUND, "not_found_error"),
            Self::Conflict(_) => (StatusCode::CONFLICT, "conflict_error"),
            Self::Unavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "unavailable_error"),
            Self::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        let message = self.to_string();
        (
            status,
            Json(json!({
                "error": {
                    "message": message,
                    "type": kind,
                    "code": status.as_u16()
                }
            })),
        )
            .into_response()
    }
}

impl From<hyphae_client::v2::ClientError> for GatewayError {
    fn from(error: hyphae_client::v2::ClientError) -> Self {
        if let hyphae_client::v2::ClientError::Product(product) = error {
            use hyphae_native_product::{ProductErrorCategory, ProductErrorCode};
            let message = format!("Hyphae operation failed: {}", product.message());
            return match product.code() {
                ProductErrorCode::ObjectNotFound | ProductErrorCode::CatalogObjectNotFound => {
                    Self::NotFound(message)
                }
                ProductErrorCode::AuthorizationDenied => Self::Unauthorized(message),
                ProductErrorCode::CatalogConflict
                | ProductErrorCode::IdempotencyConflict
                | ProductErrorCode::WriteConflict
                | ProductErrorCode::SqlUniqueViolation => Self::Conflict(message),
                ProductErrorCode::LimitExceeded => Self::BadRequest(message),
                _ => match product.category() {
                    ProductErrorCategory::InvalidRequest | ProductErrorCategory::Limit => {
                        Self::BadRequest(message)
                    }
                    ProductErrorCategory::NotFound => Self::NotFound(message),
                    ProductErrorCategory::Conflict => Self::Conflict(message),
                    ProductErrorCategory::Authorization => Self::Unauthorized(message),
                    ProductErrorCategory::Unavailable | ProductErrorCategory::Deadline => {
                        Self::Unavailable(message)
                    }
                    _ => Self::Internal(message),
                },
            };
        }
        Self::Unavailable(format!("Hyphae operation failed: {error}"))
    }
}

impl From<reqwest::Error> for GatewayError {
    fn from(error: reqwest::Error) -> Self {
        Self::Unavailable(format!("upstream request failed: {error}"))
    }
}

impl From<std::io::Error> for GatewayError {
    fn from(error: std::io::Error) -> Self {
        Self::Internal(error.to_string())
    }
}
