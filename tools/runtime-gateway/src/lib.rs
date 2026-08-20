// SPDX-License-Identifier: Apache-2.0

pub mod api;
pub mod bitnet;
pub mod embeddings;
pub mod error;
pub mod init;
pub mod model;
pub mod store;

pub const HYPHAE_VERSION: &str = "1.2.2";
pub const HYPHAE_COMMIT: &str = "0471ae25b263fd506da1578068ec57429a6783de";
pub const GATEWAY_SCHEMA_VERSION: &str = "1";
pub const SEARCH_DATABASE_ID: u128 = 100_001;
pub const SEARCH_SCHEMA_ID: u128 = 100_002;
pub const SEARCH_ANALYZER_ID: u128 = 100_003;
pub const SEARCH_COLLECTION_ID: u128 = 100_004;

pub fn read_secret_file(path: &std::path::Path) -> Result<String, error::GatewayError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(error::GatewayError::BadRequest(format!(
            "secret path is not a regular file: {}",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(error::GatewayError::BadRequest(format!(
                "secret file must not be accessible by group or others: {}",
                path.display()
            )));
        }
    }
    let secret = std::fs::read_to_string(path)?
        .trim_end_matches(['\r', '\n'])
        .to_owned();
    if secret.is_empty() {
        return Err(error::GatewayError::BadRequest(format!(
            "secret file is empty: {}",
            path.display()
        )));
    }
    Ok(secret)
}
