// SPDX-License-Identifier: Apache-2.0

use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use clap::{Parser, Subcommand};
use serde_json::{Value, json};

use celiums_runtime_gateway::{
    HYPHAE_COMMIT, HYPHAE_VERSION,
    api::{GatewayState, serve},
    bitnet::BitnetClient,
    embeddings::EmbeddingProvider,
    error::GatewayError,
    init::initialize,
    read_secret_file,
    store::{HyphaeStore, sha256_hex},
};

#[derive(Debug, Parser)]
#[command(name = "celiums-runtime-gateway", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Initialize an exclusively owned Hyphae directory and owner key.
    Init {
        #[arg(long, env = "CELIUMS_HYPHAE_DATA_DIR")]
        data_dir: PathBuf,
        #[arg(long, env = "CELIUMS_HYPHAE_KEY_FILE")]
        owner_key: PathBuf,
        #[arg(long, env = "CELIUMS_HYPHAE_GATEWAY_KEY_FILE")]
        gateway_key: PathBuf,
        #[arg(long)]
        embedding_dimension: Option<u16>,
    },
    /// Serve the HTTP RAG, memory, evidence, and registry API.
    Serve(ServeArgs),
    /// Verify a retained Hyphae proof and witness offline.
    Verify {
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        anchor: PathBuf,
    },
    /// Print exact gateway dependency provenance.
    Version,
}

#[derive(Debug, clap::Args)]
struct ServeArgs {
    #[arg(long, env = "CELIUMS_GATEWAY_BIND", default_value = "127.0.0.1:8090")]
    bind: SocketAddr,
    #[arg(long, env = "CELIUMS_HYPHAE_ENDPOINT")]
    hyphae_endpoint: String,
    #[arg(long, env = "CELIUMS_HYPHAE_KEY_FILE")]
    hyphae_key_file: PathBuf,
    #[arg(long, env = "CELIUMS_GATEWAY_API_KEY_FILE")]
    api_key_file: Option<PathBuf>,
    #[arg(
        long,
        env = "CELIUMS_BITNET_URL",
        default_value = "http://127.0.0.1:8080"
    )]
    bitnet_url: String,
    #[arg(long, env = "CELIUMS_BITNET_API_KEY_FILE")]
    bitnet_api_key_file: Option<PathBuf>,
    #[arg(long, env = "CELIUMS_PROOF_DIR", default_value = "./celiums-proofs")]
    proof_dir: PathBuf,
    #[arg(long, env = "CELIUMS_GATEWAY_LOCK_FILE")]
    lock_file: Option<PathBuf>,
    #[arg(long, env = "CELIUMS_EMBEDDING_URL")]
    embedding_url: Option<String>,
    #[arg(long, env = "CELIUMS_EMBEDDING_API_KEY_FILE")]
    embedding_api_key_file: Option<PathBuf>,
    #[arg(long, env = "CELIUMS_EMBEDDING_MODEL", default_value = "celiums-embed")]
    embedding_model: String,
    #[arg(long, env = "CELIUMS_EMBEDDING_DIMENSION")]
    embedding_dimension: Option<usize>,
    #[arg(long, env = "CELIUMS_INITIALIZED_EMBEDDING_DIMENSION")]
    initialized_embedding_dimension: Option<usize>,
    #[arg(long, env = "CELIUMS_MODEL_ID", default_value = "celiums-bitnet")]
    model_id: String,
    #[arg(long, env = "CELIUMS_MODEL_PATH")]
    model_path: Option<PathBuf>,
    #[arg(long, env = "CELIUMS_MAX_GENERATIONS", default_value_t = 2)]
    maximum_generations: usize,
    #[arg(long, env = "CELIUMS_MAX_HYPHAE_REQUESTS", default_value_t = 8)]
    maximum_hyphae_requests: usize,
}

#[tokio::main]
async fn main() -> Result<(), GatewayError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "celiums_runtime_gateway=info,tower_http=info".into()),
        )
        .json()
        .init();
    match Cli::parse().command {
        Command::Init {
            data_dir,
            owner_key,
            gateway_key,
            embedding_dimension,
        } => {
            initialize(
                &data_dir,
                &owner_key,
                &gateway_key,
                embedding_dimension,
                true,
            )?;
            println!("initialized {}", data_dir.display());
        }
        Command::Serve(args) => serve_command(args).await?,
        Command::Verify {
            proof,
            witness,
            anchor,
        } => verify(&proof, &witness, &anchor)?,
        Command::Version => println!(
            "Celiums Runtime Gateway {}\nHyphae {} ({})",
            env!("CARGO_PKG_VERSION"),
            HYPHAE_VERSION,
            HYPHAE_COMMIT
        ),
    }
    Ok(())
}

async fn serve_command(args: ServeArgs) -> Result<(), GatewayError> {
    let _writer_lock = acquire_writer_lock(
        args.lock_file
            .unwrap_or_else(|| default_lock_file(&args.hyphae_endpoint)),
    )?;
    let hyphae_key = read_secret_file(&args.hyphae_key_file)?;
    let api_key = args
        .api_key_file
        .as_deref()
        .map(read_secret_file)
        .transpose()?;
    let bitnet_key = args
        .bitnet_api_key_file
        .as_deref()
        .map(read_secret_file)
        .transpose()?;
    let embedding_key = args
        .embedding_api_key_file
        .as_deref()
        .map(read_secret_file)
        .transpose()?;
    let embeddings = match (&args.embedding_url, args.embedding_dimension) {
        (None, None) => EmbeddingProvider::disabled(),
        (Some(url), Some(dimension)) => EmbeddingProvider::http(
            url,
            embedding_key,
            args.embedding_model.clone(),
            dimension,
            Duration::from_secs(30),
        )?,
        _ => {
            return Err(GatewayError::BadRequest(
                "embedding URL and dimension must be configured together".into(),
            ));
        }
    };
    if args.embedding_dimension != args.initialized_embedding_dimension {
        return Err(GatewayError::BadRequest(
            "configured embedding dimension must equal --initialized-embedding-dimension; use neither for a lexical-only directory".into(),
        ));
    }
    let store = HyphaeStore::connect(
        args.hyphae_endpoint,
        &hyphae_key,
        embeddings,
        args.proof_dir,
        args.maximum_hyphae_requests,
    )?;
    let bitnet = BitnetClient::new(&args.bitnet_url, bitnet_key, Duration::from_mins(15))?;
    let model_sha256 = args.model_path.as_deref().map(sha256_file).transpose()?;
    let model_provenance = json!({
        "id": args.model_id,
        "sha256": model_sha256,
        "path": args.model_path.map(|path| path.display().to_string()),
        "gateway_version": env!("CARGO_PKG_VERSION"),
        "hyphae_version": HYPHAE_VERSION,
        "hyphae_commit": HYPHAE_COMMIT,
    });
    serve(
        GatewayState {
            store,
            bitnet,
            api_key,
            model_provenance,
            maximum_generations: std::sync::Arc::new(tokio::sync::Semaphore::new(
                args.maximum_generations.max(1),
            )),
        },
        args.bind,
    )
    .await
}

fn default_lock_file(endpoint: &str) -> PathBuf {
    let endpoint = PathBuf::from(endpoint);
    let mut name = endpoint
        .file_name()
        .map_or_else(|| "hyphae".into(), std::ffi::OsStr::to_os_string);
    name.push(".celiums-gateway.lock");
    endpoint.with_file_name(name)
}

fn acquire_writer_lock(path: PathBuf) -> Result<std::fs::File, GatewayError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(&path).map_err(GatewayError::internal)?;
    file.try_lock().map_err(|error| {
        GatewayError::Conflict(format!(
            "another gateway owns the writer lock {}: {error}",
            path.display()
        ))
    })?;
    Ok(file)
}

fn verify(proof: &Path, witness: &Path, anchor: &Path) -> Result<(), GatewayError> {
    let proof_bytes = fs::read(proof)?;
    let witness_bytes = fs::read(witness)?;
    let anchor_bytes = fs::read(anchor)?;
    let anchor: [u8; 32] = anchor_bytes.try_into().map_err(|_| {
        GatewayError::BadRequest("anchor file must contain exactly 32 bytes".into())
    })?;
    let report = hyphae_native_product::proof::verify_native_proof_offline(
        &proof_bytes,
        &witness_bytes,
        hyphae_native_product::proof::ExternalTrustedAnchor::new(anchor),
        &hyphae_native_product::proof::NativeVerificationLimits::default(),
    )
    .map_err(GatewayError::bad_request)?;
    let value: Value = json!({
        "valid": true,
        "scope": format!("{:?}", report.scope),
        "kind": format!("{:?}", report.kind),
        "anchor_digest": hex::encode(report.anchor_digest),
        "proof_digest": hex::encode(report.proof_digest),
        "witness_digest": hex::encode(report.witness_digest),
        "request_digest": hex::encode(report.request_digest),
        "result_digest": hex::encode(report.result_digest),
        "evidence_digest": hex::encode(report.evidence_digest),
        "file_count": report.file_count,
        "directory_count": report.directory_count,
        "total_file_bytes": report.total_file_bytes,
        "semantic_reexecution_performed": report.semantic_reexecution_performed,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(GatewayError::internal)?
    );
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, GatewayError> {
    let bytes = fs::read(path)?;
    Ok(sha256_hex(&bytes))
}
