// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::Parser;
use hyphae_native_daemon::{NativeDaemon, NativeDaemonConfig};
use hyphae_native_product::{NativeProduct, NativeProductService, NativeProductServiceConfig};

#[derive(Debug, Parser)]
#[command(name = "celiums-hyphae-sidecar", version, about)]
struct Args {
    #[arg(long, env = "CELIUMS_HYPHAE_DATA_DIR")]
    data_dir: PathBuf,
    #[arg(long, env = "CELIUMS_HYPHAE_ENDPOINT")]
    endpoint: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let parent = PathBuf::from(&args.endpoint).parent().map(PathBuf::from);
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent)?;
    }
    let service = NativeProductService::start(
        NativeProduct::open(&args.data_dir)?,
        NativeProductServiceConfig::default(),
    )?;
    let daemon = NativeDaemon::start_with_service_authenticated(
        service,
        args.endpoint,
        NativeDaemonConfig::default(),
    )?;
    eprintln!("Celiums Hyphae sidecar listening on {}", daemon.endpoint());
    tokio::signal::ctrl_c().await?;
    daemon.shutdown().await?;
    Ok(())
}
