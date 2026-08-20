// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::{
    net::SocketAddr,
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{
    Json, Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
    routing::{get, post},
};
use celiums_runtime_gateway::{
    api::{GatewayState, router},
    bitnet::BitnetClient,
    embeddings::EmbeddingProvider,
    init::initialize,
    model::{KnowledgeWrite, RetrievalMode, RetrievalRequest},
    store::HyphaeStore,
};
use hyphae_native_daemon::{NativeDaemon, NativeDaemonConfig};
use hyphae_native_product::{NativeProduct, NativeProductService, NativeProductServiceConfig};
use serde_json::json;
use tokio::sync::Mutex;
use tower::ServiceExt;

#[tokio::test]
async fn initializes_ingests_retrieves_proves_and_verifies()
-> Result<(), Box<dyn std::error::Error>> {
    let root = tempfile::tempdir()?;
    let data = root.path().join("data");
    let owner = root.path().join("owner.key");
    let gateway = root.path().join("gateway.key");
    let socket = root.path().join("hyphae.sock");
    let proofs = root.path().join("proofs");
    let upstream = FakeUpstream::start().await?;
    initialize(&data, &owner, &gateway, None, false)?;
    assert_private_file(&owner)?;
    assert_private_file(&gateway)?;
    let service = NativeProductService::start(
        NativeProduct::open(&data)?,
        NativeProductServiceConfig::default(),
    )?;
    let daemon = NativeDaemon::start_with_service_authenticated(
        service,
        socket.to_string_lossy(),
        NativeDaemonConfig::default(),
    )?;
    let key = std::fs::read_to_string(&gateway)?;
    let store = HyphaeStore::connect(
        socket.to_string_lossy().into_owned(),
        &key,
        EmbeddingProvider::disabled(),
        proofs,
        1,
    )?;
    store.health().await?;
    let first = store
        .upsert_knowledge(KnowledgeWrite {
            external_id: Some("architecture".into()),
            kind: "document".into(),
            scope: "default".into(),
            title: "Architecture".into(),
            content: "Hyphae stores local verifiable retrieval evidence for Celiums BitNet.".into(),
            index_text: None,
            metadata: json!({"source": "test"}),
            expires_at_micros: None,
        })
        .await?;
    let replay = store
        .upsert_knowledge(KnowledgeWrite {
            external_id: Some("architecture".into()),
            kind: "document".into(),
            scope: "default".into(),
            title: "Architecture".into(),
            content: "Hyphae stores local verifiable retrieval evidence for Celiums BitNet.".into(),
            index_text: None,
            metadata: json!({"source": "test"}),
            expires_at_micros: None,
        })
        .await?;
    assert_eq!(first.id, replay.id);
    let result = store
        .retrieve(RetrievalRequest {
            query: "verifiable retrieval evidence".into(),
            scope: "default".into(),
            kind: Some("document".into()),
            top_k: 5,
            candidate_limit: 64,
            mode: RetrievalMode::Lexical,
            proof: true,
        })
        .await?;
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].external_id, "architecture");
    assert!(result.proof.is_some());
    let proof = serde_json::to_value(&result.proof)?;
    store
        .save_receipt(
            "receipt-test",
            "request-test",
            &json!({"id": "test"}),
            &serde_json::to_value(&result)?,
            &json!({"response_sha256": "test"}),
            Some(&proof),
        )
        .await?;
    let verification = store.verify_receipt("receipt-test").await?;
    assert_eq!(verification["valid"], true);
    assert_eq!(verification["semantic_reexecution_performed"], true);
    store.delete_knowledge("architecture").await?;
    let empty = store
        .retrieve(RetrievalRequest {
            query: "evidence".into(),
            scope: "default".into(),
            kind: Some("document".into()),
            top_k: 5,
            candidate_limit: 64,
            mode: RetrievalMode::Lexical,
            proof: false,
        })
        .await?;
    assert!(empty.hits.is_empty());
    let recreated = store
        .upsert_knowledge(KnowledgeWrite {
            external_id: Some("architecture".into()),
            kind: "document".into(),
            scope: "default".into(),
            title: "Architecture".into(),
            content: "Hyphae stores local verifiable retrieval evidence for Celiums BitNet.".into(),
            index_text: None,
            metadata: json!({"source": "test"}),
            expires_at_micros: None,
        })
        .await?;
    assert_ne!(first.id, recreated.id);
    let proof_metadata = std::fs::metadata(&result.proof.as_ref().ok_or("proof")?.proof_path)?;
    assert_eq!(proof_metadata.permissions().mode() & 0o077, 0);
    let app = router(GatewayState {
        store: store.clone(),
        bitnet: BitnetClient::new(&upstream.origin, None, Duration::from_secs(5))?,
        api_key: None,
        model_provenance: json!({"id": "test-model", "sha256": "model-sha"}),
        maximum_generations: Arc::new(tokio::sync::Semaphore::new(1)),
    });
    let passthrough = json!({"prompt": "plain request", "max_tokens": 8});
    let first_response = request_json(
        &app,
        "/v1/completions",
        &passthrough,
        Some("http-idempotent"),
    )
    .await?;
    assert_eq!(first_response.0, StatusCode::OK);
    assert_eq!(first_response.1["choices"][0]["text"], "plain request");
    assert_eq!(upstream.completions.load(Ordering::SeqCst), 1);
    let replay_response = request_json(
        &app,
        "/v1/completions",
        &passthrough,
        Some("http-idempotent"),
    )
    .await?;
    assert_eq!(replay_response.0, StatusCode::OK);
    assert_eq!(replay_response.1, first_response.1);
    assert_eq!(upstream.completions.load(Ordering::SeqCst), 1);
    let conflicting_response = request_json(
        &app,
        "/v1/completions",
        &json!({"prompt": "different request", "max_tokens": 8}),
        Some("http-idempotent"),
    )
    .await?;
    assert_eq!(conflicting_response.0, StatusCode::CONFLICT);

    let rag_response = request_json(
        &app,
        "/v1/completions",
        &json!({"prompt": "What stores evidence?", "max_tokens": 8, "rag": {"enabled": true, "scope": "default"}}),
        Some("http-rag"),
    ).await?;
    assert_eq!(rag_response.0, StatusCode::OK);
    assert!(
        rag_response.1["choices"][0]["text"]
            .as_str()
            .is_some_and(|text| text.contains("untrusted evidence"))
    );
    assert!(rag_response.1["celiums"]["receipt_id"].is_string());
    store.delete_knowledge("architecture").await?;
    drop(store);
    tokio::time::timeout(Duration::from_secs(10), daemon.shutdown()).await??;
    verify_vector_modes(root.path(), &upstream.origin).await?;
    Ok(())
}

async fn verify_vector_modes(
    root: &Path,
    embedding_origin: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = root.join("vector-data");
    let owner = root.join("vector-owner.key");
    let gateway = root.join("vector-gateway.key");
    let socket = root.join("vector-hyphae.sock");
    initialize(&data, &owner, &gateway, Some(3), false)?;
    let service = NativeProductService::start(
        NativeProduct::open(&data)?,
        NativeProductServiceConfig::default(),
    )?;
    let daemon = NativeDaemon::start_with_service_authenticated(
        service,
        socket.to_string_lossy(),
        NativeDaemonConfig::default(),
    )?;
    let key = std::fs::read_to_string(&gateway)?;
    let store = HyphaeStore::connect(
        socket.to_string_lossy().into_owned(),
        &key,
        EmbeddingProvider::http(
            &format!("{embedding_origin}/v1/embeddings"),
            None,
            "test-embed".into(),
            3,
            Duration::from_secs(5),
        )?,
        root.join("vector-proofs"),
        1,
    )?;
    store
        .upsert_knowledge(KnowledgeWrite {
            external_id: Some("vector-document".into()),
            kind: "document".into(),
            scope: "default".into(),
            title: "Vector document".into(),
            content: "Semantic and hybrid retrieval use the configured embedding model.".into(),
            index_text: None,
            metadata: json!({}),
            expires_at_micros: None,
        })
        .await?;
    for mode in [
        RetrievalMode::Exact,
        RetrievalMode::Ann,
        RetrievalMode::HybridExact,
        RetrievalMode::HybridAnn,
    ] {
        let result = store
            .retrieve(RetrievalRequest {
                query: "semantic retrieval".into(),
                scope: "default".into(),
                kind: Some("document".into()),
                top_k: 5,
                candidate_limit: 64,
                mode,
                proof: false,
            })
            .await?;
        assert_eq!(result.hits.len(), 1, "mode {mode:?}");
    }
    drop(store);
    tokio::time::timeout(Duration::from_secs(10), daemon.shutdown()).await??;
    Ok(())
}

async fn request_json(
    app: &Router,
    path: &str,
    body: &serde_json::Value,
    idempotency_key: Option<&str>,
) -> Result<(StatusCode, serde_json::Value), Box<dyn std::error::Error>> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    if let Some(key) = idempotency_key {
        builder = builder.header("idempotency-key", key);
    }
    let response = app
        .clone()
        .oneshot(builder.body(Body::from(serde_json::to_vec(body)?))?)
        .await?;
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 8 * 1024 * 1024).await?;
    Ok((status, serde_json::from_slice(&bytes)?))
}

struct FakeUpstream {
    origin: String,
    completions: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeUpstream {
    async fn start() -> Result<Self, Box<dyn std::error::Error>> {
        let completions = Arc::new(AtomicUsize::new(0));
        let last = Arc::new(Mutex::new(serde_json::Value::Null));
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .route("/v1/models", get(|| async { Json(json!({"object": "list", "data": []})) }))
            .route("/v1/embeddings", post(|Json(_body): Json<serde_json::Value>| async { Json(json!({"data": [{"embedding": [1.0, 0.0, 0.0]}]})) }))
            .route("/v1/completions", post({
                let completions = completions.clone();
                let last = last.clone();
                move |Json(body): Json<serde_json::Value>| {
                    let completions = completions.clone();
                    let last = last.clone();
                    async move {
                        completions.fetch_add(1, Ordering::SeqCst);
                        *last.lock().await = body.clone();
                        Json(json!({"id": "fake", "object": "text_completion", "model": "fake", "choices": [{"index": 0, "text": body["prompt"].as_str().unwrap_or_default(), "finish_reason": "stop"}]}))
                    }
                }
            }))
            .route("/v1/chat/completions", post(|Json(_body): Json<serde_json::Value>| async { Json(json!({"id": "fake", "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})) }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address: SocketAddr = listener.local_addr()?;
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(Self {
            origin: format!("http://{address}"),
            completions,
            task,
        })
    }
}

impl Drop for FakeUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn assert_private_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::fs::metadata(path)?.permissions().mode();
    assert_eq!(mode & 0o077, 0);
    Ok(())
}
