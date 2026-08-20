// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::Arc,
};

use hyphae_client::v2::{ClientError, HyphaeClient, RequestOptions};
use hyphae_native_product::{
    ProductCommitOutcome, ProductDocValue, ProductDocument, ProductLexicalBranch, ProductOperation,
    ProductResponse, ProductSearchFilter, ProductSearchIngestBatch, ProductSearchOperator,
    ProductSearchRequest, ProductSqlResult, ProductValue, ProductVector, ProductVectorBranch,
    ProductVectorExecution, SnapshotIdentity,
};
use hyphae_native_types::ObjectId;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use uuid::Uuid;

use crate::{
    SEARCH_COLLECTION_ID,
    embeddings::EmbeddingProvider,
    error::GatewayError,
    init::now_micros,
    model::{
        ArtifactRegistration, ContaminationRegistration, DatasetRegistration,
        EvaluationRegistration, HardNegativeManifest, HardNegativeRequest, KnowledgeRecord,
        KnowledgeWrite, LineageRegistration, ProofReference, RetrievalHit, RetrievalMode,
        RetrievalRequest, RetrievalResponse, RunRegistration, SnapshotEvidence,
        VectorBranchEvidence,
    },
};

#[derive(Clone)]
pub struct HyphaeStore {
    client: HyphaeClient,
    embeddings: Arc<EmbeddingProvider>,
    proof_dir: PathBuf,
    permits: Arc<Semaphore>,
    writes: Arc<Mutex<()>>,
}

impl HyphaeStore {
    pub fn connect(
        endpoint: String,
        api_key: &str,
        embeddings: EmbeddingProvider,
        proof_dir: PathBuf,
        maximum_concurrency: usize,
    ) -> Result<Self, GatewayError> {
        let client = HyphaeClient::local_authenticated(endpoint, api_key)?;
        let proof_dir = prepare_private_directory(proof_dir)?;
        Ok(Self {
            client,
            embeddings: Arc::new(embeddings),
            proof_dir,
            permits: Arc::new(Semaphore::new(maximum_concurrency.max(1))),
            writes: Arc::new(Mutex::new(())),
        })
    }

    pub async fn health(&self) -> Result<Value, GatewayError> {
        let response = self
            .sql_read("SELECT id FROM celiums_documents LIMIT 1", vec![])
            .await?;
        let (_, _, snapshot) = sql_rows(response)?;
        Ok(json!({
            "status": "ok",
            "hyphae": "ok",
            "snapshot": snapshot_evidence(snapshot),
        }))
    }

    pub async fn upsert_knowledge(
        &self,
        input: KnowledgeWrite,
    ) -> Result<KnowledgeRecord, GatewayError> {
        validate_knowledge(&input)?;
        let _write = self.writes.lock().await;
        let now = now_micros();
        let external_id = input
            .external_id
            .clone()
            .unwrap_or_else(|| format!("doc-{}", Uuid::now_v7()));
        let index_text = input
            .index_text
            .clone()
            .unwrap_or_else(|| format!("{}\n{}", input.title, input.content));
        let embedding = self.embeddings.embed(&index_text).await?;
        let digest = sha256_hex(input.content.as_bytes());
        let metadata = canonical_json(&input.metadata)?;
        let revision = sha256_hex(
            format!(
                "{}\0{}\0{}\0{}\0{}\0{}\0{}",
                input.kind,
                input.scope,
                input.title,
                input.content,
                index_text,
                metadata,
                input
                    .expires_at_micros
                    .map_or_else(|| "null".to_owned(), |value| value.to_string())
            )
            .as_bytes(),
        );
        let current = self.get_knowledge_by_external_id(&external_id).await?;
        if let Some(current) = &current
            && current.kind == input.kind
            && current.scope == input.scope
            && current.title == input.title
            && current.content == input.content
            && current.index_text == index_text
            && current.metadata == input.metadata
            && current.expires_at_micros == input.expires_at_micros
            && current.state == "active"
        {
            return Ok(current.clone());
        }
        let same_pending_revision = current.as_ref().is_some_and(|current| {
            current.revision_sha256 == revision && current.state == "indexing"
        });
        let (id, created_at, incarnation, publication_id, update) = if let Some(current) = current {
            let publication_id = if same_pending_revision {
                current.publication_id
            } else {
                Uuid::now_v7().to_string()
            };
            (
                current.id,
                current.created_at_micros,
                current.incarnation,
                publication_id,
                true,
            )
        } else {
            (
                new_i64_id(),
                now,
                Uuid::now_v7().to_string(),
                Uuid::now_v7().to_string(),
                false,
            )
        };
        let embedding_json = embedding
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(GatewayError::internal)?;
        let state = "indexing";
        if same_pending_revision {
            // The SQL half already committed. Retry only the missing search/activation phases.
        } else if update {
            self.sql_mutation(
                "UPDATE celiums_documents SET kind = ?, scope = ?, title = ?, content = ?, index_text = ?, metadata_json = ?, embedding_json = ?, content_sha256 = ?, revision_sha256 = ?, publication_id = ?, state = ?, updated_at_micros = ?, expires_at_micros = ? WHERE id = ?",
                vec![
                    text(&input.kind), text(&input.scope), text(&input.title), text(&input.content),
                    text(&index_text), text(&metadata), nullable_text(embedding_json.clone()), text(&digest),
                    text(&revision), text(&publication_id), text(state), signed(now), nullable_signed(input.expires_at_micros), signed(id),
                ],
                stable_token(&format!("document-update:{incarnation}:{publication_id}")),
            ).await?;
        } else {
            self.sql_mutation(
                "INSERT INTO celiums_documents (id, external_id, kind, scope, title, content, index_text, metadata_json, embedding_json, content_sha256, revision_sha256, incarnation, publication_id, state, created_at_micros, updated_at_micros, expires_at_micros) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                vec![
                    signed(id), text(&external_id), text(&input.kind), text(&input.scope), text(&input.title),
                    text(&input.content), text(&index_text), text(&metadata), nullable_text(embedding_json.clone()),
                    text(&digest), text(&revision), text(&incarnation), text(&publication_id), text(state), signed(created_at), signed(now), nullable_signed(input.expires_at_micros),
                ],
                stable_token(&format!("document-insert:{incarnation}:{publication_id}")),
            ).await?;
        }

        let document = search_document(
            id,
            &input.kind,
            &input.scope,
            &index_text,
            &revision,
            embedding.as_deref(),
        )?;
        if update {
            let permit = self.permit().await?;
            let updated = self
                .client
                .search_document_update(
                    object_id(SEARCH_COLLECTION_ID)?,
                    hyphae_native_product::ProductSearchDocumentUpdate {
                        idempotency_id: stable_token(&format!(
                            "search-update:{incarnation}:{publication_id}"
                        )),
                        document: document.clone(),
                    },
                    self.options(true, None),
                )
                .await;
            if matches!(
                updated,
                Err(ClientError::Product(ref error))
                    if error.code() == hyphae_native_product::ProductErrorCode::ObjectNotFound
            ) {
                drop(permit);
                let _permit = self.permit().await?;
                self.client
                    .search_ingest(
                        object_id(SEARCH_COLLECTION_ID)?,
                        ProductSearchIngestBatch {
                            idempotency_id: stable_token(&format!(
                                "search-recover:{incarnation}:{publication_id}"
                            )),
                            documents: vec![document],
                        },
                        self.options(true, None),
                    )
                    .await?;
            } else {
                updated?;
            }
        } else {
            let _permit = self.permit().await?;
            self.client
                .search_ingest(
                    object_id(SEARCH_COLLECTION_ID)?,
                    ProductSearchIngestBatch {
                        idempotency_id: stable_token(&format!(
                            "search-insert:{incarnation}:{publication_id}"
                        )),
                        documents: vec![document],
                    },
                    self.options(true, None),
                )
                .await?;
        }
        self.sql_mutation(
            "UPDATE celiums_documents SET state = ? WHERE id = ?",
            vec![text("active"), signed(id)],
            stable_token(&format!("document-activate:{incarnation}:{publication_id}")),
        )
        .await?;
        Ok(KnowledgeRecord {
            id,
            external_id,
            kind: input.kind,
            scope: input.scope,
            title: input.title,
            content: input.content,
            index_text,
            metadata: input.metadata,
            embedding,
            content_sha256: digest,
            revision_sha256: revision,
            incarnation,
            publication_id,
            state: "active".to_owned(),
            created_at_micros: created_at,
            updated_at_micros: now,
            expires_at_micros: input.expires_at_micros,
        })
    }

    pub async fn delete_knowledge(&self, external_id: &str) -> Result<(), GatewayError> {
        let _write = self.writes.lock().await;
        let record = self
            .get_knowledge_by_external_id(external_id)
            .await?
            .ok_or_else(|| {
                GatewayError::NotFound(format!("knowledge item not found: {external_id}"))
            })?;
        let token = stable_token(&format!(
            "document-delete:{}:{}",
            record.incarnation, record.revision_sha256
        ));
        let permit = self.permit().await?;
        self.client
            .search_document_delete(
                object_id(SEARCH_COLLECTION_ID)?,
                hyphae_native_product::ProductSearchDocumentDelete {
                    idempotency_id: token,
                    object_id: object_id(
                        u128::try_from(record.id).map_err(GatewayError::internal)?,
                    )?,
                },
                self.options(true, None),
            )
            .await?;
        drop(permit);
        self.sql_mutation(
            "DELETE FROM celiums_documents WHERE id = ?",
            vec![signed(record.id)],
            token,
        )
        .await?;
        Ok(())
    }

    pub async fn get_knowledge_by_external_id(
        &self,
        external_id: &str,
    ) -> Result<Option<KnowledgeRecord>, GatewayError> {
        let response = self
            .sql_read(
                "SELECT id, external_id, kind, scope, title, content, index_text, metadata_json, embedding_json, content_sha256, revision_sha256, incarnation, publication_id, state, created_at_micros, updated_at_micros, expires_at_micros FROM celiums_documents WHERE external_id = ?",
                vec![text(external_id)],
            )
            .await?;
        let (_, rows, _) = sql_rows(response)?;
        rows.into_iter().next().map(decode_knowledge).transpose()
    }

    pub async fn retrieve(
        &self,
        request: RetrievalRequest,
    ) -> Result<RetrievalResponse, GatewayError> {
        validate_retrieval(&request)?;
        let vector = if matches!(request.mode, RetrievalMode::Lexical) {
            None
        } else {
            self.embeddings.embed(&request.query).await?
        };
        if !matches!(request.mode, RetrievalMode::Lexical) && vector.is_none() {
            return Err(GatewayError::BadRequest(
                "vector retrieval requested but no embedding provider is configured".into(),
            ));
        }
        let operation = search_operation(&request, vector.as_deref())?;
        let permit = self.permit().await?;
        let response = if request.proof {
            self.client
                .prove(operation, bounded_proof_limits(), self.options(false, None))
                .await?
        } else {
            self.client
                .execute(operation, self.options(false, None))
                .await?
        };
        drop(permit);
        let (search, proof) = match response {
            ProductResponse::IntegratedSearch(search) => (search, None),
            ProductResponse::Proven { response, artifact } => {
                let ProductResponse::IntegratedSearch(search) = *response else {
                    return Err(GatewayError::Internal(
                        "unexpected proven search response".into(),
                    ));
                };
                let reference = self.persist_proof(&request.query, *artifact)?;
                (search, Some(reference))
            }
            _ => {
                return Err(GatewayError::Internal(
                    "unexpected Hyphae search response".into(),
                ));
            }
        };
        let ids = search
            .hits
            .iter()
            .map(|hit| {
                i64::try_from(hit.object_id.get()).map_err(|_| {
                    GatewayError::Internal("document ID exceeds signed SQL range".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let records = self.fetch_knowledge(&ids).await?;
        let mut by_id = records
            .into_iter()
            .map(|record| (record.id, record))
            .collect::<BTreeMap<_, _>>();
        let mut hits = search
            .hits
            .iter()
            .filter_map(|hit| {
                let id = i64::try_from(hit.object_id.get()).ok()?;
                let record = by_id.remove(&id)?;
                let indexed_revision = hit.doc_values.get("revision");
                if record.state != "active"
                    || record
                        .expires_at_micros
                        .is_some_and(|expires| expires <= now_micros())
                    || indexed_revision
                        != Some(&ProductDocValue::String(record.revision_sha256.clone()))
                {
                    return None;
                }
                Some(RetrievalHit {
                    id,
                    external_id: record.external_id,
                    kind: record.kind,
                    scope: record.scope,
                    title: record.title,
                    content: record.content,
                    metadata: record.metadata,
                    content_sha256: record.content_sha256,
                    revision_sha256: record.revision_sha256,
                    score: hit.score,
                })
            })
            .collect::<Vec<_>>();
        hits.truncate(request.top_k);
        Ok(RetrievalResponse {
            query: request.query,
            mode: request.mode,
            snapshot: snapshot_evidence(search.snapshot),
            approximate: search.approximate,
            total_documents: search.total_documents,
            eligible_documents: search.eligible_documents,
            lexical_candidates: search.lexical_candidates,
            retrieval_candidates: search.retrieval_candidates,
            matched_candidates: search.matched_candidates,
            vector_branches: search
                .vector_branches
                .into_iter()
                .map(|branch| VectorBranchEvidence {
                    target: branch.target,
                    strategy: format!("{:?}", branch.strategy),
                    approximate: branch.approximate,
                    eligible_documents: branch.eligible_documents,
                    candidate_count: branch.candidate_count,
                    visited_nodes: branch.visited_nodes,
                    exact_reranked: branch.exact_reranked,
                })
                .collect(),
            hits,
            proof,
        })
    }

    pub async fn get_receipt(&self, id: &str) -> Result<Value, GatewayError> {
        let response = self.sql_read(
            "SELECT id, request_id, model_json, retrieval_json, generation_json, proof_json, created_at_micros FROM celiums_receipts WHERE id = ?",
            vec![text(id)],
        ).await?;
        let (_, rows, _) = sql_rows(response)?;
        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| GatewayError::NotFound(format!("receipt not found: {id}")))?;
        let [
            ProductValue::Text(id),
            ProductValue::Text(request_id),
            ProductValue::Text(model),
            ProductValue::Text(retrieval),
            ProductValue::Text(generation),
            proof,
            ProductValue::Signed(created),
        ] = row.as_slice()
        else {
            return Err(GatewayError::Internal("invalid receipt row".into()));
        };
        Ok(json!({
            "id": id,
            "request_id": request_id,
            "model": parse_json(model)?,
            "retrieval": parse_json(retrieval)?,
            "generation": parse_json(generation)?,
            "proof": optional_json(proof)?,
            "created_at_micros": created,
        }))
    }

    pub async fn verify_receipt(&self, id: &str) -> Result<Value, GatewayError> {
        let receipt = self.get_receipt(id).await?;
        let proof = receipt
            .get("proof")
            .and_then(Value::as_object)
            .ok_or_else(|| GatewayError::BadRequest(format!("receipt has no proof: {id}")))?;
        let proof_file = proof_path(proof, "proof_path")?;
        let witness_file = proof_path(proof, "witness_path")?;
        let anchor_file = proof_path(proof, "anchor_path")?;
        if [&proof_file, &witness_file, &anchor_file]
            .iter()
            .any(|path| !path.starts_with(&self.proof_dir))
        {
            return Err(GatewayError::Internal(
                "stored proof reference escapes the configured proof directory".into(),
            ));
        }
        let proof_bytes = tokio::fs::read(&proof_file).await?;
        let witness_bytes = tokio::fs::read(&witness_file).await?;
        let anchor_bytes = tokio::fs::read(&anchor_file).await?;
        let expected_proof = proof
            .get("proof_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| GatewayError::Internal("proof reference has no proof hash".into()))?;
        let expected_witness = proof
            .get("witness_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| GatewayError::Internal("proof reference has no witness hash".into()))?;
        let expected_anchor = proof
            .get("trusted_anchor")
            .and_then(Value::as_str)
            .ok_or_else(|| GatewayError::Internal("proof reference has no anchor digest".into()))?;
        if sha256_hex(&proof_bytes) != expected_proof
            || sha256_hex(&witness_bytes) != expected_witness
            || hex::encode(&anchor_bytes) != expected_anchor
        {
            return Err(GatewayError::BadRequest(
                "proof artifacts do not match their receipt hashes".into(),
            ));
        }
        let anchor: [u8; 32] = anchor_bytes
            .try_into()
            .map_err(|_| GatewayError::Internal("stored proof anchor is not 32 bytes".into()))?;
        let report = tokio::task::spawn_blocking(move || {
            hyphae_native_product::proof::verify_native_proof_offline(
                &proof_bytes,
                &witness_bytes,
                hyphae_native_product::proof::ExternalTrustedAnchor::new(anchor),
                &hyphae_native_product::proof::NativeVerificationLimits::default(),
            )
        })
        .await
        .map_err(GatewayError::internal)?
        .map_err(GatewayError::bad_request)?;
        let proof_request_digest = proof
            .get("request_digest")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                GatewayError::BadRequest("proof reference has no request digest".into())
            })?;
        let proof_result_digest = proof
            .get("result_digest")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                GatewayError::BadRequest("proof reference has no result digest".into())
            })?;
        let proof_evidence_digest = proof
            .get("evidence_digest")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                GatewayError::BadRequest("proof reference has no evidence digest".into())
            })?;
        if hex::encode(report.request_digest) != proof_request_digest
            || hex::encode(report.result_digest) != proof_result_digest
            || hex::encode(report.evidence_digest) != proof_evidence_digest
        {
            return Err(GatewayError::BadRequest(
                "proof semantic digests do not match the receipt".into(),
            ));
        }
        let retrieval = receipt
            .get("retrieval")
            .and_then(|value| value.as_object())
            .ok_or_else(|| GatewayError::BadRequest("receipt has no retrieval claim".into()))?;
        if retrieval
            .get("snapshot")
            .and_then(|value| value.get("root_digest"))
            .and_then(Value::as_str)
            .is_none()
            || retrieval
                .get("hits")
                .and_then(Value::as_array)
                .is_none_or(|hits| {
                    hits.iter().any(|hit| {
                        hit.get("revision_sha256").and_then(Value::as_str).is_none()
                            || hit.get("content_sha256").and_then(Value::as_str).is_none()
                    })
                })
        {
            return Err(GatewayError::BadRequest(
                "receipt retrieval claim lacks snapshot or revision binding".into(),
            ));
        }
        Ok(json!({
            "receipt_id": id,
            "valid": true,
            "scope": format!("{:?}", report.scope),
            "kind": format!("{:?}", report.kind),
            "anchor_digest": hex::encode(report.anchor_digest),
            "proof_digest": hex::encode(report.proof_digest),
            "witness_digest": hex::encode(report.witness_digest),
            "request_digest": hex::encode(report.request_digest),
            "result_digest": hex::encode(report.result_digest),
            "evidence_digest": hex::encode(report.evidence_digest),
            "semantic_reexecution_performed": report.semantic_reexecution_performed,
            "file_count": report.file_count,
            "total_file_bytes": report.total_file_bytes,
        }))
    }

    pub async fn save_receipt(
        &self,
        receipt_id: &str,
        request_id: &str,
        model: &Value,
        retrieval: &Value,
        generation: &Value,
        proof: Option<&Value>,
    ) -> Result<(), GatewayError> {
        self.sql_mutation(
            "INSERT INTO celiums_receipts (id, request_id, model_json, retrieval_json, generation_json, proof_json, created_at_micros) VALUES (?, ?, ?, ?, ?, ?, ?)",
            vec![
                text(receipt_id), text(request_id), text(&canonical_json(model)?),
                text(&canonical_json(retrieval)?), text(&canonical_json(generation)?),
                nullable_text(proof.map(canonical_json).transpose()?), signed(now_micros()),
            ],
            stable_token(&format!("receipt:{receipt_id}")),
        ).await?;
        Ok(())
    }

    pub async fn save_request_pending(
        &self,
        id: &str,
        kind: &str,
        digest: &str,
    ) -> Result<(), GatewayError> {
        if let Some(existing) = self.request_state(id).await? {
            if existing[0] != digest {
                return Err(GatewayError::Conflict(format!(
                    "idempotency key {id} is bound to a different request"
                )));
            }
            return Err(GatewayError::Conflict(format!(
                "idempotency key {id} already has state {}",
                existing[1]
            )));
        }
        let now = now_micros();
        self.sql_mutation(
            "INSERT INTO celiums_requests (id, request_sha256, kind, state, response_json, error, created_at_micros, updated_at_micros) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            vec![text(id), text(digest), text(kind), text("pending"), ProductValue::Null, ProductValue::Null, signed(now), signed(now)],
            stable_token(&format!("request:{id}:{digest}")),
        ).await?;
        Ok(())
    }

    pub async fn begin_request(
        &self,
        id: &str,
        kind: &str,
        digest: &str,
    ) -> Result<Option<Value>, GatewayError> {
        let Some(existing) = self.request_state(id).await? else {
            self.save_request_pending(id, kind, digest).await?;
            return Ok(None);
        };
        if existing[0] != digest {
            return Err(GatewayError::Conflict(format!(
                "idempotency key {id} is bound to a different request"
            )));
        }
        match existing[1].as_str() {
            "completed" if !existing[2].is_empty() => parse_json(&existing[2]).map(Some),
            "failed" => {
                self.sql_mutation(
                    "UPDATE celiums_requests SET state = ?, error = ?, updated_at_micros = ? WHERE id = ?",
                    vec![text("pending"), ProductValue::Null, signed(now_micros()), text(id)],
                    stable_token(&format!("request-retry:{id}:{}", now_micros())),
                )
                .await?;
                Ok(None)
            }
            state => Err(GatewayError::Conflict(format!(
                "idempotency key {id} already has state {state}"
            ))),
        }
    }

    pub async fn complete_request(&self, id: &str, response: &Value) -> Result<(), GatewayError> {
        self.sql_mutation(
            "UPDATE celiums_requests SET state = ?, response_json = ?, error = ?, updated_at_micros = ? WHERE id = ?",
            vec![text("completed"), text(&canonical_json(response)?), ProductValue::Null, signed(now_micros()), text(id)],
            stable_token(&format!("request-complete:{id}")),
        ).await?;
        Ok(())
    }

    pub async fn fail_request(&self, id: &str, error: &str) -> Result<(), GatewayError> {
        self.sql_mutation(
            "UPDATE celiums_requests SET state = ?, error = ?, updated_at_micros = ? WHERE id = ?",
            vec![text("failed"), text(error), signed(now_micros()), text(id)],
            stable_token(&format!(
                "request-failed:{id}:{}",
                sha256_hex(error.as_bytes())
            )),
        )
        .await?;
        Ok(())
    }

    pub async fn completed_request(
        &self,
        id: &str,
        digest: &str,
    ) -> Result<Option<Value>, GatewayError> {
        let Some(state) = self.request_state(id).await? else {
            return Ok(None);
        };
        if state[0] != digest {
            return Err(GatewayError::Conflict(format!(
                "idempotency key {id} is bound to a different request"
            )));
        }
        if state[1] != "completed" {
            return Ok(None);
        }
        if state[2].is_empty() {
            Ok(None)
        } else {
            parse_json(&state[2]).map(Some)
        }
    }

    pub async fn cache_lookup(
        &self,
        prompt: &str,
        embedding: Option<&[f32]>,
        model_id: &str,
        scope: &str,
        knowledge_digest: &str,
        policy_key: &str,
        similarity_threshold: f32,
    ) -> Result<Option<Value>, GatewayError> {
        if !similarity_threshold.is_finite() || !(0.0..=1.0).contains(&similarity_threshold) {
            return Err(GatewayError::BadRequest(
                "cache similarity must be finite and between 0 and 1".into(),
            ));
        }
        let exact_key = cache_key(prompt, model_id, scope, knowledge_digest, policy_key);
        let exact = self
            .sql_read(
                "SELECT response_json, expires_at_micros FROM celiums_cache WHERE cache_key = ?",
                vec![text(&exact_key)],
            )
            .await?;
        let (_, rows, _) = sql_rows(exact)?;
        if let Some(row) = rows.into_iter().next() {
            let [ProductValue::Text(response), ProductValue::Signed(expires)] = row.as_slice()
            else {
                return Err(GatewayError::Internal("invalid exact cache row".into()));
            };
            if *expires > now_micros() {
                return parse_json(response).map(Some);
            }
        }
        let Some(embedding) = embedding else {
            return Ok(None);
        };
        let response = self
            .sql_read(
                "SELECT embedding_json, response_json FROM celiums_cache WHERE scope = ? AND model_id = ? AND knowledge_digest = ? AND policy_key = ? AND embedding_model = ? AND expires_at_micros > ? LIMIT 1024",
                vec![text(scope), text(model_id), text(knowledge_digest), text(policy_key), text(self.embeddings.model().unwrap_or("disabled")), signed(now_micros())],
            )
            .await?;
        let (_, rows, _) = sql_rows(response)?;
        let mut best: Option<(f32, Value)> = None;
        for row in rows {
            let [ProductValue::Text(candidate), ProductValue::Text(response)] = row.as_slice()
            else {
                continue;
            };
            let Ok(candidate) = serde_json::from_str::<Vec<f32>>(candidate) else {
                continue;
            };
            let Ok(similarity) = cosine_similarity(embedding, &candidate) else {
                continue;
            };
            if similarity >= similarity_threshold
                && best.as_ref().is_none_or(|(score, _)| similarity > *score)
            {
                best = Some((similarity, parse_json(response)?));
            }
        }
        Ok(best.map(|(_, value)| value))
    }

    pub async fn cache_store(
        &self,
        prompt: &str,
        embedding: Option<&[f32]>,
        response: &Value,
        model_id: &str,
        scope: &str,
        knowledge_digest: &str,
        policy_key: &str,
        ttl_seconds: u64,
    ) -> Result<(), GatewayError> {
        let key = cache_key(prompt, model_id, scope, knowledge_digest, policy_key);
        let id = format!("cache-{}", &key[..32]);
        let created = now_micros();
        let ttl_micros = i64::try_from(ttl_seconds)
            .unwrap_or(i64::MAX)
            .saturating_mul(1_000_000);
        let parameters = vec![
            text(&key),
            text(policy_key),
            text(prompt),
            nullable_text(
                embedding
                    .map(serde_json::to_string)
                    .transpose()
                    .map_err(GatewayError::internal)?,
            ),
            text(self.embeddings.model().unwrap_or("disabled")),
            text(&canonical_json(response)?),
            text(model_id),
            text(scope),
            text(knowledge_digest),
            signed(created),
            signed(created.saturating_add(ttl_micros)),
            text(&id),
        ];
        let existing = self
            .sql_read("SELECT id FROM celiums_cache WHERE id = ?", vec![text(&id)])
            .await?;
        let (_, rows, _) = sql_rows(existing)?;
        if rows.is_empty() {
            self.sql_mutation(
                "INSERT INTO celiums_cache (cache_key, policy_key, prompt, embedding_json, embedding_model, response_json, model_id, scope, knowledge_digest, created_at_micros, expires_at_micros, id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                parameters,
                stable_token(&format!("semantic-cache-insert:{key}")),
            ).await?;
        } else {
            self.sql_mutation(
                "UPDATE celiums_cache SET cache_key = ?, policy_key = ?, prompt = ?, embedding_json = ?, embedding_model = ?, response_json = ?, model_id = ?, scope = ?, knowledge_digest = ?, created_at_micros = ?, expires_at_micros = ? WHERE id = ?",
                parameters,
                stable_token(&format!("semantic-cache-update:{key}:{created}")),
            ).await?;
        }
        Ok(())
    }

    pub async fn embed(&self, input: &str) -> Result<Option<Vec<f32>>, GatewayError> {
        self.embeddings.embed(input).await
    }

    async fn request_state(&self, id: &str) -> Result<Option<[String; 3]>, GatewayError> {
        let response = self
            .sql_read(
                "SELECT request_sha256, state, response_json FROM celiums_requests WHERE id = ?",
                vec![text(id)],
            )
            .await?;
        let (_, rows, _) = sql_rows(response)?;
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        let [
            ProductValue::Text(digest),
            ProductValue::Text(state),
            response,
        ] = row.as_slice()
        else {
            return Err(GatewayError::Internal("invalid request state row".into()));
        };
        Ok(Some([
            digest.clone(),
            state.clone(),
            optional_text(response)?.unwrap_or_default(),
        ]))
    }

    pub async fn register_artifact(&self, value: ArtifactRegistration) -> Result<(), GatewayError> {
        validate_sha256(&value.sha256)?;
        self.insert_registry(
            "INSERT INTO celiums_artifacts (id, kind, uri, sha256, size_bytes, media_type, metadata_json, created_at_micros) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            vec![text(&value.id), text(&value.kind), text(&value.uri), text(&value.sha256), signed(value.size_bytes), text(&value.media_type), text(&canonical_json(&value.metadata)?), signed(now_micros())],
            &format!("artifact:{}", value.id),
        ).await
    }

    pub async fn get_artifact(&self, id: &str) -> Result<Value, GatewayError> {
        self.registry_row(
            "SELECT id, kind, uri, sha256, size_bytes, media_type, metadata_json, created_at_micros FROM celiums_artifacts WHERE id = ?",
            id,
            &["id", "kind", "uri", "sha256", "size_bytes", "media_type", "metadata", "created_at_micros"],
            &[6],
        ).await
    }

    pub async fn register_dataset(&self, value: DatasetRegistration) -> Result<(), GatewayError> {
        validate_sha256(&value.manifest_sha256)?;
        let canonical = canonical_json(&value.manifest)?;
        if sha256_hex(canonical.as_bytes()) != value.manifest_sha256 {
            return Err(GatewayError::BadRequest(
                "dataset manifest SHA-256 does not match canonical JSON".into(),
            ));
        }
        self.insert_registry(
            "INSERT INTO celiums_datasets (id, name, manifest_sha256, manifest_json, license, created_at_micros) VALUES (?, ?, ?, ?, ?, ?)",
            vec![text(&value.id), text(&value.name), text(&value.manifest_sha256), text(&canonical), text(&value.license), signed(now_micros())],
            &format!("dataset:{}", value.id),
        ).await
    }

    pub async fn get_dataset(&self, id: &str) -> Result<Value, GatewayError> {
        self.registry_row(
            "SELECT id, name, manifest_sha256, manifest_json, license, created_at_micros FROM celiums_datasets WHERE id = ?",
            id,
            &["id", "name", "manifest_sha256", "manifest", "license", "created_at_micros"],
            &[3],
        ).await
    }

    pub async fn register_lineage(&self, value: LineageRegistration) -> Result<(), GatewayError> {
        self.insert_registry(
            "INSERT INTO celiums_lineage (id, source_id, target_id, transform, metadata_json, created_at_micros) VALUES (?, ?, ?, ?, ?, ?)",
            vec![text(&value.id), text(&value.source_id), text(&value.target_id), text(&value.transform), text(&canonical_json(&value.metadata)?), signed(now_micros())],
            &format!("lineage:{}", value.id),
        ).await
    }

    pub async fn register_run(&self, value: RunRegistration) -> Result<(), GatewayError> {
        self.insert_registry(
            "INSERT INTO celiums_runs (id, kind, state, dataset_id, model_id, seed, config_json, metrics_json, started_at_micros, completed_at_micros) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            vec![text(&value.id), text(&value.kind), text(&value.status), text(&value.dataset_id), text(&value.model_id), signed(value.seed), text(&canonical_json(&value.config)?), text(&canonical_json(&value.metrics)?), signed(value.started_at_micros), nullable_signed(value.completed_at_micros)],
            &format!("run:{}", value.id),
        ).await
    }

    pub async fn get_run(&self, id: &str) -> Result<Value, GatewayError> {
        self.registry_row(
            "SELECT id, kind, state, dataset_id, model_id, seed, config_json, metrics_json, started_at_micros, completed_at_micros FROM celiums_runs WHERE id = ?",
            id,
            &["id", "kind", "state", "dataset_id", "model_id", "seed", "config", "metrics", "started_at_micros", "completed_at_micros"],
            &[6, 7],
        ).await
    }

    pub async fn register_evaluation(
        &self,
        value: EvaluationRegistration,
    ) -> Result<(), GatewayError> {
        self.insert_registry(
            "INSERT INTO celiums_evaluations (id, run_id, suite, metric, value, evidence_json, created_at_micros) VALUES (?, ?, ?, ?, ?, ?, ?)",
            vec![text(&value.id), text(&value.run_id), text(&value.suite), text(&value.metric), text(&value.value), text(&canonical_json(&value.evidence)?), signed(now_micros())],
            &format!("evaluation:{}", value.id),
        ).await
    }

    pub async fn register_contamination(
        &self,
        value: ContaminationRegistration,
    ) -> Result<(), GatewayError> {
        self.insert_registry(
            "INSERT INTO celiums_contamination (id, dataset_id, benchmark_id, candidate_id, method, score, decision, evidence_json, created_at_micros) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            vec![text(&value.id), text(&value.dataset_id), text(&value.benchmark_id), text(&value.candidate_id), text(&value.method), text(&value.score), text(&value.decision), text(&canonical_json(&value.evidence)?), signed(now_micros())],
            &format!("contamination:{}", value.id),
        ).await
    }

    pub async fn hard_negatives(
        &self,
        request: HardNegativeRequest,
    ) -> Result<HardNegativeManifest, GatewayError> {
        if request.queries.is_empty()
            || request.queries.len() > 256
            || request.negatives_per_query == 0
            || request.negatives_per_query > 128
        {
            return Err(GatewayError::BadRequest(
                "hard-negative request is outside bounded limits".into(),
            ));
        }
        let mut rows = BTreeMap::new();
        let mut snapshots = BTreeMap::new();
        for query in &request.queries {
            let positive = query.positive_external_ids.iter().collect::<BTreeSet<_>>();
            let result = self
                .retrieve(RetrievalRequest {
                    query: query.text.clone(),
                    scope: request.scope.clone(),
                    kind: Some("document".into()),
                    top_k: (request.negatives_per_query + positive.len()).min(1_024),
                    candidate_limit: ((request.negatives_per_query + positive.len()) * 8)
                        .min(10_000),
                    mode: request.mode,
                    proof: false,
                })
                .await?;
            snapshots.insert(query.id.clone(), result.snapshot.clone());
            rows.insert(
                query.id.clone(),
                result
                    .hits
                    .into_iter()
                    .filter(|hit| !positive.contains(&hit.external_id))
                    .take(request.negatives_per_query)
                    .collect(),
            );
        }
        let created_at_micros = now_micros();
        let body = json!({"created_at_micros": created_at_micros, "mode": request.mode, "scope": request.scope, "rows": rows, "retrieval_snapshots": snapshots});
        let manifest_sha256 = sha256_hex(canonical_json(&body)?.as_bytes());
        Ok(HardNegativeManifest {
            schema: "celiums-hard-negatives-v1",
            created_at_micros,
            mode: request.mode,
            scope: request.scope,
            rows,
            retrieval_snapshots: snapshots,
            manifest_sha256,
        })
    }

    async fn insert_registry(
        &self,
        statement: &str,
        parameters: Vec<ProductValue>,
        token: &str,
    ) -> Result<(), GatewayError> {
        self.sql_mutation(statement, parameters, stable_token(token))
            .await?;
        Ok(())
    }

    async fn registry_row(
        &self,
        statement: &str,
        id: &str,
        fields: &[&str],
        json_fields: &[usize],
    ) -> Result<Value, GatewayError> {
        let response = self.sql_read(statement, vec![text(id)]).await?;
        let (_, rows, snapshot) = sql_rows(response)?;
        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| GatewayError::NotFound(format!("registry object not found: {id}")))?;
        if row.len() != fields.len() {
            return Err(GatewayError::Internal("invalid registry row".into()));
        }
        let mut object = serde_json::Map::new();
        for (index, (field, value)) in fields.iter().zip(&row).enumerate() {
            let value = if json_fields.contains(&index) {
                match value {
                    ProductValue::Text(value) => parse_json(value)?,
                    _ => return Err(GatewayError::Internal("invalid registry JSON field".into())),
                }
            } else {
                product_value_json(value)?
            };
            object.insert((*field).to_owned(), value);
        }
        object.insert(
            "snapshot".into(),
            serde_json::to_value(snapshot_evidence(snapshot)).map_err(GatewayError::internal)?,
        );
        Ok(Value::Object(object))
    }

    async fn fetch_knowledge(&self, ids: &[i64]) -> Result<Vec<KnowledgeRecord>, GatewayError> {
        let mut records = Vec::with_capacity(ids.len());
        for id in ids {
            let response = self.sql_read(
                "SELECT id, external_id, kind, scope, title, content, index_text, metadata_json, embedding_json, content_sha256, revision_sha256, incarnation, publication_id, state, created_at_micros, updated_at_micros, expires_at_micros FROM celiums_documents WHERE id = ?",
                vec![signed(*id)],
            ).await?;
            let (_, rows, _) = sql_rows(response)?;
            if let Some(row) = rows.into_iter().next() {
                records.push(decode_knowledge(row)?);
            }
        }
        Ok(records)
    }

    fn persist_proof(
        &self,
        query: &str,
        artifact: hyphae_native_product::proof::NativeOperationProofArtifact,
    ) -> Result<ProofReference, GatewayError> {
        let id = format!("{}-{}", now_micros(), &sha256_hex(query.as_bytes())[..16]);
        let proof_path = self.proof_dir.join(format!("{id}.hynproof"));
        let witness_path = self.proof_dir.join(format!("{id}.hynwit"));
        let anchor_path = self.proof_dir.join(format!("{id}.anchor"));
        let anchor = artifact.trusted_anchor.digest();
        write_private_file(&proof_path, &artifact.proof_bytes)?;
        write_private_file(&witness_path, &artifact.witness_bytes)?;
        write_private_file(&anchor_path, &anchor)?;
        Ok(ProofReference {
            proof_path: proof_path.display().to_string(),
            witness_path: witness_path.display().to_string(),
            anchor_path: anchor_path.display().to_string(),
            proof_sha256: sha256_hex(&artifact.proof_bytes),
            witness_sha256: sha256_hex(&artifact.witness_bytes),
            trusted_anchor: hex::encode(anchor),
            request_digest: hex::encode(artifact.proof.content().request.digest()),
            result_digest: hex::encode(artifact.proof.content().result.digest()),
            evidence_digest: hex::encode(artifact.proof.content().evidence.digest()),
        })
    }

    async fn sql_read(
        &self,
        statement: &str,
        parameters: Vec<ProductValue>,
    ) -> Result<ProductResponse, GatewayError> {
        let _permit = self.permit().await?;
        Ok(self
            .client
            .sql(statement, parameters, self.options(false, None))
            .await?)
    }

    async fn sql_mutation(
        &self,
        statement: &str,
        parameters: Vec<ProductValue>,
        token: u128,
    ) -> Result<ProductResponse, GatewayError> {
        let permit = self.permit().await?;
        let result = self
            .client
            .sql(statement, parameters, self.options(true, Some(token)))
            .await;
        let response = match result {
            Ok(response) => response,
            Err(ClientError::Product(error))
                if error.code() == hyphae_native_product::ProductErrorCode::UnknownCommit =>
            {
                drop(permit);
                let status = self
                    .client
                    .transaction_status_by_idempotency(token, self.options(false, None))
                    .await?;
                match status {
                    ProductResponse::TransactionStatus(
                        hyphae_native_product::ProductTransactionStatus::Committed(receipt),
                    ) => {
                        return Ok(ProductResponse::TransactionStatus(
                            hyphae_native_product::ProductTransactionStatus::Committed(receipt),
                        ));
                    }
                    ProductResponse::TransactionStatus(
                        hyphae_native_product::ProductTransactionStatus::RolledBack { .. },
                    ) => {
                        return Err(GatewayError::Unavailable(
                            "Hyphae mutation was rolled back".into(),
                        ));
                    }
                    _ => {
                        return Err(GatewayError::Unavailable(format!(
                            "Hyphae commit outcome remains unknown for transaction {:?}",
                            error.details().transaction_id()
                        )));
                    }
                }
            }
            Err(error) => return Err(error.into()),
        };
        if let ProductResponse::Sql {
            commit: Some(ProductCommitOutcome::OutcomeUnknown { transaction_id }),
            ..
        } = response
        {
            drop(permit);
            let status = self
                .client
                .transaction_status(transaction_id, self.options(false, None))
                .await?;
            return match status {
                ProductResponse::TransactionStatus(
                    hyphae_native_product::ProductTransactionStatus::Committed(receipt),
                ) => Ok(ProductResponse::TransactionStatus(
                    hyphae_native_product::ProductTransactionStatus::Committed(receipt),
                )),
                ProductResponse::TransactionStatus(
                    hyphae_native_product::ProductTransactionStatus::RolledBack { .. },
                ) => Err(GatewayError::Unavailable(
                    "Hyphae mutation was rolled back".into(),
                )),
                _ => Err(GatewayError::Unavailable(
                    "Hyphae commit outcome remains unknown".into(),
                )),
            };
        }
        Ok(response)
    }

    fn options(&self, mutation: bool, idempotency_token: Option<u128>) -> RequestOptions {
        let now = now_micros();
        RequestOptions {
            logical_time_micros: now,
            deadline_micros: now.checked_add(if mutation { 30_000_000 } else { 10_000_000 }),
            idempotency_token: idempotency_token.filter(|token| *token != 0),
            ..RequestOptions::default()
        }
    }

    async fn permit(&self) -> Result<OwnedSemaphorePermit, GatewayError> {
        self.permits
            .clone()
            .acquire_owned()
            .await
            .map_err(GatewayError::internal)
    }
}

fn validate_knowledge(value: &KnowledgeWrite) -> Result<(), GatewayError> {
    let index_text_bytes = value.index_text.as_ref().map_or(
        value
            .title
            .len()
            .saturating_add(1)
            .saturating_add(value.content.len()),
        String::len,
    );
    if value.content.is_empty()
        || value.content.len() > 65_536
        || index_text_bytes > 65_536
        || value.title.len() > 16 * 1024
        || value.scope.is_empty()
        || value.scope.len() > 4_096
        || value.kind.is_empty()
        || value.kind.len() > 4_096
    {
        return Err(GatewayError::BadRequest(
            "knowledge item is empty or exceeds a bounded field limit".into(),
        ));
    }
    if value
        .external_id
        .as_ref()
        .is_some_and(|id| id.is_empty() || id.len() > 4_096)
    {
        return Err(GatewayError::BadRequest(
            "external_id is empty or too long".into(),
        ));
    }
    Ok(())
}

fn validate_retrieval(value: &RetrievalRequest) -> Result<(), GatewayError> {
    let uses_lexical = matches!(
        value.mode,
        RetrievalMode::Lexical | RetrievalMode::HybridExact | RetrievalMode::HybridAnn
    );
    if value.query.trim().is_empty()
        || value.query.len() > 64 * 1024
        || (uses_lexical && value.query.len() > 4_096)
        || value.scope.is_empty()
        || value.scope.len() > 4_096
        || value.top_k == 0
        || value.top_k > 1_024
        || value.candidate_limit < value.top_k
        || value.candidate_limit > 10_000
        || (matches!(value.mode, RetrievalMode::Ann | RetrievalMode::HybridAnn)
            && value.candidate_limit > 512)
    {
        return Err(GatewayError::BadRequest(
            "retrieval request is outside bounded limits".into(),
        ));
    }
    Ok(())
}

fn search_operation(
    request: &RetrievalRequest,
    vector: Option<&[f32]>,
) -> Result<ProductOperation, GatewayError> {
    let lexical = matches!(
        request.mode,
        RetrievalMode::Lexical | RetrievalMode::HybridExact | RetrievalMode::HybridAnn
    )
    .then(|| ProductLexicalBranch {
        query: request.query.clone(),
        candidate_limit: request.candidate_limit,
        weight: 1,
    });
    let vectors = match request.mode {
        RetrievalMode::Lexical => vec![],
        RetrievalMode::Exact | RetrievalMode::HybridExact => vec![vector_branch(
            "semantic_exact",
            vector,
            request.candidate_limit,
            ProductVectorExecution::Exact,
        )?],
        RetrievalMode::Ann | RetrievalMode::HybridAnn => vec![vector_branch(
            "semantic_ann",
            vector,
            request.candidate_limit,
            ProductVectorExecution::Ann {
                ef_search: request.candidate_limit.clamp(64, 512),
                exact_rerank: Some(request.candidate_limit),
            },
        )?],
    };
    let mut filters = vec![ProductSearchFilter::Compare {
        field: "scope".into(),
        operator: ProductSearchOperator::Equal,
        value: ProductDocValue::String(request.scope.clone()),
    }];
    if let Some(kind) = &request.kind {
        filters.push(ProductSearchFilter::Compare {
            field: "kind".into(),
            operator: ProductSearchOperator::Equal,
            value: ProductDocValue::String(kind.clone()),
        });
    }
    Ok(ProductOperation::SearchCollection {
        collection: object_id(SEARCH_COLLECTION_ID)?,
        request: ProductSearchRequest {
            lexical,
            vectors,
            filter: ProductSearchFilter::All(filters),
            sort: vec![],
            facets: vec![],
            aggregations: vec![],
            limit: request.candidate_limit,
        },
    })
}

fn vector_branch(
    target: &str,
    vector: Option<&[f32]>,
    candidate_limit: usize,
    execution: ProductVectorExecution,
) -> Result<ProductVectorBranch, GatewayError> {
    let vector = vector
        .ok_or_else(|| GatewayError::BadRequest("vector retrieval needs an embedding".into()))?;
    Ok(ProductVectorBranch {
        target: target.into(),
        query: ProductVector::new(vector.iter().copied()).map_err(GatewayError::internal)?,
        candidate_limit,
        weight: 1,
        execution: Some(execution),
    })
}

fn search_document(
    id: i64,
    kind: &str,
    scope: &str,
    text_value: &str,
    revision: &str,
    embedding: Option<&[f32]>,
) -> Result<ProductDocument, GatewayError> {
    let mut vectors = BTreeMap::new();
    if let Some(embedding) = embedding {
        let vector =
            ProductVector::new(embedding.iter().copied()).map_err(GatewayError::internal)?;
        vectors.insert("semantic_exact".into(), vector.clone());
        vectors.insert("semantic_ann".into(), vector);
    }
    Ok(ProductDocument {
        object_id: object_id(u128::try_from(id).map_err(GatewayError::internal)?)?,
        text: text_value.to_owned(),
        doc_values: BTreeMap::from([
            ("kind".into(), ProductDocValue::String(kind.into())),
            ("scope".into(), ProductDocValue::String(scope.into())),
            ("revision".into(), ProductDocValue::String(revision.into())),
        ]),
        vectors,
    })
}

fn decode_knowledge(row: Vec<ProductValue>) -> Result<KnowledgeRecord, GatewayError> {
    let [
        ProductValue::Signed(id),
        ProductValue::Text(external_id),
        ProductValue::Text(kind),
        ProductValue::Text(scope),
        ProductValue::Text(title),
        ProductValue::Text(content),
        ProductValue::Text(index_text),
        ProductValue::Text(metadata),
        embedding,
        ProductValue::Text(content_sha256),
        ProductValue::Text(revision_sha256),
        ProductValue::Text(incarnation),
        ProductValue::Text(publication_id),
        ProductValue::Text(state),
        ProductValue::Signed(created),
        ProductValue::Signed(updated),
        expires,
    ] = row.as_slice()
    else {
        return Err(GatewayError::Internal("invalid knowledge row".into()));
    };
    Ok(KnowledgeRecord {
        id: *id,
        external_id: external_id.clone(),
        kind: kind.clone(),
        scope: scope.clone(),
        title: title.clone(),
        content: content.clone(),
        index_text: index_text.clone(),
        metadata: parse_json(metadata)?,
        embedding: optional_text(embedding)?
            .map(|value| serde_json::from_str(&value).map_err(GatewayError::internal))
            .transpose()?,
        content_sha256: content_sha256.clone(),
        revision_sha256: revision_sha256.clone(),
        incarnation: incarnation.clone(),
        publication_id: publication_id.clone(),
        state: state.clone(),
        created_at_micros: *created,
        updated_at_micros: *updated,
        expires_at_micros: optional_signed(expires)?,
    })
}

fn sql_rows(
    response: ProductResponse,
) -> Result<(Vec<String>, Vec<Vec<ProductValue>>, SnapshotIdentity), GatewayError> {
    match response {
        ProductResponse::Sql {
            result: ProductSqlResult::Rows { columns, rows },
            snapshot: Some(snapshot),
            ..
        } => Ok((columns, rows, snapshot)),
        _ => Err(GatewayError::Internal(
            "unexpected Hyphae SQL read response".into(),
        )),
    }
}

fn snapshot_evidence(snapshot: SnapshotIdentity) -> SnapshotEvidence {
    SnapshotEvidence {
        directory_lineage: hex::encode(snapshot.directory_lineage),
        visible_csn: snapshot.visible_csn.map(hyphae_native_product::Csn::get),
        catalog_version: snapshot.catalog_version.get(),
        root_digest: hex::encode(snapshot.root_digest),
        logical_time_micros: snapshot.logical_time_micros,
    }
}

fn bounded_proof_limits() -> hyphae_native_product::proof::NativeProofGenerationLimits {
    let mut limits = hyphae_native_product::proof::NativeProofGenerationLimits::default();
    limits.proof.max_proof_bytes = 8 * 1024 * 1024;
    limits.witness.max_witness_bytes = 8 * 1024 * 1024;
    limits.witness.max_file_bytes = 8 * 1024 * 1024;
    limits.witness.max_total_file_bytes = 8 * 1024 * 1024;
    limits.witness.max_decoded_bytes = 8 * 1024 * 1024;
    limits
}

pub fn canonical_json(value: &Value) -> Result<String, GatewayError> {
    serde_json::to_string(value).map_err(GatewayError::internal)
}
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn stable_token(value: &str) -> u128 {
    let digest = Sha256::digest(value.as_bytes());
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    let token = u128::from_be_bytes(bytes);
    if token == 0 { 1 } else { token }
}
fn new_i64_id() -> i64 {
    let bytes = Uuid::now_v7().into_bytes();
    i64::from_be_bytes(bytes[..8].try_into().unwrap_or([0; 8])) & i64::MAX
}
fn object_id(value: u128) -> Result<ObjectId, GatewayError> {
    ObjectId::new(value).map_err(GatewayError::internal)
}
fn text(value: &str) -> ProductValue {
    ProductValue::Text(value.to_owned())
}
fn signed(value: i64) -> ProductValue {
    ProductValue::Signed(value)
}
fn nullable_text(value: Option<String>) -> ProductValue {
    value.map_or(ProductValue::Null, ProductValue::Text)
}
fn nullable_signed(value: Option<i64>) -> ProductValue {
    value.map_or(ProductValue::Null, ProductValue::Signed)
}
fn optional_text(value: &ProductValue) -> Result<Option<String>, GatewayError> {
    match value {
        ProductValue::Null => Ok(None),
        ProductValue::Text(value) => Ok(Some(value.clone())),
        _ => Err(GatewayError::Internal("expected nullable text".into())),
    }
}
fn optional_signed(value: &ProductValue) -> Result<Option<i64>, GatewayError> {
    match value {
        ProductValue::Null => Ok(None),
        ProductValue::Signed(value) => Ok(Some(*value)),
        _ => Err(GatewayError::Internal("expected nullable integer".into())),
    }
}
fn parse_json(value: &str) -> Result<Value, GatewayError> {
    serde_json::from_str(value).map_err(GatewayError::internal)
}
fn optional_json(value: &ProductValue) -> Result<Option<Value>, GatewayError> {
    optional_text(value)?.as_deref().map(parse_json).transpose()
}
fn product_value_json(value: &ProductValue) -> Result<Value, GatewayError> {
    Ok(match value {
        ProductValue::Null => Value::Null,
        ProductValue::Boolean(value) => json!(value),
        ProductValue::Signed(value) => json!(value),
        ProductValue::Unsigned(value) => json!(value),
        ProductValue::Text(value) => json!(value),
        ProductValue::Binary(value) => json!(hex::encode(value)),
        _ => {
            return Err(GatewayError::Internal(
                "unsupported registry scalar type".into(),
            ));
        }
    })
}
fn validate_sha256(value: &str) -> Result<(), GatewayError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(GatewayError::BadRequest(
            "SHA-256 must be 64 lowercase hexadecimal characters".into(),
        ))
    }
}

fn prepare_private_directory(path: PathBuf) -> Result<PathBuf, GatewayError> {
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()?.join(path)
    };
    fs::create_dir_all(&absolute)?;
    let metadata = fs::symlink_metadata(&absolute)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(GatewayError::BadRequest(format!(
            "proof path is not a regular directory: {}",
            absolute.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&absolute, fs::Permissions::from_mode(0o700))?;
    }
    absolute.canonicalize().map_err(Into::into)
}

fn write_private_file(path: &PathBuf, bytes: &[u8]) -> Result<(), GatewayError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let mut output = options.open(path)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    Ok(())
}

fn proof_path(
    proof: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<PathBuf, GatewayError> {
    let path =
        PathBuf::from(proof.get(field).and_then(Value::as_str).ok_or_else(|| {
            GatewayError::Internal(format!("proof reference is missing {field}"))
        })?);
    if !path.starts_with(std::env::current_dir().unwrap_or_default()) && !path.is_absolute() {
        return Err(GatewayError::Internal(
            "proof reference is not an absolute path".into(),
        ));
    }
    Ok(path)
}

fn cache_key(
    prompt: &str,
    model_id: &str,
    scope: &str,
    knowledge_digest: &str,
    policy_key: &str,
) -> String {
    sha256_hex(
        format!("{model_id}\0{scope}\0{knowledge_digest}\0{policy_key}\0{prompt}").as_bytes(),
    )
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32, GatewayError> {
    if left.len() != right.len() || left.is_empty() {
        return Err(GatewayError::Internal(
            "cached embedding dimension mismatch".into(),
        ));
    }
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (&left, &right) in left.iter().zip(right) {
        dot += f64::from(left) * f64::from(right);
        left_norm += f64::from(left) * f64::from(left);
        right_norm += f64::from(right) * f64::from(right);
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(0.0);
    }
    Ok((dot / (left_norm.sqrt() * right_norm.sqrt())) as f32)
}

#[cfg(test)]
mod tests {
    use super::{sha256_hex, stable_token};
    #[test]
    fn hashes_and_tokens_are_stable() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(stable_token("abc"), stable_token("abc"));
        assert_ne!(stable_token("abc"), 0);
    }
}
