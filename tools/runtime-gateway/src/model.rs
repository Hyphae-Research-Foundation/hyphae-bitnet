// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMode {
    #[default]
    Lexical,
    Exact,
    Ann,
    HybridExact,
    HybridAnn,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct RetrievalRequest {
    pub query: String,
    pub scope: String,
    pub kind: Option<String>,
    pub top_k: usize,
    pub candidate_limit: usize,
    pub mode: RetrievalMode,
    pub proof: bool,
}

impl Default for RetrievalRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            scope: "default".to_owned(),
            kind: Some("document".to_owned()),
            top_k: 5,
            candidate_limit: 64,
            mode: RetrievalMode::Lexical,
            proof: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct RagOptions {
    pub enabled: bool,
    pub query: Option<String>,
    pub scope: String,
    pub kind: Option<String>,
    pub top_k: usize,
    pub candidate_limit: usize,
    pub mode: RetrievalMode,
    pub proof: bool,
    pub context_max_bytes: usize,
    pub cache: bool,
    pub cache_ttl_seconds: u64,
    pub cache_similarity: f32,
}

impl Default for RagOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            query: None,
            scope: "default".to_owned(),
            kind: Some("document".to_owned()),
            top_k: 5,
            candidate_limit: 64,
            mode: RetrievalMode::Lexical,
            proof: false,
            context_max_bytes: 24 * 1024,
            cache: false,
            cache_ttl_seconds: 3_600,
            cache_similarity: 0.985,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct KnowledgeWrite {
    pub external_id: Option<String>,
    #[serde(default = "default_document_kind")]
    pub kind: String,
    #[serde(default = "default_scope")]
    pub scope: String,
    #[serde(default)]
    pub title: String,
    pub content: String,
    pub index_text: Option<String>,
    #[serde(default)]
    pub metadata: Value,
    pub expires_at_micros: Option<i64>,
}

fn default_document_kind() -> String {
    "document".to_owned()
}

fn default_scope() -> String {
    "default".to_owned()
}

#[derive(Clone, Debug, Serialize)]
pub struct KnowledgeRecord {
    pub id: i64,
    pub external_id: String,
    pub kind: String,
    pub scope: String,
    pub title: String,
    pub content: String,
    pub index_text: String,
    pub metadata: Value,
    pub embedding: Option<Vec<f32>>,
    pub content_sha256: String,
    pub revision_sha256: String,
    pub incarnation: String,
    pub publication_id: String,
    pub state: String,
    pub created_at_micros: i64,
    pub updated_at_micros: i64,
    pub expires_at_micros: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RetrievalHit {
    pub id: i64,
    pub external_id: String,
    pub kind: String,
    pub scope: String,
    pub title: String,
    pub content: String,
    pub metadata: Value,
    pub content_sha256: String,
    pub revision_sha256: String,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotEvidence {
    pub directory_lineage: String,
    pub visible_csn: Option<u64>,
    pub catalog_version: u64,
    pub root_digest: String,
    pub logical_time_micros: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorBranchEvidence {
    pub target: String,
    pub strategy: String,
    pub approximate: bool,
    pub eligible_documents: usize,
    pub candidate_count: usize,
    pub visited_nodes: usize,
    pub exact_reranked: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProofReference {
    pub proof_path: String,
    pub witness_path: String,
    pub anchor_path: String,
    pub proof_sha256: String,
    pub witness_sha256: String,
    pub trusted_anchor: String,
    pub request_digest: String,
    pub result_digest: String,
    pub evidence_digest: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RetrievalResponse {
    pub query: String,
    pub mode: RetrievalMode,
    pub snapshot: SnapshotEvidence,
    pub approximate: bool,
    pub total_documents: usize,
    pub eligible_documents: usize,
    pub lexical_candidates: usize,
    pub retrieval_candidates: usize,
    pub matched_candidates: usize,
    pub vector_branches: Vec<VectorBranchEvidence>,
    pub hits: Vec<RetrievalHit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<ProofReference>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ArtifactRegistration {
    pub id: String,
    pub kind: String,
    pub uri: String,
    pub sha256: String,
    pub size_bytes: i64,
    pub media_type: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DatasetRegistration {
    pub id: String,
    pub name: String,
    pub manifest_sha256: String,
    pub manifest: Value,
    #[serde(default)]
    pub license: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LineageRegistration {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub transform: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RunRegistration {
    pub id: String,
    pub kind: String,
    pub status: String,
    pub dataset_id: String,
    pub model_id: String,
    pub seed: i64,
    #[serde(default)]
    pub config: Value,
    #[serde(default)]
    pub metrics: Value,
    pub started_at_micros: i64,
    pub completed_at_micros: Option<i64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EvaluationRegistration {
    pub id: String,
    pub run_id: String,
    pub suite: String,
    pub metric: String,
    pub value: String,
    #[serde(default)]
    pub evidence: Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ContaminationRegistration {
    pub id: String,
    pub dataset_id: String,
    pub benchmark_id: String,
    pub candidate_id: String,
    pub method: String,
    pub score: String,
    pub decision: String,
    #[serde(default)]
    pub evidence: Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HardNegativeRequest {
    pub queries: Vec<MiningQuery>,
    #[serde(default = "default_scope")]
    pub scope: String,
    #[serde(default = "default_mining_limit")]
    pub negatives_per_query: usize,
    #[serde(default)]
    pub mode: RetrievalMode,
}

fn default_mining_limit() -> usize {
    8
}

#[derive(Clone, Debug, Deserialize)]
pub struct MiningQuery {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub positive_external_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HardNegativeManifest {
    pub schema: &'static str,
    pub created_at_micros: i64,
    pub mode: RetrievalMode,
    pub scope: String,
    pub rows: BTreeMap<String, Vec<RetrievalHit>>,
    pub retrieval_snapshots: BTreeMap<String, SnapshotEvidence>,
    pub manifest_sha256: String,
}
