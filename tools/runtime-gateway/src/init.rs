// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path};

use hyphae_native_catalog::{
    AnalyzerDefinition, AnalyzerFilter, AnalyzerTokenizer, AnnIndexDefinition, CatalogName,
    CatalogObjectV2, DefinitionVersion, FieldSourcePolicy, IncrementalVectorLifecycle,
    LexicalIndexPolicy, NamedVectorDefinition, ObjectHeaderV2, QualifiedName,
    SearchCollectionDefinitionV2, SearchFieldDefinitionV2, SearchFieldOptions, VectorMetric,
    VectorSearchPolicy,
};
use hyphae_native_product::{
    LogicalCatalogObject, NativeProduct, ProductDurability, ProductError, ProductOperation,
    ProductPrincipal, ProductRequestContext, ProductResponse, ProductSession, ProductSessionId,
    ProductSqlResult,
};
use hyphae_native_types::{EngineKind, FieldId, LogicalType, ObjectId, VectorElement, VectorType};

use crate::{
    SEARCH_ANALYZER_ID, SEARCH_COLLECTION_ID, SEARCH_DATABASE_ID, SEARCH_SCHEMA_ID,
    error::GatewayError,
};

const TABLES: &[&str] = &[
    "CREATE TABLE celiums_documents (id BIGINT PRIMARY KEY, external_id TEXT NOT NULL, kind TEXT NOT NULL, scope TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL, index_text TEXT NOT NULL, metadata_json TEXT NOT NULL, embedding_json TEXT, content_sha256 TEXT NOT NULL, revision_sha256 TEXT NOT NULL, incarnation TEXT NOT NULL, publication_id TEXT NOT NULL, state TEXT NOT NULL, created_at_micros BIGINT NOT NULL, updated_at_micros BIGINT NOT NULL, expires_at_micros BIGINT)",
    "CREATE UNIQUE INDEX celiums_documents_external_id ON celiums_documents (external_id)",
    "CREATE TABLE celiums_requests (id TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL, response_json TEXT, error TEXT, created_at_micros BIGINT NOT NULL, updated_at_micros BIGINT NOT NULL)",
    "CREATE TABLE celiums_receipts (id TEXT PRIMARY KEY, request_id TEXT NOT NULL, model_json TEXT NOT NULL, retrieval_json TEXT NOT NULL, generation_json TEXT NOT NULL, proof_json TEXT, created_at_micros BIGINT NOT NULL)",
    "CREATE TABLE celiums_cache (id TEXT PRIMARY KEY, cache_key TEXT NOT NULL, policy_key TEXT NOT NULL, prompt TEXT NOT NULL, embedding_json TEXT, embedding_model TEXT NOT NULL, response_json TEXT NOT NULL, model_id TEXT NOT NULL, scope TEXT NOT NULL, knowledge_digest TEXT NOT NULL, created_at_micros BIGINT NOT NULL, expires_at_micros BIGINT NOT NULL)",
    "CREATE INDEX celiums_cache_key ON celiums_cache (cache_key)",
    "CREATE TABLE celiums_artifacts (id TEXT PRIMARY KEY, kind TEXT NOT NULL, uri TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes BIGINT NOT NULL, media_type TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at_micros BIGINT NOT NULL)",
    "CREATE TABLE celiums_datasets (id TEXT PRIMARY KEY, name TEXT NOT NULL, manifest_sha256 TEXT NOT NULL, manifest_json TEXT NOT NULL, license TEXT NOT NULL, created_at_micros BIGINT NOT NULL)",
    "CREATE TABLE celiums_lineage (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL, transform TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at_micros BIGINT NOT NULL)",
    "CREATE TABLE celiums_runs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL, dataset_id TEXT NOT NULL, model_id TEXT NOT NULL, seed BIGINT NOT NULL, config_json TEXT NOT NULL, metrics_json TEXT NOT NULL, started_at_micros BIGINT NOT NULL, completed_at_micros BIGINT)",
    "CREATE TABLE celiums_evaluations (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, suite TEXT NOT NULL, metric TEXT NOT NULL, value TEXT NOT NULL, evidence_json TEXT NOT NULL, created_at_micros BIGINT NOT NULL)",
    "CREATE TABLE celiums_contamination (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, benchmark_id TEXT NOT NULL, candidate_id TEXT NOT NULL, method TEXT NOT NULL, score TEXT NOT NULL, decision TEXT NOT NULL, evidence_json TEXT NOT NULL, created_at_micros BIGINT NOT NULL)",
];

pub fn initialize(
    data_dir: &Path,
    owner_key: &Path,
    gateway_key: &Path,
    dimension: Option<u16>,
    force_empty_cleanup: bool,
) -> Result<(), GatewayError> {
    if data_dir.exists() {
        if force_empty_cleanup && fs::read_dir(data_dir)?.next().is_none() {
            fs::remove_dir(data_dir)?;
        } else {
            return Err(GatewayError::Conflict(format!(
                "data directory already exists: {}",
                data_dir.display()
            )));
        }
    }
    if owner_key.exists() {
        return Err(GatewayError::Conflict(format!(
            "owner key file already exists: {}",
            owner_key.display()
        )));
    }
    if gateway_key.exists() {
        return Err(GatewayError::Conflict(format!(
            "gateway key file already exists: {}",
            gateway_key.display()
        )));
    }
    if let Some(parent) = owner_key.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = gateway_key.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut product = NativeProduct::create(data_dir).map_err(GatewayError::internal)?;
    let mut session = ProductSession::new(
        ProductSessionId::new(1)
            .ok_or_else(|| GatewayError::Internal("invalid bootstrap session ID".into()))?,
        ProductPrincipal::new("celiums-runtime-gateway-bootstrap")
            .ok_or_else(|| GatewayError::Internal("invalid bootstrap principal".into()))?,
        hyphae_native_product::ProductAuthorization::ALL,
    );
    for (offset, statement) in TABLES.iter().enumerate() {
        let context = bootstrap_context(
            &session,
            u128::try_from(offset + 1).map_err(GatewayError::internal)?,
        );
        match product.dispatch(
            &mut session,
            &context,
            ProductOperation::ExecuteSql {
                statement: (*statement).to_owned(),
                parameters: vec![],
            },
        ) {
            Ok(ProductResponse::Sql {
                result: ProductSqlResult::Command { .. },
                ..
            }) => {}
            Ok(_) => {
                return Err(GatewayError::Internal(
                    "unexpected Hyphae SQL bootstrap response".into(),
                ));
            }
            Err(error) => return Err(GatewayError::internal(error)),
        }
    }

    configure_search(&mut product, dimension)?;
    product
        .bootstrap_access_control_to_file(
            "Celiums Gateway Owner",
            "celiums-runtime-gateway",
            owner_key,
            now_micros(),
        )
        .map_err(GatewayError::internal)?;
    let owner_secret = fs::read_to_string(owner_key)?;
    let owner = product
        .authenticate_api_key(&owner_secret, now_micros())
        .map_err(GatewayError::internal)?;
    let gateway = product
        .create_security_principal(&owner, "Celiums Runtime Gateway", now_micros())
        .map_err(GatewayError::internal)?;
    let owner = product
        .authenticate_api_key(&owner_secret, now_micros())
        .map_err(GatewayError::internal)?;
    product
        .assign_built_in_role(
            &owner,
            gateway.principal_id,
            hyphae_native_product::BuiltInRole::Writer,
            hyphae_native_product::ProductScope::Instance,
            now_micros(),
        )
        .map_err(GatewayError::internal)?;
    let owner = product
        .authenticate_api_key(&owner_secret, now_micros())
        .map_err(GatewayError::internal)?;
    product
        .set_security_principal_enabled(&owner, gateway.principal_id, true, now_micros())
        .map_err(GatewayError::internal)?;
    let owner = product
        .authenticate_api_key(&owner_secret, now_micros())
        .map_err(GatewayError::internal)?;
    product
        .issue_api_key_to_file(
            &owner,
            gateway.principal_id,
            "celiums-runtime-gateway",
            [hyphae_native_product::BuiltInRole::Writer],
            hyphae_native_product::BuiltInRole::Writer.authorization(),
            None,
            gateway_key,
            now_micros(),
        )
        .map_err(GatewayError::internal)?;
    Ok(())
}

fn configure_search(
    product: &mut NativeProduct,
    dimension: Option<u16>,
) -> Result<(), GatewayError> {
    for object in [
        LogicalCatalogObject::V2(CatalogObjectV2::Database(header(
            SEARCH_DATABASE_ID,
            EngineKind::Kernel,
            "celiums",
            None,
        )?)),
        LogicalCatalogObject::V2(CatalogObjectV2::Schema(header(
            SEARCH_SCHEMA_ID,
            EngineKind::Kernel,
            "knowledge",
            Some(SEARCH_DATABASE_ID),
        )?)),
        LogicalCatalogObject::V2(CatalogObjectV2::Analyzer(AnalyzerDefinition {
            header: header(
                SEARCH_ANALYZER_ID,
                EngineKind::Search,
                "multilingual",
                Some(SEARCH_SCHEMA_ID),
            )?,
            tokenizer: AnalyzerTokenizer::UnicodeWord,
            filters: vec![AnalyzerFilter::Lowercase, AnalyzerFilter::AsciiFolding],
        })),
    ] {
        product
            .create_catalog_object_v2(object, ProductDurability::Strict)
            .map_err(GatewayError::internal)?;
    }

    let vectors = if let Some(dimension) = dimension {
        let lifecycle = IncrementalVectorLifecycle {
            delta_max_entries: 1_000,
            consolidate_after_deltas: 4,
            retain_generations: 2,
        };
        let ann = AnnIndexDefinition::new(VectorMetric::Cosine, 16, 64, 64, 512, 7)
            .map_err(GatewayError::internal)?;
        vec![
            NamedVectorDefinition {
                id: FieldId::new(5).map_err(GatewayError::internal)?,
                name: CatalogName::unquoted("semantic_exact").map_err(GatewayError::internal)?,
                vector_type: VectorType::new(VectorElement::Float32, dimension)
                    .map_err(GatewayError::internal)?,
                metric: VectorMetric::Cosine,
                policy: VectorSearchPolicy::Exact,
                lifecycle,
            },
            NamedVectorDefinition {
                id: FieldId::new(6).map_err(GatewayError::internal)?,
                name: CatalogName::unquoted("semantic_ann").map_err(GatewayError::internal)?,
                vector_type: VectorType::new(VectorElement::Float32, dimension)
                    .map_err(GatewayError::internal)?,
                metric: VectorMetric::Cosine,
                policy: VectorSearchPolicy::Ann(ann),
                lifecycle,
            },
        ]
    } else {
        vec![]
    };
    let collection = LogicalCatalogObject::V2(CatalogObjectV2::SearchCollection(
        SearchCollectionDefinitionV2 {
            header: header(
                SEARCH_COLLECTION_ID,
                EngineKind::Search,
                "documents",
                Some(SEARCH_SCHEMA_ID),
            )?,
            fields: vec![
                SearchFieldDefinitionV2 {
                    id: FieldId::new(1).map_err(GatewayError::internal)?,
                    name: CatalogName::unquoted("body").map_err(GatewayError::internal)?,
                    logical_type: LogicalType::Text,
                    analyzer: Some(
                        ObjectId::new(SEARCH_ANALYZER_ID).map_err(GatewayError::internal)?,
                    ),
                    options: SearchFieldOptions {
                        stored: true,
                        doc_values: false,
                        source: FieldSourcePolicy::Retained,
                        lexical: LexicalIndexPolicy::Positions,
                    },
                },
                SearchFieldDefinitionV2 {
                    id: FieldId::new(2).map_err(GatewayError::internal)?,
                    name: CatalogName::unquoted("kind").map_err(GatewayError::internal)?,
                    logical_type: LogicalType::Text,
                    analyzer: None,
                    options: SearchFieldOptions {
                        stored: true,
                        doc_values: true,
                        source: FieldSourcePolicy::Retained,
                        lexical: LexicalIndexPolicy::None,
                    },
                },
                SearchFieldDefinitionV2 {
                    id: FieldId::new(3).map_err(GatewayError::internal)?,
                    name: CatalogName::unquoted("scope").map_err(GatewayError::internal)?,
                    logical_type: LogicalType::Text,
                    analyzer: None,
                    options: SearchFieldOptions {
                        stored: true,
                        doc_values: true,
                        source: FieldSourcePolicy::Retained,
                        lexical: LexicalIndexPolicy::None,
                    },
                },
                SearchFieldDefinitionV2 {
                    id: FieldId::new(4).map_err(GatewayError::internal)?,
                    name: CatalogName::unquoted("revision").map_err(GatewayError::internal)?,
                    logical_type: LogicalType::Text,
                    analyzer: None,
                    options: SearchFieldOptions {
                        stored: true,
                        doc_values: true,
                        source: FieldSourcePolicy::Retained,
                        lexical: LexicalIndexPolicy::None,
                    },
                },
            ],
            vectors,
        },
    ));
    product
        .create_catalog_object_v2(collection, ProductDurability::Strict)
        .map_err(GatewayError::internal)?;
    product
        .provision_search_collection(
            ObjectId::new(SEARCH_COLLECTION_ID).map_err(GatewayError::internal)?,
            now_micros(),
            ProductDurability::Strict,
        )
        .map_err(GatewayError::internal)?;
    Ok(())
}

fn header(
    id: u128,
    owner: EngineKind,
    name: &str,
    parent: Option<u128>,
) -> Result<ObjectHeaderV2, GatewayError> {
    Ok(ObjectHeaderV2 {
        id: ObjectId::new(id).map_err(GatewayError::internal)?,
        owner,
        name: QualifiedName::new(
            CatalogName::unquoted("celiums").map_err(GatewayError::internal)?,
            CatalogName::unquoted("knowledge").map_err(GatewayError::internal)?,
            CatalogName::unquoted(name).map_err(GatewayError::internal)?,
        ),
        parent: parent
            .map(ObjectId::new)
            .transpose()
            .map_err(GatewayError::internal)?,
        definition_version: DefinitionVersion::FIRST,
    })
}

fn bootstrap_context(session: &ProductSession, request_id: u128) -> ProductRequestContext {
    ProductRequestContext::new(
        request_id,
        session.id(),
        now_micros(),
        session.principal().clone(),
        session.authorization(),
    )
    .with_idempotency_token(request_id)
}

pub fn now_micros() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    i64::try_from(micros).unwrap_or(i64::MAX)
}

pub fn remove_partial_directory(data_dir: &Path) -> Result<(), ProductError> {
    if data_dir.exists() {
        fs::remove_dir_all(data_dir).map_err(|_| {
            hyphae_native_product::ProductError::from_code(
                hyphae_native_product::ProductErrorCode::Io,
            )
        })?;
    }
    Ok(())
}
