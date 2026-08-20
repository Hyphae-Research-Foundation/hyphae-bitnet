# Celiums Local AI Gateway

The Celiums Runtime Gateway combines the Celiums BitNet inference runtime with
Hyphae Native storage and search without placing either dependency inside GGML
or the public runtime C ABI.

## Architecture

```text
client
  |
  v
celiums-runtime-gateway             HTTP loopback, optional bearer key
  |                       |
  | HYPHLCL1 UDS          | HTTP loopback
  v                       v
celiums-hyphae-sidecar    celiums-bitnet serve
  |                       |
  v                       v
Hyphae Native directory  read-only GGUF
```

The gateway owns context selection, prompt boundaries, receipts, idempotency,
semantic-cache policy, embedding calls, and API limits. Hyphae owns durable SQL,
BM25, exact and HNSW vector indexes, snapshots, WAL, proofs, and witnesses.
BitNet owns tokenization, chat templates, generation, streaming, and model
provenance.

Hyphae is pinned to release `1.2.2`, commit
`0471ae25b263fd506da1578068ec57429a6783de`. The gateway never follows Hyphae
`main`.

## Binaries

- `celiums-runtime-gateway`: initialization, HTTP API, receipts, registry, and
  offline proof verification.
- `celiums-hyphae-sidecar`: authenticated UDS owner for one initialized Hyphae
  directory.
- `celiums-runtime-mcp`: separate MCP stdio adapter for retrieval, RAG, memory,
  and receipt verification.

## Build

Rust 1.97.1 is the pinned build and release toolchain.

```bash
cargo build --manifest-path tools/runtime-gateway/Cargo.toml --locked --release --bins
```

The optional CMake integration keeps normal C/C++ builds independent from
Rust:

```bash
cmake -S . -B build-gateway \
  -DCELIUMS_BITNET_BUILD_GATEWAY=ON \
  -DBUILD_SHARED_LIBS=ON
cmake --build build-gateway --target celiums-runtime-gateway-binaries
```

Release archives build and install all three gateway binaries.

## Initialize

Initialization must run while no sidecar owns the data directory. The data
directory and both output key files must not exist.

```bash
install -d -m 0700 "$HOME/.local/share/celiums" "$XDG_RUNTIME_DIR/celiums"

celiums-runtime-gateway init \
  --data-dir "$HOME/.local/share/celiums/hyphae" \
  --owner-key "$HOME/.local/share/celiums/owner.key" \
  --gateway-key "$HOME/.local/share/celiums/gateway.key"
```

`owner.key` has full offline administrative authority and should not be mounted
into the gateway service. `gateway.key` belongs to a separate Writer principal
and supplies only the application permissions needed for SQL, search, ingest,
and proof generation.

To enable vectors, the exact embedding dimension must be fixed at initialization:

```bash
celiums-runtime-gateway init \
  --data-dir "$HOME/.local/share/celiums/hyphae" \
  --owner-key "$HOME/.local/share/celiums/owner.key" \
  --gateway-key "$HOME/.local/share/celiums/gateway.key" \
  --embedding-dimension 768
```

The initializer creates:

- durable document, request, receipt, cache, artifact, dataset, lineage, run,
  evaluation, and contamination tables;
- one Unicode lowercase/ascii-folding lexical analyzer;
- one integrated search collection with `scope` and `kind` doc values;
- optional exact and HNSW cosine vector targets;
- Owner and least-privilege gateway credentials.

## Start Services

Start the data sidecar:

```bash
celiums-hyphae-sidecar \
  --data-dir "$HOME/.local/share/celiums/hyphae" \
  --endpoint "$XDG_RUNTIME_DIR/celiums/hyphae.sock"
```

Start BitNet on loopback:

```bash
CELIUMS_BITNET_API_KEY=runtime-secret \
celiums-bitnet serve \
  --model /models/model.gguf \
  --host 127.0.0.1 --port 8080
```

Store gateway and BitNet bearer secrets in regular `0600` files, then start the
gateway:

```bash
celiums-runtime-gateway serve \
  --bind 127.0.0.1:8090 \
  --hyphae-endpoint "$XDG_RUNTIME_DIR/celiums/hyphae.sock" \
  --hyphae-key-file "$HOME/.local/share/celiums/gateway.key" \
  --api-key-file "$HOME/.local/share/celiums/http.key" \
  --bitnet-url http://127.0.0.1:8080 \
  --bitnet-api-key-file "$HOME/.local/share/celiums/bitnet.key" \
  --model-path /models/model.gguf \
  --proof-dir "$HOME/.local/share/celiums/proofs"
```

The gateway holds an exclusive OS file lock next to the Hyphae endpoint for its
entire lifetime. Override it with `--lock-file` only when the runtime layout
requires another private location. A second gateway writer fails startup.

The gateway rejects non-loopback binding unless a gateway API key is supplied.
The BitNet and embedding URLs are restricted to loopback and cannot be supplied
by prompts or request bodies.

## Ingest And Retrieve

```bash
curl -X PUT http://127.0.0.1:8090/v1/knowledge \
  -H 'Authorization: Bearer gateway-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "external_id": "runtime-contract",
    "kind": "document",
    "scope": "bitnet",
    "title": "Numerical contract",
    "content": "Strict I2_S uses unsigned two-bit codes and exact recovery.",
    "metadata": {"source": "docs/NUMERICAL_CONTRACT.md"}
  }'
```

```bash
curl -X POST http://127.0.0.1:8090/v1/retrieval \
  -H 'Authorization: Bearer gateway-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How does I2_S recover output?",
    "scope": "bitnet",
    "kind": "document",
    "top_k": 5,
    "candidate_limit": 64,
    "mode": "lexical",
    "proof": true
  }'
```

Modes are `lexical`, `exact`, `ann`, `hybrid_exact`, and `hybrid_ann`. A vector
mode requires both an initialized vector dimension and an OpenAI-compatible
loopback embedding endpoint configured at gateway startup:

```bash
celiums-runtime-gateway serve \
  ... \
  --embedding-url http://127.0.0.1:8081/v1/embeddings \
  --embedding-model ctt-embed \
  --embedding-dimension 768 \
  --initialized-embedding-dimension 768
```

The repeated initialized dimension is an explicit startup assertion. A mismatch
fails before the gateway accepts traffic; omit both dimension flags for a
lexical-only directory.

## RAG Completion

The gateway preserves the BitNet OpenAI-compatible routes and accepts an
additional opt-in `rag` object. Requests without `rag.enabled=true` pass through
without retrieval or prompt modification:

```bash
curl -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Authorization: Bearer gateway-secret' \
  -H 'Idempotency-Key: example-001' \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Explain the I2_S recovery rule."}],
    "temperature": 0,
    "max_tokens": 256,
    "rag": {
      "enabled": true,
      "scope": "bitnet",
      "mode": "lexical",
      "top_k": 5,
      "proof": true,
      "context_max_bytes": 24576
    }
  }'
```

Retrieved text is delimited and labeled as untrusted evidence. This reduces but
does not eliminate model-level prompt injection; consequential tools still need
independent authorization. The response includes `celiums.receipt_id`, retrieval identity,
document hashes, and the caller request identity. Non-streaming retries with the
same `Idempotency-Key` and identical request return the completed response;
conflicting payloads fail.

Semantic caching is opt-in with `rag.cache=true`. Exact cache identity binds the
model, scope, retrieval root digest, and prompt. Semantic cache reuse additionally
requires the configured embedding model and threshold. Cache entries expire and
are never reused across retrieval snapshots.

## Proofs And Receipts

A proof-bearing retrieval returns paths and SHA-256 values for a `.hynproof`,
`.hynwit`, and independent 32-byte anchor file. Verify a receipt through the
gateway:

```bash
curl -X POST http://127.0.0.1:8090/v1/receipts/RECEIPT_ID/verify \
  -H 'Authorization: Bearer gateway-secret'
```

Or verify without either running service:

```bash
celiums-runtime-gateway verify \
  --proof receipt.hynproof \
  --witness receipt.hynwit \
  --anchor receipt.anchor
```

The verifier checks artifact integrity, receipt-recorded file hashes, anchor
binding, and semantic re-execution against the retained native authority. For a
security assertion, distribute and pin the trusted anchor through a channel
independent of the proof directory. Proof generation
is bounded to 8 MiB in this gateway because the local protocol response is
bounded to 16 MiB. A witness is a complete retained authority and must be
protected like a backup. A Hyphae proof establishes retrieval execution and
state, not the truth of generated prose.

## Training And Evaluation Registry

The registry endpoints retain immutable references and evidence, not large
checkpoint bytes:

- `POST /v1/registry/artifacts`
- `GET /v1/registry/artifacts/{id}`
- `POST /v1/registry/datasets`
- `GET /v1/registry/datasets/{id}`
- `POST /v1/registry/lineage`
- `POST /v1/registry/runs`
- `GET /v1/registry/runs/{id}`
- `POST /v1/registry/evaluations`
- `POST /v1/registry/contamination`
- `POST /v1/mining/hard-negatives`

Dataset IDs should be derived from canonical manifest SHA-256. Artifacts store
URI, SHA-256, size, media type, and metadata; shards, safetensors, optimizer
state, logits, and traces remain in content-addressed object storage. Hard
negative mining records snapshot identities and excludes declared positives.
ANN only generates candidates for contamination checks; absence from ANN is not
proof of non-contamination.

Registry object reads are available by ID for artifacts, datasets, and runs.
Lineage, evaluation, and contamination registration records remain append-only
evidence in this first API; broader listing and workflow validation stay with
the caller until a paginated contract is versioned.

## MCP

```bash
celiums-runtime-mcp \
  --gateway-url http://127.0.0.1:8090 \
  --api-key-file "$HOME/.local/share/celiums/http.key"
```

The stdio server exposes:

- `celiums_retrieve_context`
- `celiums_rag_answer`
- `celiums_memory_write`
- `celiums_memory_search`
- `celiums_answer_verify`

This is separate from Hyphae's own read-only MCP adapter, so no existing Hyphae
contract is widened.

## Limits And Recovery

- Gateway request body: 8 MiB.
- Hyphae product request/response: 16 MiB.
- Integrated ingest batch: 256 documents and 16 MiB.
- Integrated collection: 10,000 durable documents.
- Retrieval branch candidates: 10,000.
- Retrieval hits: 1,024.
- Concurrent generation and Hyphae request counts are bounded at startup.
- A Hyphae directory has one owning process. Stop the sidecar before offline
  backup, restore, owner-key administration, or catalog re-provisioning.
- Generation never runs inside a Hyphae transaction. Requests transition from
  `pending` to `completed` or `failed` in separate strict commits.
- If SQL publication precedes integrated indexing and indexing fails, the row
  remains `indexing` and is excluded from retrieval; retrying the same external
  ID repairs or completes indexing.
- Use Hyphae checkpoint, backup, restore, and doctor procedures against the
  stopped data directory. Restore always targets a new directory.

## Validation

```bash
cargo fmt --manifest-path tools/runtime-gateway/Cargo.toml --all -- --check
CARGO_INCREMENTAL=0 cargo clippy \
  --manifest-path tools/runtime-gateway/Cargo.toml --locked --all-targets -- -D warnings
CARGO_INCREMENTAL=0 cargo test \
  --manifest-path tools/runtime-gateway/Cargo.toml --locked --all-targets
```

The end-to-end Rust test creates a real Hyphae directory, bootstraps separate
Owner and Writer credentials, starts authenticated UDS, ingests and updates a
document, runs BM25, generates a proof/witness, verifies semantic re-execution,
and deletes the document.
