# Gateway Threat Model

## Trust Boundaries

| Boundary | Trusted | Untrusted |
| --- | --- | --- |
| Client to gateway | Gateway configuration and server binary | HTTP bodies, headers, prompts, tool output |
| Gateway to Hyphae | Pinned Hyphae client and authenticated UDS | Stored document content and metadata |
| Gateway to BitNet | Loopback URL fixed by operator | Generated tokens and model output |
| Gateway to embeddings | Loopback URL fixed by operator | Embedding service response |
| Proof verification | Independently pinned anchor | Delivered proof and witness |
| Artifact registry | Canonical metadata and verified hashes | URI targets and external bytes |

## Protected Assets

- Hyphae owner and gateway API keys.
- Gateway and BitNet bearer keys.
- GGUF files and their SHA-256 identities.
- Hyphae data directory, WAL, backups, proofs, witnesses, and anchors.
- Private documents, memory, dataset manifests, run metadata, and receipts.
- Availability of CPU, memory, disk, and generation slots.

## Primary Threats

### Prompt Injection

Retrieved records are explicitly delimited as untrusted evidence and injected
below a system rule that prohibits treating records as instructions. Request
bodies cannot choose service URLs, credential paths, object IDs, or storage
paths. Retrieval remains scoped by `scope` and `kind` doc values.

This boundary reduces but does not eliminate model-level prompt injection.
Applications must keep high-impact tools behind independent authorization and
human confirmation.

### SSRF And Credential Exfiltration

BitNet uses plain HTTP only on loopback. Embeddings accept HTTP or HTTPS only
for `127.0.0.1`, `::1`, or `localhost`. URLs and keys come only from startup
configuration. Secret files must be regular files without group or other access.
Secrets are not logged or returned in receipts.

### Unauthorized Local Access

Hyphae UDS uses API-key authentication and a separate Writer principal. The
owner key is offline. UDS parent directories should be `0700` and the socket is
`0600`. Gateway bearer authentication covers every route when configured.
Non-loopback gateway binding without authentication is rejected.

### Resource Exhaustion

The gateway limits request bodies, retrieval top-k, branch candidates, context
bytes, proof size, concurrent generations, and concurrent Hyphae requests.
Hyphae independently enforces count, byte, work, and memory envelopes. Operators
should add cgroup/namespace limits, disk quotas, connection limits, and reverse
proxy timeouts for shared machines.

### Idempotency And Partial Publication

Mutations use stable nonzero Hyphae idempotency tokens. Search ingest/update and
delete use their own durable identities. Generation is outside transactions and
has explicit `pending`, `completed`, and `failed` states. A SQL document row is
marked `indexing` before search publication and activated afterward; indexing
failures do not become retrievable and retries can repair a missing search item.

### Proof Substitution

Proof, witness, and anchor paths are retained under a private configured proof
root, with SHA-256 values recorded and checked. Verification rejects paths
outside that root and performs origin-independent semantic re-execution. Trust
still requires the anchor to be distributed independently from untrusted proof
delivery; the colocated anchor is transport material, not an independent trust
root.

### Malicious Models And Artifacts

GGUF and artifact bytes remain untrusted. The runtime strict validator and
SHA-256 checks should run before serving a model. Registry records never imply
that a URI is safe merely because it was stored. Artifact consumers must verify
size and digest before parsing.

### Data Disclosure

Proof witnesses may include the complete native authority and must be encrypted,
access-controlled, retained minimally, and handled like backups. Receipts can
reveal document IDs, hashes, prompts, model identity, and retrieval state.
Applications should enforce per-scope authorization before calling the gateway;
the initial gateway API is single-tenant and does not infer tenant identity from
prompts.

## Non-Claims

- A valid retrieval proof does not prove a generated answer is true.
- ANN evidence does not prove complete nearest-neighbor results.
- The gateway does not provide TLS; terminate TLS at a trusted local proxy.
- Hyphae does not provide cluster replication or object storage.
- The gateway is not a distributed training coordinator.
- The current integrated collection remains bounded to 10,000 documents. Larger
  corpora require explicit sharding and gateway-level result fusion.
- Run one gateway writer for each Hyphae data directory. The gateway holds an
  exclusive OS file lock for its lifetime; every writer using the same sidecar
  must use the same private lock path.
