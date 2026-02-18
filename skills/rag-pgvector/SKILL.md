---
name: rag-pgvector
description: "RAG (Retrieval-Augmented Generation) pipelines with pgvector and PostgreSQL: vector storage, HNSW/IVFFlat indexes, hybrid search (dense + lexical), RRF fusion, chunking strategies, embedding generation, reranking, citation extraction, and streaming. Use when building or reviewing RAG systems: (1) Setting up pgvector with HNSW indexes and vector columns, (2) Implementing hybrid search combining vector similarity and full-text search, (3) Chunking documents with adaptive strategies, (4) Generating and storing embeddings, (5) Reranking retrieval results, (6) Extracting and streaming citations, (7) Building corrective RAG with query reformulation."
---

# RAG with pgvector

Build production RAG pipelines using PostgreSQL with pgvector for vector storage, hybrid search, and retrieval-augmented generation.

## Quick Setup

### Enable pgvector Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Schema with Vector Column

```sql
CREATE TABLE chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id),
  content TEXT NOT NULL,
  embedding VECTOR(1024) NOT NULL,     -- Dimension matches your model
  chunk_index INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  metadata JSONB DEFAULT '{}',
  search_vector TSVECTOR                -- For lexical/BM25 search
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_chunks_embedding ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- GIN index for full-text search
CREATE INDEX idx_chunks_search ON chunks USING gin (search_vector);

-- Filter index for scoping by document
CREATE INDEX idx_chunks_document ON chunks (document_id);
```

**HNSW vs IVFFlat:** Always prefer HNSW — it offers better recall, faster queries, and no training step. IVFFlat requires `CREATE INDEX ... WITH (lists = N)` and periodic retraining via `REINDEX`. Use IVFFlat only when memory is extremely constrained.

For detailed pgvector SQL, index tuning, and distance operators, see `references/pgvector-setup.md`.

## Chunking Strategies

Adapt chunk size to content type for optimal retrieval quality:

| Content Type | Chunk Size (tokens) | Overlap (tokens) | Rationale |
|-------------|--------------------:|------------------:|-----------|
| PDF / academic | 512 | 64 | Dense, structured content |
| Web pages | 1024 | 128 | Mixed content, headers/lists |
| Markdown docs | 1500 | 150 | Section-based, heading-aware |
| Plain text | 2048 | 200 | Unstructured, needs more context |

### Chunking Best Practices

1. **Split on semantic boundaries** — headings, paragraph breaks, section markers
2. **Preserve overlap** between adjacent chunks to maintain context continuity
3. **Track metadata** — source document, chunk index, section title, page number
4. **Clean before chunking** — strip boilerplate, normalize whitespace, remove navigation elements
5. **Respect token limits** — chunk size + overlap must fit within embedding model's context window

For implementation patterns including recursive splitting, heading-aware chunking, and batch embedding, see `references/chunking-embeddings.md`.

## Vector Search

### Basic Similarity Search

```sql
-- Cosine distance (most common for normalized embeddings)
SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
FROM chunks
WHERE document_id = ANY($2)
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

### Distance Operators

| Operator | Distance | Best For |
|----------|----------|----------|
| `<=>` | Cosine | Normalized embeddings (OpenAI, Voyage, Cohere) |
| `<->` | L2 (Euclidean) | Unnormalized embeddings |
| `<#>` | Inner product (negative) | When using dot product similarity |

## Hybrid Search (Dense + Lexical)

Combine vector similarity with full-text search for better recall. Neither method alone is sufficient — vector search captures semantic meaning but misses exact terms; lexical search finds exact matches but misses paraphrases.

### Two-Query Approach with RRF Fusion

```sql
-- Step 1: Dense retrieval (vector similarity)
WITH dense AS (
  SELECT id, content,
    ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank_dense
  FROM chunks
  WHERE document_id = ANY($3)
  ORDER BY embedding <=> $1::vector
  LIMIT 150
),
-- Step 2: Lexical retrieval (full-text search)
lexical AS (
  SELECT id, content,
    ROW_NUMBER() OVER (ORDER BY ts_rank(search_vector, plainto_tsquery('english', $2)) DESC) AS rank_lexical
  FROM chunks
  WHERE document_id = ANY($3)
    AND search_vector @@ plainto_tsquery('english', $2)
  LIMIT 150
),
-- Step 3: Reciprocal Rank Fusion
fused AS (
  SELECT
    COALESCE(d.id, l.id) AS id,
    COALESCE(d.content, l.content) AS content,
    -- RRF formula: 1/(k + rank), k=60 is standard
    COALESCE(1.0 / (60 + d.rank_dense), 0) * 4  -- Dense weight: 4x
    + COALESCE(1.0 / (60 + l.rank_lexical), 0) * 1  -- Lexical weight: 1x
    AS rrf_score
  FROM dense d
  FULL OUTER JOIN lexical l ON d.id = l.id
)
SELECT id, content, rrf_score
FROM fused
ORDER BY rrf_score DESC
LIMIT 20;
```

**Key parameters:**
- `k = 60` — RRF smoothing constant (standard value, rarely needs tuning)
- Dense weight `4x` vs lexical `1x` — favors semantic similarity; adjust based on domain
- Initial pool of `150` candidates per source, fused down to top `20`

For the full hybrid search pipeline, reranking, and corrective RAG patterns, see `references/hybrid-search-reranking.md`.

## RAG Pipeline Architecture

### Retrieval → Rerank → Generate

```
Query → Embed → Hybrid Search (150+150) → RRF Fusion (top-N)
  → Rerank (cross-encoder, top-20) → Assemble Prompt → LLM → Stream Response
```

### Two-Stage Retrieval

1. **Stage 1 — Broad retrieval:** Hybrid search returns ~150 candidates from each source (dense + lexical), fused with RRF
2. **Stage 2 — Reranking:** Cross-encoder model scores each candidate against the query, selects top 20

Reranking is critical for quality — it uses a more expensive model that sees query + passage together, catching relevance that embedding similarity misses.

### Prompt Assembly with Citations

```
You are answering questions based on the provided sources.
Rules:
- Only use information from the sources below
- Cite sources using [Source N] notation inline
- If the sources don't contain the answer, say so

Sources:
[Source 1] (doc: "Architecture Guide", chunk 3)
<content of chunk>

[Source 2] (doc: "API Reference", chunk 7)
<content of chunk>

Question: {user_query}
```

### Citation Extraction

Parse `[Source N]` references from LLM output to link back to original documents:

```typescript
const CITATION_RE = /\[Source\s+(\d+)\]/g
let match: RegExpExecArray | null
const cited = new Set<number>()
while ((match = CITATION_RE.exec(response)) !== null) {
  cited.add(parseInt(match[1], 10))
}
// Map cited indices back to chunk metadata for source links
```

For the complete RAG pipeline including SSE streaming, corrective RAG, and logging, see `references/rag-pipeline.md`.

## Common Pitfalls

### Embedding Dimension Mismatch
**Problem:** `VECTOR(1536)` column with 1024-dim embeddings (or vice versa).
**Fix:** Match column dimension to your embedding model exactly. OpenAI ada-002 = 1536, Voyage AI = 1024, Cohere v3 = 1024.

### Missing HNSW Index
**Problem:** Queries scan entire table (exact search), extremely slow at scale.
**Fix:** Always create HNSW index. Set `ef_construction = 200` for good recall, increase for higher precision needs.

### Pure Vector Search Without Lexical
**Problem:** Vector search misses exact keyword matches (product names, error codes, acronyms).
**Fix:** Always use hybrid search combining vector + full-text search with RRF fusion.

### Chunking Too Large or Too Small
**Problem:** Large chunks dilute relevance; small chunks lose context.
**Fix:** Use adaptive chunk sizes by content type. Include overlap. Track what works with retrieval metrics.

### No Reranking Stage
**Problem:** Bi-encoder retrieval returns false positives that look similar in embedding space but aren't relevant.
**Fix:** Add a cross-encoder reranking stage that scores query-passage pairs directly.

### Stale Embeddings After Model Change
**Problem:** Switching embedding models without re-embedding existing data.
**Fix:** Re-embed all chunks when changing models. Store model name in metadata for tracking.
