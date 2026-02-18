# pgvector Setup & Index Tuning

## Installation

```sql
-- Enable the extension (requires superuser or rds_superuser on AWS)
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify version
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

## Vector Column Types

```sql
-- Fixed-dimension vector
embedding VECTOR(1024)

-- Half-precision (saves 50% storage, slight precision loss)
embedding HALFVEC(1024)

-- Sparse vector (for BM25/SPLADE models with high-dimensional sparse output)
embedding SPARSEVEC(30000)

-- Binary vector (for binary quantization)
embedding BIT(1024)
```

## Distance Operators

| Operator | Function | Distance Type | Use Case |
|----------|----------|---------------|----------|
| `<=>` | `cosine_distance()` | Cosine | Normalized embeddings (most common) |
| `<->` | `l2_distance()` | Euclidean (L2) | Unnormalized embeddings |
| `<#>` | `inner_product()` (neg.) | Inner product | Dot product similarity |
| `<~>` | `l1_distance()` | Manhattan (L1) | Sparse data |
| `<+>` | `hamming_distance()` | Hamming | Binary vectors |
| `<%>` | `jaccard_distance()` | Jaccard | Binary vectors |

## HNSW Index

Hierarchical Navigable Small World — the recommended index type.

### Create Index

```sql
-- Cosine distance (most common)
CREATE INDEX idx_embedding ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- L2 distance
CREATE INDEX idx_embedding_l2 ON chunks
  USING hnsw (embedding vector_l2_ops)
  WITH (m = 16, ef_construction = 200);

-- Inner product
CREATE INDEX idx_embedding_ip ON chunks
  USING hnsw (embedding vector_ip_ops)
  WITH (m = 16, ef_construction = 200);

-- Half-precision
CREATE INDEX idx_halfvec ON chunks
  USING hnsw (embedding halfvec_cosine_ops)
  WITH (m = 16, ef_construction = 200);
```

### HNSW Parameters

| Parameter | Default | Description | Guidance |
|-----------|---------|-------------|----------|
| `m` | 16 | Max connections per node per layer | Higher = better recall, more memory. 12–48 range. |
| `ef_construction` | 64 | Search width during build | Higher = better recall, slower build. 100–500 range. |

### Query-Time Tuning

```sql
-- Increase search candidates for higher recall (default: 40)
SET hnsw.ef_search = 100;

-- Then run your query
SELECT * FROM chunks ORDER BY embedding <=> $1 LIMIT 20;
```

**Rule of thumb:** `ef_search` should be ≥ your LIMIT value. For top-20 queries, `ef_search = 100` gives good recall.

## IVFFlat Index

Inverted File Flat — older, only use when memory is very limited.

```sql
-- Requires a training step (data must exist first)
CREATE INDEX idx_ivfflat ON chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Query-time tuning
SET ivfflat.probes = 10;  -- Default: 1. Higher = better recall, slower.
```

### IVFFlat vs HNSW

| Aspect | HNSW | IVFFlat |
|--------|------|---------|
| Build speed | Slower | Faster |
| Query speed | Faster | Slower at high recall |
| Recall | Higher | Lower (without tuning) |
| Memory | More | Less |
| Training | Not needed | Requires existing data |
| Updates | No rebuild needed | Periodic REINDEX needed |

**Recommendation:** Always use HNSW unless memory is extremely constrained.

## Schema Patterns

### Document + Chunks (Most Common)

```sql
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  source_url TEXT,
  content_type TEXT NOT NULL,  -- 'pdf', 'web', 'markdown', 'text'
  raw_content TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  embedding VECTOR(1024) NOT NULL,
  chunk_index INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  section_title TEXT,             -- For heading-aware chunking
  page_number INTEGER,           -- For PDF sources
  metadata JSONB DEFAULT '{}',
  search_vector TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (document_id, chunk_index)
);

-- Indexes
CREATE INDEX idx_chunks_embedding ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);
CREATE INDEX idx_chunks_search ON chunks USING gin (search_vector);
CREATE INDEX idx_chunks_document ON chunks (document_id);
```

### Multi-Collection Pattern

Partition vectors by collection for multi-tenant or multi-domain RAG:

```sql
CREATE TABLE collections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  embedding_model TEXT NOT NULL,     -- Track which model generated embeddings
  embedding_dimension INTEGER NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  collection_id UUID NOT NULL REFERENCES collections(id),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  embedding VECTOR(1024) NOT NULL,
  -- ... other fields
);

-- Partial index per collection for faster scoped queries
CREATE INDEX idx_chunks_coll_embed ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WHERE collection_id = '<specific-uuid>';
```

## Performance Guidelines

### Sizing

| Rows | HNSW m | ef_construction | ef_search | Expected Recall |
|------|--------|-----------------|-----------|-----------------|
| <100K | 16 | 200 | 100 | >99% |
| 100K–1M | 24 | 256 | 200 | >98% |
| 1M–10M | 32 | 300 | 300 | >97% |
| >10M | 48 | 400 | 400 | >96% |

### Memory

HNSW index memory ≈ `rows × dimensions × 4 bytes × m / 8`. For 1M rows × 1024 dims × m=16:
~8 GB index memory.

### Bulk Insert Optimization

```sql
-- Drop index before bulk insert
DROP INDEX IF EXISTS idx_chunks_embedding;

-- Insert all chunks
INSERT INTO chunks (document_id, content, embedding, chunk_index, token_count) VALUES ...;

-- Rebuild index after insert
CREATE INDEX idx_chunks_embedding ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);
```

### Exact vs Approximate Search

```sql
-- Force exact (sequential) search for debugging
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SELECT * FROM chunks ORDER BY embedding <=> $1 LIMIT 10;

-- Compare with indexed (approximate) results to measure recall
```
