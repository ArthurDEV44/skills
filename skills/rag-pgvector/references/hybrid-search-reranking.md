# Hybrid Search & Reranking

## Why Hybrid Search?

Neither pure vector search nor pure lexical search is sufficient alone:

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| Vector (dense) | Semantic understanding, paraphrases, synonyms | Misses exact terms, numbers, codes |
| Lexical (BM25/tsvector) | Exact keyword match, fast, no embedding needed | No semantic understanding |
| **Hybrid** | **Best of both** | Slightly more complex |

## Implementing Hybrid Search

### Full-Text Search Setup (PostgreSQL)

```sql
-- Generated tsvector column (auto-maintained)
ALTER TABLE chunks ADD COLUMN search_vector TSVECTOR
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- GIN index for fast text search
CREATE INDEX idx_chunks_search ON chunks USING gin (search_vector);
```

### TypeScript Implementation

```typescript
interface SearchResult {
  id: string
  content: string
  score: number
  documentId: string
  metadata: Record<string, unknown>
}

interface HybridSearchParams {
  queryEmbedding: number[]
  queryText: string
  documentIds?: string[]
  denseLimit?: number       // Candidates from vector search (default: 150)
  lexicalLimit?: number     // Candidates from text search (default: 150)
  finalLimit?: number       // Final results after fusion (default: 20)
  denseWeight?: number      // RRF weight for dense (default: 4)
  lexicalWeight?: number    // RRF weight for lexical (default: 1)
  k?: number                // RRF smoothing constant (default: 60)
}

async function hybridSearch(
  db: Database,
  params: HybridSearchParams
): Promise<SearchResult[]> {
  const {
    queryEmbedding,
    queryText,
    documentIds,
    denseLimit = 150,
    lexicalLimit = 150,
    finalLimit = 20,
    denseWeight = 4,
    lexicalWeight = 1,
    k = 60,
  } = params

  const docFilter = documentIds?.length
    ? sql`AND document_id = ANY(${documentIds})`
    : sql``

  const results = await db.execute(sql`
    WITH dense AS (
      SELECT id, content, document_id, metadata,
        ROW_NUMBER() OVER (
          ORDER BY embedding <=> ${JSON.stringify(queryEmbedding)}::vector
        ) AS rank_pos
      FROM chunks
      WHERE true ${docFilter}
      ORDER BY embedding <=> ${JSON.stringify(queryEmbedding)}::vector
      LIMIT ${denseLimit}
    ),
    lexical AS (
      SELECT id, content, document_id, metadata,
        ROW_NUMBER() OVER (
          ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', ${queryText})) DESC
        ) AS rank_pos
      FROM chunks
      WHERE search_vector @@ plainto_tsquery('english', ${queryText})
        ${docFilter}
      LIMIT ${lexicalLimit}
    ),
    fused AS (
      SELECT
        COALESCE(d.id, l.id) AS id,
        COALESCE(d.content, l.content) AS content,
        COALESCE(d.document_id, l.document_id) AS document_id,
        COALESCE(d.metadata, l.metadata) AS metadata,
        COALESCE(1.0 / (${k} + d.rank_pos), 0) * ${denseWeight}
          + COALESCE(1.0 / (${k} + l.rank_pos), 0) * ${lexicalWeight}
          AS rrf_score
      FROM dense d
      FULL OUTER JOIN lexical l ON d.id = l.id
    )
    SELECT id, content, document_id, metadata, rrf_score AS score
    FROM fused
    ORDER BY rrf_score DESC
    LIMIT ${finalLimit}
  `)

  return results.rows as SearchResult[]
}
```

## Reciprocal Rank Fusion (RRF)

RRF merges ranked lists without needing normalized scores.

### Formula

```
RRF(d) = Σ ( weight_i / (k + rank_i(d)) )
```

Where:
- `k = 60` (smoothing constant, standard value from the original paper)
- `rank_i(d)` = rank of document d in list i (1-indexed)
- `weight_i` = importance weight for list i

### Why RRF Over Score Normalization?

- **Score normalization** requires min/max of each list, which changes per query
- **RRF** only uses rank positions, which are stable and comparable
- RRF is simple, robust, and parameter-free (aside from k)

### Tuning RRF Weights

| Scenario | Dense Weight | Lexical Weight | Rationale |
|----------|-------------|----------------|-----------|
| General QA | 4 | 1 | Semantic similarity dominates |
| Code search | 2 | 3 | Exact identifiers matter more |
| Legal/medical | 3 | 2 | Mix of concepts and terminology |
| Keyword-heavy | 1 | 4 | Users search by exact terms |

## Reranking

### Why Rerank?

Bi-encoder embeddings (used in retrieval) encode query and passage independently — they're fast but imprecise. Cross-encoder rerankers see query + passage together, producing more accurate relevance scores.

### Cross-Encoder Reranking

```typescript
interface RerankResult {
  index: number
  relevanceScore: number
}

async function rerankWithCohere(
  query: string,
  passages: string[],
  topN = 20
): Promise<RerankResult[]> {
  const response = await fetch('https://api.cohere.com/v2/rerank', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.COHERE_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'rerank-v3.5',
      query,
      documents: passages,
      top_n: topN,
      return_documents: false,
    }),
  })
  const data = await response.json()
  return data.results.map((r: any) => ({
    index: r.index,
    relevanceScore: r.relevance_score,
  }))
}
```

```typescript
// Voyage AI Reranker
async function rerankWithVoyage(
  query: string,
  passages: string[],
  topN = 20
): Promise<RerankResult[]> {
  const response = await fetch('https://api.voyageai.com/v1/rerank', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.VOYAGE_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'rerank-2',
      query,
      documents: passages,
      top_k: topN,
    }),
  })
  const data = await response.json()
  return data.data.map((r: any) => ({
    index: r.index,
    relevanceScore: r.relevance_score,
  }))
}
```

### Two-Stage Pipeline

```typescript
async function retrieveAndRerank(
  db: Database,
  query: string,
  queryEmbedding: number[],
  documentIds: string[],
  options?: { candidatePool?: number; finalTopN?: number }
): Promise<SearchResult[]> {
  const { candidatePool = 150, finalTopN = 20 } = options ?? {}

  // Stage 1: Broad hybrid retrieval
  const candidates = await hybridSearch(db, {
    queryEmbedding,
    queryText: query,
    documentIds,
    denseLimit: candidatePool,
    lexicalLimit: candidatePool,
    finalLimit: candidatePool,  // Keep all fused results for reranking
  })

  if (candidates.length === 0) return []

  // Stage 2: Cross-encoder reranking
  const passages = candidates.map((c) => c.content)
  const reranked = await rerankWithCohere(query, passages, finalTopN)

  // Map reranked indices back to original results
  return reranked.map((r) => ({
    ...candidates[r.index],
    score: r.relevanceScore,
  }))
}
```

## Corrective RAG

When initial retrieval quality is low, reformulate the query and retry.

### Relevance Threshold Check

```typescript
async function correctiveRAG(
  db: Database,
  query: string,
  queryEmbedding: number[],
  documentIds: string[],
  options?: {
    relevanceThreshold?: number
    maxRetries?: number
  }
): Promise<SearchResult[]> {
  const { relevanceThreshold = 0.5, maxRetries = 2 } = options ?? {}

  let results = await retrieveAndRerank(db, query, queryEmbedding, documentIds)
  let retries = 0

  while (retries < maxRetries) {
    // Check if top result meets quality threshold
    const topScore = results[0]?.score ?? 0
    if (topScore >= relevanceThreshold) break

    // Reformulate query using LLM
    const reformulated = await reformulateQuery(query, results)
    const newEmbedding = await embedText(reformulated)

    results = await retrieveAndRerank(db, reformulated, newEmbedding, documentIds)
    retries++
  }

  return results
}

async function reformulateQuery(
  originalQuery: string,
  poorResults: SearchResult[]
): Promise<string> {
  const context = poorResults.slice(0, 3).map((r) => r.content).join('\n---\n')

  const response = await llm.complete({
    messages: [{
      role: 'user',
      content: `The following search query returned poor results:
Query: "${originalQuery}"

Top results (low relevance):
${context}

Reformulate the query to better find the answer. Return only the reformulated query, nothing else.`,
    }],
  })

  return response.text.trim()
}
```

### When to Use Corrective RAG

- User queries are vague or ambiguous
- Domain has specialized vocabulary the user might not know
- Multi-hop questions that need decomposition
- When retrieval recall is critical (medical, legal)

## Filtering Strategies

### Pre-filtering (Before Vector Search)

```sql
-- Scope search to specific documents
SELECT * FROM chunks
WHERE document_id = ANY($2)
ORDER BY embedding <=> $1::vector
LIMIT 20;

-- Date range filtering
SELECT * FROM chunks
WHERE created_at > '2024-01-01'
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

### Post-filtering (After Vector Search)

```sql
-- Metadata filtering after retrieval
SELECT * FROM (
  SELECT *, embedding <=> $1::vector AS distance
  FROM chunks
  ORDER BY embedding <=> $1::vector
  LIMIT 200              -- Over-fetch to account for filtering
) sub
WHERE (metadata->>'language')::text = 'en'
LIMIT 20;
```

**Pre-filtering is preferred** — it reduces the search space and is more efficient. Use post-filtering only when the filter would eliminate too many index entries.
