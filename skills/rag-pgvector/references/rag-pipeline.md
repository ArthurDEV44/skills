# RAG Pipeline

## End-to-End Architecture

```
User Query
  ↓
Query Processing (clean, expand abbreviations)
  ↓
Embed Query (using 'query' input type)
  ↓
Hybrid Search (dense + lexical, RRF fusion)
  ↓
Rerank (cross-encoder, top-N)
  ↓
Relevance Check → Low quality? → Corrective RAG (reformulate + retry)
  ↓
Prompt Assembly (system prompt + sources + query)
  ↓
LLM Generation (streaming)
  ↓
Citation Extraction (parse [Source N] references)
  ↓
Response + Citations → Client
```

## Prompt Assembly

### Source-Grounded Prompt Template

```typescript
function assembleRAGPrompt(
  query: string,
  sources: Array<{ content: string; title: string; chunkIndex: number }>,
  systemInstructions?: string
): Message[] {
  const sourcesBlock = sources
    .map((s, i) => `[Source ${i + 1}] (doc: "${s.title}", chunk ${s.chunkIndex})\n${s.content}`)
    .join('\n\n')

  const systemPrompt = systemInstructions ?? `You are a helpful assistant that answers questions based on the provided sources.

Rules:
- Only use information from the sources below to answer
- Cite sources inline using [Source N] notation
- If multiple sources support a point, cite all of them
- If the sources don't contain enough information, say so clearly
- Be concise and direct`

  return [
    { role: 'system', content: systemPrompt },
    {
      role: 'user',
      content: `Sources:\n${sourcesBlock}\n\nQuestion: ${query}`,
    },
  ]
}
```

### Token Budget Management

```typescript
function fitSourcesInBudget(
  sources: Source[],
  maxSourceTokens: number,
  queryTokens: number,
  systemTokens: number,
  modelContextWindow: number
): Source[] {
  // Reserve tokens for: system prompt + query + generation headroom
  const generationHeadroom = 2000
  const available = modelContextWindow - systemTokens - queryTokens - generationHeadroom
  const budget = Math.min(maxSourceTokens, available)

  const fitted: Source[] = []
  let used = 0

  for (const source of sources) {
    const sourceTokens = countTokens(
      `[Source ${fitted.length + 1}] (doc: "${source.title}", chunk ${source.chunkIndex})\n${source.content}`
    )
    if (used + sourceTokens > budget) break
    fitted.push(source)
    used += sourceTokens
  }

  return fitted
}
```

## Citation Extraction

### Parse Citations from LLM Output

```typescript
interface Citation {
  sourceIndex: number        // 1-indexed reference to the source
  documentId: string
  documentTitle: string
  chunkIndex: number
  excerpt: string            // The chunk content
}

function extractCitations(
  llmResponse: string,
  sources: Array<{ id: string; title: string; chunkIndex: number; content: string }>
): Citation[] {
  const CITATION_RE = /\[Source\s+(\d+)\]/g
  const cited = new Set<number>()

  // Handle code blocks — don't extract citations from inside them
  const codeBlockRanges: Array<[number, number]> = []
  const CODE_BLOCK_RE = /```[\s\S]*?```/g
  let codeMatch: RegExpExecArray | null
  while ((codeMatch = CODE_BLOCK_RE.exec(llmResponse)) !== null) {
    codeBlockRanges.push([codeMatch.index, codeMatch.index + codeMatch[0].length])
  }

  let match: RegExpExecArray | null
  while ((match = CITATION_RE.exec(llmResponse)) !== null) {
    const pos = match.index
    const inCodeBlock = codeBlockRanges.some(([start, end]) => pos >= start && pos < end)
    if (!inCodeBlock) {
      const idx = parseInt(match[1], 10)
      if (idx >= 1 && idx <= sources.length) {
        cited.add(idx)
      }
    }
  }

  return Array.from(cited)
    .sort((a, b) => a - b)
    .map((idx) => {
      const source = sources[idx - 1]
      return {
        sourceIndex: idx,
        documentId: source.id,
        documentTitle: source.title,
        chunkIndex: source.chunkIndex,
        excerpt: source.content.slice(0, 200) + '...',
      }
    })
}
```

## SSE Streaming with Incremental Citations

Stream the LLM response while extracting citations progressively.

### Server-Side (Node.js / Express)

```typescript
import { Response } from 'express'

interface SSEEvent {
  type: 'text' | 'citation' | 'done' | 'error'
  data: unknown
}

function sendSSE(res: Response, event: SSEEvent) {
  res.write(`event: ${event.type}\n`)
  res.write(`data: ${JSON.stringify(event.data)}\n\n`)
}

async function streamRAGResponse(
  res: Response,
  query: string,
  sources: Source[]
) {
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')

  const messages = assembleRAGPrompt(query, sources)
  const seenCitations = new Set<number>()
  let fullResponse = ''

  try {
    const stream = await llm.stream({ messages })

    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content ?? ''
      if (!text) continue

      fullResponse += text

      // Send text chunk
      sendSSE(res, { type: 'text', data: { text } })

      // Check for new citations in accumulated text
      const CITATION_RE = /\[Source\s+(\d+)\]/g
      let match: RegExpExecArray | null
      while ((match = CITATION_RE.exec(fullResponse)) !== null) {
        const idx = parseInt(match[1], 10)
        if (!seenCitations.has(idx) && idx >= 1 && idx <= sources.length) {
          seenCitations.add(idx)
          const source = sources[idx - 1]
          sendSSE(res, {
            type: 'citation',
            data: {
              sourceIndex: idx,
              documentId: source.id,
              documentTitle: source.title,
              chunkIndex: source.chunkIndex,
            },
          })
        }
      }
    }

    // Final citation summary
    sendSSE(res, {
      type: 'done',
      data: {
        totalCitations: seenCitations.size,
        citedSources: Array.from(seenCitations).sort((a, b) => a - b),
      },
    })
  } catch (error) {
    sendSSE(res, {
      type: 'error',
      data: { message: error instanceof Error ? error.message : 'Unknown error' },
    })
  } finally {
    res.end()
  }
}
```

### Client-Side (TypeScript)

```typescript
async function consumeRAGStream(
  url: string,
  query: string,
  onText: (text: string) => void,
  onCitation: (citation: Citation) => void,
  onDone: (summary: { totalCitations: number }) => void
) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''     // Keep incomplete line in buffer

    let eventType = ''
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7)
      } else if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6))
        switch (eventType) {
          case 'text': onText(data.text); break
          case 'citation': onCitation(data); break
          case 'done': onDone(data); break
        }
      }
    }
  }
}
```

## RAG Logging & Metrics

Track retrieval quality for continuous improvement.

### Log Schema

```sql
CREATE TABLE rag_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  query TEXT NOT NULL,
  reformulated_query TEXT,              -- If corrective RAG was used
  document_ids UUID[] NOT NULL,
  retrieval_count INTEGER NOT NULL,     -- Candidates before reranking
  reranked_count INTEGER NOT NULL,      -- Results after reranking
  top_score FLOAT,                      -- Highest reranker score
  citations_count INTEGER,              -- Citations in final response
  latency_retrieval_ms INTEGER,
  latency_rerank_ms INTEGER,
  latency_generation_ms INTEGER,
  latency_total_ms INTEGER,
  corrective_retries INTEGER DEFAULT 0,
  model TEXT,                           -- LLM model used
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### Metrics to Track

| Metric | Target | Alert If |
|--------|--------|----------|
| Top reranker score | >0.5 | <0.3 (retrieval quality issue) |
| Citations per response | 2–5 | 0 (not grounding answers) |
| Corrective retries | 0 (most queries) | >1 frequently (query understanding issue) |
| Retrieval latency | <200ms | >500ms (index or query issue) |
| Total latency (excl. generation) | <500ms | >1s |

### Logging Implementation

```typescript
async function logRAGMetrics(db: Database, metrics: {
  query: string
  reformulatedQuery?: string
  documentIds: string[]
  retrievalCount: number
  rerankedCount: number
  topScore: number
  citationsCount: number
  latencyRetrievalMs: number
  latencyRerankMs: number
  latencyGenerationMs: number
  correctiveRetries: number
  model: string
}) {
  await db.insert(ragLogs).values({
    ...metrics,
    latencyTotalMs: metrics.latencyRetrievalMs + metrics.latencyRerankMs + metrics.latencyGenerationMs,
  })
}
```

## Full Pipeline Example

```typescript
async function answerWithRAG(
  db: Database,
  query: string,
  documentIds: string[],
  options?: { stream?: boolean }
): Promise<{ answer: string; citations: Citation[] }> {
  const startTime = Date.now()

  // 1. Embed query
  const queryEmbedding = await embedText(query, 'query')
  const embedTime = Date.now()

  // 2. Hybrid search + rerank (with corrective RAG)
  const sources = await correctiveRAG(db, query, queryEmbedding, documentIds)
  const retrievalTime = Date.now()

  // 3. Fit sources into token budget
  const fittedSources = fitSourcesInBudget(sources, 8000, countTokens(query), 200, 128000)

  // 4. Assemble prompt
  const messages = assembleRAGPrompt(query, fittedSources)

  // 5. Generate response
  const response = await llm.complete({ messages, model: 'claude-sonnet-4-6' })
  const generateTime = Date.now()

  // 6. Extract citations
  const citations = extractCitations(response.text, fittedSources)

  // 7. Log metrics
  await logRAGMetrics(db, {
    query,
    documentIds,
    retrievalCount: sources.length,
    rerankedCount: fittedSources.length,
    topScore: sources[0]?.score ?? 0,
    citationsCount: citations.length,
    latencyRetrievalMs: retrievalTime - embedTime,
    latencyRerankMs: 0,     // Included in retrieval for corrective RAG
    latencyGenerationMs: generateTime - retrievalTime,
    correctiveRetries: 0,
    model: 'claude-sonnet-4-6',
  })

  return { answer: response.text, citations }
}
```
