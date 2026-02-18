# Chunking & Embeddings

## Adaptive Chunking Strategy

Different content types require different chunk sizes for optimal retrieval.

### Content-Aware Chunk Sizes

```typescript
interface ChunkConfig {
  maxTokens: number
  overlapTokens: number
  splitStrategy: 'heading' | 'paragraph' | 'sentence' | 'recursive'
}

const CHUNK_CONFIGS: Record<string, ChunkConfig> = {
  pdf: { maxTokens: 512, overlapTokens: 64, splitStrategy: 'paragraph' },
  web: { maxTokens: 1024, overlapTokens: 128, splitStrategy: 'heading' },
  markdown: { maxTokens: 1500, overlapTokens: 150, splitStrategy: 'heading' },
  text: { maxTokens: 2048, overlapTokens: 200, splitStrategy: 'recursive' },
}
```

### Heading-Aware Chunking

Split on section boundaries to keep coherent blocks together:

```typescript
function chunkByHeadings(
  markdown: string,
  maxTokens: number,
  overlapTokens: number
): Chunk[] {
  const sections = markdown.split(/^(#{1,3}\s+.+)$/gm)
  const chunks: Chunk[] = []
  let currentChunk = ''
  let currentHeading = ''
  let chunkIndex = 0

  for (const section of sections) {
    if (/^#{1,3}\s+/.test(section)) {
      currentHeading = section.trim()
      continue
    }

    const combined = currentChunk + '\n' + section
    if (countTokens(combined) > maxTokens && currentChunk) {
      chunks.push({
        content: currentChunk.trim(),
        chunkIndex: chunkIndex++,
        tokenCount: countTokens(currentChunk),
        sectionTitle: currentHeading,
      })
      // Overlap: keep tail of previous chunk
      const sentences = currentChunk.split(/(?<=[.!?])\s+/)
      currentChunk = takeTailTokens(sentences, overlapTokens) + '\n' + section
    } else {
      currentChunk = combined
    }
  }

  if (currentChunk.trim()) {
    chunks.push({
      content: currentChunk.trim(),
      chunkIndex: chunkIndex++,
      tokenCount: countTokens(currentChunk),
      sectionTitle: currentHeading,
    })
  }

  return chunks
}
```

### Recursive Character Splitting

Fallback for unstructured text — split on decreasing granularity:

```typescript
const SEPARATORS = ['\n\n', '\n', '. ', ', ', ' ', '']

function recursiveSplit(
  text: string,
  maxTokens: number,
  separators = SEPARATORS
): string[] {
  const [separator, ...remaining] = separators
  if (!separator) return [text]                    // Base case: split by character

  const parts = text.split(separator)
  const chunks: string[] = []
  let current = ''

  for (const part of parts) {
    const combined = current ? current + separator + part : part
    if (countTokens(combined) > maxTokens) {
      if (current) chunks.push(current)
      // If single part exceeds limit, split with finer separator
      if (countTokens(part) > maxTokens) {
        chunks.push(...recursiveSplit(part, maxTokens, remaining))
        current = ''
      } else {
        current = part
      }
    } else {
      current = combined
    }
  }

  if (current) chunks.push(current)
  return chunks
}
```

## Embedding Models

### Popular Models Comparison

| Model | Dimensions | Max Tokens | Best For |
|-------|-----------|------------|----------|
| OpenAI text-embedding-3-large | 3072 (or 1536/256) | 8191 | General purpose, adjustable dims |
| OpenAI text-embedding-3-small | 1536 | 8191 | Cost-effective general use |
| Voyage AI voyage-3 | 1024 | 32000 | Code and technical content |
| Cohere embed-v3 | 1024 | 512 | Multilingual, search-optimized |
| BGE-M3 | 1024 | 8192 | Open-source, multilingual |
| Nomic embed-text-v1.5 | 768 | 8192 | Open-source, Matryoshka |

### Embedding Generation (TypeScript)

```typescript
// OpenAI
import OpenAI from 'openai'

const openai = new OpenAI()

async function embedTexts(texts: string[]): Promise<number[][]> {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: texts,
    dimensions: 1024,         // Reduce dimensions for efficiency
  })
  return response.data.map((d) => d.embedding)
}
```

```typescript
// Voyage AI
async function embedTextsVoyage(
  texts: string[],
  inputType: 'document' | 'query'
): Promise<number[][]> {
  const response = await fetch('https://api.voyageai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.VOYAGE_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'voyage-3',
      input: texts,
      input_type: inputType,       // 'document' for indexing, 'query' for search
    }),
  })
  const data = await response.json()
  return data.data.map((d: any) => d.embedding)
}
```

### Batch Embedding

Always batch embedding requests to avoid rate limits and improve throughput:

```typescript
async function embedInBatches(
  texts: string[],
  batchSize = 128
): Promise<number[][]> {
  const embeddings: number[][] = []

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize)
    const batchEmbeddings = await embedTexts(batch)
    embeddings.push(...batchEmbeddings)

    // Rate limit pause between batches
    if (i + batchSize < texts.length) {
      await new Promise((r) => setTimeout(r, 200))
    }
  }

  return embeddings
}
```

## Storing Embeddings in pgvector

### Bulk Insert Pattern

```typescript
import { sql } from 'drizzle-orm'

async function storeChunks(
  db: Database,
  documentId: string,
  chunks: Chunk[],
  embeddings: number[][]
) {
  // Use a transaction for atomicity
  await db.transaction(async (tx) => {
    // Delete existing chunks for this document (re-ingestion)
    await tx.delete(chunksTable)
      .where(eq(chunksTable.documentId, documentId))

    // Insert in batches
    const BATCH = 100
    for (let i = 0; i < chunks.length; i += BATCH) {
      const batch = chunks.slice(i, i + BATCH).map((chunk, j) => ({
        documentId,
        content: chunk.content,
        embedding: sql`${JSON.stringify(embeddings[i + j])}::vector`,
        chunkIndex: chunk.chunkIndex,
        tokenCount: chunk.tokenCount,
        sectionTitle: chunk.sectionTitle ?? null,
        metadata: chunk.metadata ?? {},
      }))
      await tx.insert(chunksTable).values(batch)
    }
  })
}
```

### With Drizzle ORM Schema

```typescript
import { pgTable, uuid, text, integer, jsonb, index, timestamp } from 'drizzle-orm/pg-core'
import { sql } from 'drizzle-orm'

// Note: Drizzle doesn't have native vector support yet.
// Use customType or raw SQL for vector columns.
import { customType } from 'drizzle-orm/pg-core'

const vector = customType<{ data: number[]; dpiverName: string }>({
  dataType() {
    return 'vector(1024)'
  },
  toDriver(value: number[]) {
    return JSON.stringify(value)
  },
  fromDriver(value: unknown) {
    return value as number[]
  },
})

export const chunks = pgTable('chunks', {
  id: uuid('id').defaultRandom().primaryKey(),
  documentId: uuid('document_id').notNull().references(() => documents.id, { onDelete: 'cascade' }),
  content: text('content').notNull(),
  embedding: vector('embedding').notNull(),
  chunkIndex: integer('chunk_index').notNull(),
  tokenCount: integer('token_count').notNull(),
  sectionTitle: text('section_title'),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
})
```

## Embedding Pipeline Architecture

```
Document Upload
  ↓
Content Extraction (PDF → text, HTML → text, etc.)
  ↓
Preprocessing (clean, normalize, detect content type)
  ↓
Adaptive Chunking (by content type)
  ↓
Batch Embedding (128 chunks per request)
  ↓
Store in pgvector (transaction, bulk insert)
  ↓
Indexes auto-update (HNSW)
```

### Important: Query vs Document Embeddings

Some embedding models (Voyage AI, Cohere) distinguish between `document` and `query` input types:
- **Document type:** Used when indexing/storing chunks
- **Query type:** Used when embedding the user's search query

Always use the correct input type — mixing them degrades retrieval quality.
