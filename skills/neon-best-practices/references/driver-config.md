# Driver Configuration & Advanced Patterns

## @neondatabase/serverless — Three Client Modes

### 1. `neon()` — HTTP Fetch (Stateless)

Best for: Server Components, Server Actions, Edge Functions, single queries.

```typescript
import { neon } from '@neondatabase/serverless'

const sql = neon(process.env.DATABASE_URL!)

// Tagged template — safe, parameterized
const users = await sql`SELECT * FROM users WHERE active = ${true}`

// Parameterized query
const result = await sql('SELECT * FROM users WHERE id = $1', [userId])

// Multiple statements (each runs in its own implicit transaction)
const [users, posts] = await Promise.all([
  sql`SELECT * FROM users LIMIT 10`,
  sql`SELECT * FROM posts LIMIT 10`,
])
```

**Characteristics:**
- Lowest latency, sub-10ms cold starts
- No connection state — each query is independent
- No transactions (use Pool/Client for transactions)
- Works everywhere: Vercel Edge, Cloudflare Workers, Deno Deploy

### 2. `Pool` — WebSocket Connection Pool

Best for: Serverless functions needing transactions, batch operations.

```typescript
import { Pool } from '@neondatabase/serverless'

// In serverless: create INSIDE request handler
export async function handler(req: Request, ctx: any) {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL })

  try {
    // Simple query
    const { rows } = await pool.query('SELECT * FROM users WHERE id = $1', [userId])

    // Transaction
    const client = await pool.connect()
    try {
      await client.query('BEGIN')
      await client.query('UPDATE accounts SET balance = balance - $1 WHERE id = $2', [100, fromId])
      await client.query('UPDATE accounts SET balance = balance + $1 WHERE id = $2', [100, toId])
      await client.query('COMMIT')
    } catch (err) {
      await client.query('ROLLBACK')
      throw err
    } finally {
      client.release()
    }

    return Response.json({ success: true })
  } finally {
    ctx.waitUntil(pool.end())  // Don't block the response
  }
}
```

**Characteristics:**
- Full PostgreSQL wire protocol over WebSocket
- Supports transactions, prepared statements, `LISTEN/NOTIFY`
- Must create and destroy within request handler in serverless
- Drop-in compatible with `pg.Pool` (node-postgres)

### 3. `Client` — Single WebSocket Connection

Best for: Interactive transactions, long-running sessions, scripts.

```typescript
import { Client, neonConfig } from '@neondatabase/serverless'

const client = new Client(process.env.DATABASE_URL)
await client.connect()

try {
  // Simple query
  const { rows } = await client.query('SELECT NOW()')

  // Prepared statement
  const result = await client.query({
    text: 'SELECT * FROM users WHERE id = $1',
    values: [42],
  })

  // Interactive transaction
  await client.query('BEGIN')
  await client.query('UPDATE accounts SET balance = balance - $1 WHERE id = $2', [50, 1])
  await client.query('UPDATE accounts SET balance = balance + $1 WHERE id = $2', [50, 2])
  await client.query('COMMIT')
} catch (err) {
  await client.query('ROLLBACK').catch(() => {})
  throw err
} finally {
  await client.end()
}
```

**Characteristics:**
- Single persistent connection over WebSocket
- Full session state support
- Not recommended for serverless (use Pool instead)
- Drop-in compatible with `pg.Client` (node-postgres)

## neonConfig — Global Settings

```typescript
import { neonConfig } from '@neondatabase/serverless'

// --- WebSocket ---
// Required for Node.js v21 and earlier (v22+ has built-in WebSocket)
import ws from 'ws'
neonConfig.webSocketConstructor = ws

// Alternative: use undici's WebSocket
// import { WebSocket } from 'undici'
// neonConfig.webSocketConstructor = WebSocket

// --- Connection Optimization ---
// Pipeline startup messages (skip round-trips during connection)
neonConfig.pipelineConnect = 'password'  // default: 'password'

// Batch multiple writes into single WebSocket frames
neonConfig.coalesceWrites = true  // default: true

// --- TLS ---
// Use secure WebSocket (wss://)
neonConfig.useSecureWebSocket = true  // default: true

// Disable Postgres-level TLS (when already using wss://)
neonConfig.forceDisablePgSSL = true  // default: false

// --- HTTP Fetch ---
// Route Pool.query() over HTTP for lower latency (experimental)
neonConfig.poolQueryViaFetch = true  // default: false

// Custom fetch endpoint
neonConfig.fetchEndpoint = (host, port, options) => {
  if (process.env.NODE_ENV === 'development') {
    return `http://localhost:4444/sql`  // Local proxy
  }
  return `https://${host}/sql`  // Default
}
```

### Per-Client Overrides

Override global settings for individual clients:

```typescript
const client = new Client(process.env.DATABASE_URL)
client.neonConfig.pipelineConnect = false
client.neonConfig.coalesceWrites = false
await client.connect()
```

## Driver Selection Guide

| Scenario | Driver | Why |
|----------|--------|-----|
| Next.js Server Component | `neon()` | Stateless, lowest latency |
| Next.js Server Action | `neon()` | Stateless, single query |
| Server Action with transaction | `Pool` | Transaction support |
| Next.js Route Handler | `neon()` | Stateless by default |
| Edge Function | `neon()` | HTTP-only, no WebSocket needed |
| Edge Function with transaction | `Pool` | Create inside handler, end in finally |
| Script / migration | `Client` | Long-running, session state |
| Auth.js adapter | `Pool` | Adapter requires Pool interface |
| Real-time (`LISTEN/NOTIFY`) | `Client` | Persistent connection needed |

## ORM Integration Patterns

### Drizzle ORM — HTTP Driver (Recommended)

```typescript
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema'

const sql = neon(process.env.DATABASE_URL!)
export const db = drizzle(sql, { schema })
```

### Drizzle ORM — WebSocket Pool

```typescript
import { Pool } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-serverless'
import * as schema from './schema'

const pool = new Pool({ connectionString: process.env.DATABASE_URL! })
export const db = drizzle({ client: pool, schema })
```

### Prisma

```prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")  // Use pooled connection
}
```

Prisma 5.10.0+ works with a single pooled Neon connection string.

## Error Handling & Retry

```typescript
import { neon } from '@neondatabase/serverless'

const sql = neon(process.env.DATABASE_URL!)

async function queryWithRetry<T>(
  queryFn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 100
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await queryFn()
    } catch (err: any) {
      const isRetryable =
        err.code === 'ECONNRESET' ||
        err.code === '57P01' || // admin_shutdown (compute restart)
        err.message?.includes('Connection terminated')

      if (!isRetryable || attempt === maxRetries - 1) throw err

      const delay = baseDelay * Math.pow(2, attempt)
      await new Promise((r) => setTimeout(r, delay))
    }
  }
  throw new Error('Unreachable')
}

// Usage
const users = await queryWithRetry(() => sql`SELECT * FROM users`)
```

## Local Development

### Option 1: Local Postgres + Neon Driver

```env
# .env.local
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/myapp"
```

The `neon()` HTTP driver only works against Neon endpoints. For local development, use `postgres` or `pg` directly, or use Neon's local proxy.

### Option 2: Neon Dev Branch (Recommended)

Create a dedicated dev branch and use its connection string locally:

```bash
neon branches create --name dev/local --project-id <project-id>
neon connection-string dev/local --project-id <project-id> --pooled
```

```env
# .env.local
DATABASE_URL="postgresql://...@ep-xxx-pooler.../dbname?sslmode=require"
```

This gives you a real Neon environment with branching, instant reset, and zero setup.
