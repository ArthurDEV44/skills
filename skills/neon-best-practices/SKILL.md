---
name: neon-best-practices
description: "Neon serverless Postgres best practices for Next.js App Router: setup, connection pooling, branching, autoscaling, and authentication. Use when writing, reviewing, or refactoring Next.js code with Neon: (1) Setting up @neondatabase/serverless driver with neon() HTTP or Pool/Client WebSocket, (2) Connection pooling with PgBouncer (-pooler endpoint), pooled vs direct connections, transaction mode caveats, (3) Database branching for development, preview deployments, CI/CD with GitHub Actions and Vercel integration, (4) Autoscaling and scale-to-zero configuration (CU limits, suspend_timeout_seconds, cold starts), (5) Server Components and Server Actions with Neon queries, (6) Neon Authorize with Row-Level Security (RLS), JWT, JWKS, Clerk or Auth.js integration, (7) Neon Auth built-in authentication with Stack Auth components, (8) Drizzle ORM or Prisma integration with Neon, drizzle-kit migrations, (9) Edge Functions and Vercel Edge Runtime with Neon, (10) Environment variables (DATABASE_URL, pooled vs unpooled), branch-specific URLs, (11) Neon CLI (neon branches create/reset/delete, neon connection-string), (12) Any @neondatabase/serverless, neonConfig, or Neon API usage."
---

# Neon Serverless Postgres — Next.js Best Practices

## Quick Setup

```bash
npm install @neondatabase/serverless
# With Drizzle ORM (recommended)
npm install drizzle-orm
npm install -D drizzle-kit
```

```env
# .env.local
# Pooled connection (serverless functions, Drizzle, Prisma)
DATABASE_URL="postgresql://user:pass@ep-cool-darkness-123456-pooler.us-east-2.aws.neon.tech/dbname?sslmode=require"

# Direct connection (migrations, pg_dump, session-based work)
DATABASE_URL_UNPOOLED="postgresql://user:pass@ep-cool-darkness-123456.us-east-2.aws.neon.tech/dbname?sslmode=require"
```

The `-pooler` suffix in the hostname enables PgBouncer connection pooling (up to 10,000 concurrent connections). Always use pooled connections for application code and direct connections for migrations and `pg_dump`.

## Driver Options

### 1. HTTP with `neon()` — Serverless & Edge (Recommended Default)

Single-shot stateless queries over HTTP fetch. Lowest latency, sub-10ms cold starts. Ideal for Server Components, Server Actions, Route Handlers, and Edge Functions.

```typescript
// src/db/index.ts
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema'

const sql = neon(process.env.DATABASE_URL!)
export const db = drizzle(sql, { schema })
```

### 2. WebSocket with `Pool` — Transactions & Sessions

Full PostgreSQL wire protocol over WebSocket. Required for interactive transactions, `LISTEN/NOTIFY`, or session state.

```typescript
// src/db/index.ts
import { Pool, neonConfig } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-serverless'
import * as schema from './schema'

// WebSocket config only needed for Node.js v21 and earlier
// import ws from 'ws'
// neonConfig.webSocketConstructor = ws

const pool = new Pool({ connectionString: process.env.DATABASE_URL! })
export const db = drizzle({ client: pool, schema })
```

### 3. Raw `neon()` without ORM

```typescript
import { neon } from '@neondatabase/serverless'

const sql = neon(process.env.DATABASE_URL!)

// Tagged template (safe, parameterized)
const users = await sql`SELECT * FROM users WHERE id = ${userId}`

// Parameterized query
await sql('INSERT INTO comments (body) VALUES ($1)', [comment])
```

## Next.js Integration Patterns

### Server Components

```typescript
// app/users/page.tsx
import { db } from '@/db'
import { users } from '@/db/schema'

export default async function UsersPage() {
  const allUsers = await db.select().from(users)
  return (
    <ul>
      {allUsers.map((u) => <li key={u.id}>{u.name}</li>)}
    </ul>
  )
}
```

### Server Actions

```typescript
// app/actions.ts
'use server'

import { db } from '@/db'
import { comments } from '@/db/schema'
import { revalidatePath } from 'next/cache'

export async function createComment(formData: FormData) {
  const body = formData.get('comment') as string
  await db.insert(comments).values({ body })
  revalidatePath('/')
}
```

### Route Handlers

```typescript
// app/api/users/route.ts
import { db } from '@/db'
import { users } from '@/db/schema'
import { eq } from 'drizzle-orm'
import { NextResponse } from 'next/server'

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)
  const id = searchParams.get('id')
  if (!id) return NextResponse.json({ error: 'Missing id' }, { status: 400 })

  const [user] = await db.select().from(users).where(eq(users.id, id))
  if (!user) return NextResponse.json({ error: 'Not found' }, { status: 404 })

  return NextResponse.json(user)
}
```

### Edge Functions

When using `Pool` in Edge/serverless environments, create and destroy the pool within the request handler:

```typescript
import { Pool } from '@neondatabase/serverless'

export const runtime = 'edge'

export async function GET(req: Request) {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL })
  try {
    const { rows } = await pool.query('SELECT NOW()')
    return Response.json(rows[0])
  } finally {
    // Use ctx.waitUntil in Vercel Edge to avoid blocking the response
    await pool.end()
  }
}
```

## Connection Pooling

Neon uses PgBouncer in **transaction mode** by default.

| Setting | Default |
|---------|---------|
| `pool_mode` | `transaction` |
| `max_client_conn` | 10,000 |
| `default_pool_size` | `0.9 * max_connections` |
| `query_wait_timeout` | 120s |

**When to use pooled vs direct connections:**

| Use Case | Connection Type |
|----------|----------------|
| Application queries | Pooled (`-pooler`) |
| Server Components / Server Actions | Pooled |
| Drizzle ORM queries | Pooled |
| Prisma queries | Pooled |
| Migrations (`drizzle-kit push/migrate`) | Direct (unpooled) |
| `pg_dump` / `pg_restore` | Direct (unpooled) |
| Persistent `SET` commands / `search_path` | Direct (unpooled) |

**Transaction mode caveats:** `SET` statements (like `search_path`) only persist within a single transaction over pooled connections. Use `ALTER ROLE ... SET search_path TO ...` for persistent settings, or qualify schema names explicitly in queries.

## Drizzle Kit Migrations

```typescript
// drizzle.config.ts
import type { Config } from 'drizzle-kit'

export default {
  schema: './src/db/schema.ts',
  out: './drizzle',
  dialect: 'postgresql',
  dbCredentials: {
    // Use DIRECT (unpooled) connection for migrations
    url: process.env.DATABASE_URL_UNPOOLED!,
  },
} satisfies Config
```

```bash
# Development: push schema directly
npx drizzle-kit push

# Production: generate and apply migration files
npx drizzle-kit generate
npx drizzle-kit migrate

# Inspect existing database
npx drizzle-kit pull

# Visual schema browser
npx drizzle-kit studio
```

## Prisma Integration

```prisma
// schema.prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")  // Pooled connection string
}
```

Prisma 5.10.0+ works with a single pooled Neon connection string — no separate direct URL needed for Prisma Migrate.

## Common Pitfalls

1. **Using direct connections in serverless** — Always use pooled (`-pooler`) connections in serverless functions to avoid connection exhaustion
2. **Creating Pool outside request handler** — In Edge/serverless, create `Pool` inside the handler and close it in `finally`; never at module scope
3. **SET over pooled connections** — `SET search_path` and other session variables do not persist across transactions in PgBouncer transaction mode
4. **Running migrations on pooled connection** — Use `DATABASE_URL_UNPOOLED` for `drizzle-kit push/migrate` and `pg_dump`
5. **Ignoring cold starts** — Free-tier computes auto-suspend after 5 minutes. Paid plans can configure or disable `suspend_timeout_seconds`
6. **Not using parameterized queries** — Always use `sql\`...\`` tagged templates or `$1` parameters with `neon()`, never string interpolation
7. **WebSocket constructor in old Node.js** — Node.js v21 and earlier need `neonConfig.webSocketConstructor = ws`
8. **Forgetting SSL** — Always include `?sslmode=require` in connection strings

## References

- `references/branching-workflows.md` — Database branching for development, preview deployments, CI/CD with GitHub Actions and Vercel
- `references/autoscaling-pooling.md` — Autoscaling, scale-to-zero, cold starts, compute sizing, PgBouncer deep dive
- `references/auth-rls.md` — Neon Authorize (RLS), Neon Auth, JWT/JWKS, Clerk and Auth.js integration
- `references/driver-config.md` — Advanced neonConfig options, WebSocket tuning, fetch endpoints, per-client overrides
