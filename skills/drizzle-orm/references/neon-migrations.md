# Neon Serverless Setup & Migrations

## Driver Options

### Neon HTTP (`drizzle-orm/neon-http`)

Best for: **Serverless, edge functions, Vercel/Cloudflare Workers**. One HTTP request per query, no persistent connection.

```typescript
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema'

const sql = neon(process.env.DATABASE_URL!)
export const db = drizzle(sql, { schema })
```

Limitations:
- No transactions (each query is a separate HTTP request)
- No `LISTEN`/`NOTIFY`
- Higher latency per query (HTTP overhead)

### Neon Serverless with Pool (`drizzle-orm/neon-serverless`)

Best for: **Long-running Node.js servers, Fastify, Express**. WebSocket connection, supports transactions and pooling.

```typescript
import { Pool, neonConfig } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-serverless'
import ws from 'ws'
import * as schema from './schema'

// Required for Node.js (not needed in edge/browser)
neonConfig.webSocketConstructor = ws

const pool = new Pool({
  connectionString: process.env.DATABASE_URL!,
  max: 10,                      // max connections in pool
  idleTimeoutMillis: 30_000,     // close idle connections after 30s
  connectionTimeoutMillis: 5_000, // fail if connection takes > 5s
})

export const db = drizzle({ client: pool, schema })
```

### Lazy Singleton (Edge-Safe)

Avoid connecting at module load time in serverless:

```typescript
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema'

function createDb() {
  const url = process.env.DATABASE_URL
  if (!url) throw new Error('DATABASE_URL is not set')
  return drizzle(neon(url), { schema })
}

let instance: ReturnType<typeof createDb> | undefined

export const db = new Proxy({} as ReturnType<typeof createDb>, {
  get(_target, prop, receiver) {
    if (!instance) instance = createDb()
    const value = Reflect.get(instance, prop, receiver)
    return typeof value === 'function' ? value.bind(instance) : value
  },
})
```

### Pool Stats Helper

```typescript
export function getPoolStats() {
  return {
    totalCount: pool.totalCount,
    idleCount: pool.idleCount,
    waitingCount: pool.waitingCount,
  }
}
```

## drizzle-kit Configuration

```typescript
// drizzle.config.ts
import { defineConfig } from 'drizzle-kit'

export default defineConfig({
  // Required
  dialect: 'postgresql',
  schema: './src/db/schema/index.ts',  // or glob: './src/db/schema/*.ts'
  out: './drizzle/migrations',

  // Database connection
  dbCredentials: {
    // Use UNPOOLED URL for migrations (DDL doesn't work through connection poolers)
    url: process.env.DATABASE_URL_UNPOOLED || process.env.DATABASE_URL!,
  },

  // Optional
  verbose: true,    // log SQL statements
  strict: true,     // ask for confirmation before destructive changes
  tablesFilter: ['!_prisma_migrations'],  // exclude tables
})
```

### Environment Variables

```env
# .env
DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/mydb?sslmode=require

# For migrations — use the direct (unpooled) connection
DATABASE_URL_UNPOOLED=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/mydb?sslmode=require
```

Neon provides two connection strings:
- **Pooled** (default, via PgBouncer): For application queries. Supports `?pgbouncer=true`.
- **Unpooled** (direct): For migrations, DDL, `LISTEN/NOTIFY`. Required for `drizzle-kit`.

## CLI Commands

```bash
# Generate migration SQL from schema diff
bunx drizzle-kit generate
# Output: drizzle/migrations/0001_migration_name.sql

# Apply pending migrations to database
bunx drizzle-kit migrate

# Push schema directly to database (dev only, no migration files)
bunx drizzle-kit push

# Pull existing database schema into Drizzle schema files
bunx drizzle-kit pull
# Also known as: bunx drizzle-kit introspect

# Open visual database browser
bunx drizzle-kit studio

# Check schema diff without generating
bunx drizzle-kit check
```

### Recommended package.json Scripts

```json
{
  "scripts": {
    "db:generate": "bunx drizzle-kit generate",
    "db:migrate": "bunx drizzle-kit migrate",
    "db:push": "bunx drizzle-kit push",
    "db:pull": "bunx drizzle-kit pull",
    "db:studio": "bunx drizzle-kit studio",
    "db:seed": "bun src/db/seed.ts"
  }
}
```

## Migration Workflow

### Development (Push)

For rapid iteration during development:

```bash
# Edit schema files → push directly
bunx drizzle-kit push
```

No migration files generated. Schema changes are applied directly to the database.

### Production (Generate + Migrate)

For production deployments with version-controlled migrations:

```bash
# 1. Edit schema files
# 2. Generate migration
bunx drizzle-kit generate

# 3. Review generated SQL in drizzle/migrations/
# 4. Commit migration files

# 5. Apply in production
bunx drizzle-kit migrate
```

### Running Migrations Programmatically

```typescript
import { migrate } from 'drizzle-orm/neon-http/migrator'
// or: import { migrate } from 'drizzle-orm/neon-serverless/migrator'

await migrate(db, { migrationsFolder: './drizzle/migrations' })
```

### Seeding

```typescript
// src/db/seed.ts
import { db } from './index'
import { users, products } from './schema'

async function seed() {
  console.log('Seeding database...')

  // Check if already seeded
  const [{ count }] = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(users)
  if (count > 0) {
    console.log('Database already seeded')
    return
  }

  await db.insert(users).values([
    { name: 'Admin', email: 'admin@example.com', role: 'ADMIN' },
    { name: 'User', email: 'user@example.com', role: 'CUSTOMER' },
  ])

  console.log('Seeding complete')
}

seed().catch(console.error)
```

## Next.js Integration

### App Router (Server Components / Server Actions)

```typescript
// src/db/index.ts — use neon-http for serverless
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema'

const sql = neon(process.env.DATABASE_URL!)
export const db = drizzle(sql, { schema })
```

```typescript
// app/users/page.tsx — Server Component
import { db } from '@/db'
import { users } from '@/db/schema'

export default async function UsersPage() {
  const allUsers = await db.query.users.findMany({
    orderBy: (users, { desc }) => [desc(users.createdAt)],
  })
  return <UserList users={allUsers} />
}
```

```typescript
// app/actions/users.ts — Server Action
'use server'
import { db } from '@/db'
import { users } from '@/db/schema'
import { eq } from 'drizzle-orm'
import { revalidatePath } from 'next/cache'

export async function updateUser(id: string, data: { name: string }) {
  await db.update(users)
    .set({ ...data, updatedAt: new Date() })
    .where(eq(users.id, id))
  revalidatePath('/users')
}
```

### With React Query (Client Components)

```typescript
// Route Handler
// app/api/users/route.ts
import { db } from '@/db'
import { users } from '@/db/schema'
import { NextResponse } from 'next/server'

export async function GET() {
  const result = await db.query.users.findMany()
  return NextResponse.json(result)
}
```

## Troubleshooting

### "Cannot find module 'drizzle-orm/neon-http'"
Install both packages: `bun add drizzle-orm @neondatabase/serverless`

### "db.query is undefined"
Pass `{ schema }` to `drizzle()`: `drizzle(sql, { schema })`

### Migrations fail with "prepared statement already exists"
Use the unpooled connection URL for migrations (PgBouncer doesn't support DDL properly).

### "NeonDbError: connection is not allowed"
Check that `DATABASE_URL` has `?sslmode=require` and the Neon project is not suspended.
