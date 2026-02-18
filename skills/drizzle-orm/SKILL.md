---
name: drizzle-orm
description: "Drizzle ORM for TypeScript/PostgreSQL: type-safe schema definition, queries, migrations, and Neon serverless integration. Use when writing, reviewing, or refactoring code with Drizzle ORM: (1) Defining schemas with pgTable, columns, constraints, indexes, enums, (2) Relations with relations(), one(), many(), self-referencing, many-to-many, (3) CRUD with db.select/insert/update/delete and filter operators (eq, and, or, like, inArray, between), (4) Relational queries with db.query, findFirst, findMany, with, (5) Joins: leftJoin, innerJoin, subqueries, aggregations (count, sum, min, max), (6) Transactions with db.transaction and Tx type pattern, (7) Migrations with drizzle-kit generate/push/migrate, drizzle.config.ts, (8) Neon serverless setup (neon-http, neon-serverless, Pool), (9) Type inference with $inferSelect, $inferInsert, $type, (10) Raw SQL with sql template tag, db.execute, (11) JSON columns, decimal for money, typed enums with pgEnum, (12) Any drizzle-orm or drizzle-kit imports."
---

# Drizzle ORM — PostgreSQL + Neon

## Quick Setup

```typescript
// src/db/index.ts — Neon HTTP (serverless, edge-compatible)
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema'

const sql = neon(process.env.DATABASE_URL!)
export const db = drizzle(sql, { schema })
```

```typescript
// src/db/index.ts — Neon Pool (long-running, Node.js)
import { Pool, neonConfig } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-serverless'
import ws from 'ws'
import * as schema from './schema'

neonConfig.webSocketConstructor = ws
const pool = new Pool({ connectionString: process.env.DATABASE_URL! })
export const db = drizzle({ client: pool, schema })
```

```bash
bun add drizzle-orm @neondatabase/serverless
bun add -D drizzle-kit
```

## Schema Definition

```typescript
import { pgTable, serial, text, varchar, integer, boolean, timestamp,
         uuid, numeric, json, index, uniqueIndex, primaryKey } from 'drizzle-orm/pg-core'
import { pgEnum } from 'drizzle-orm/pg-core'
import { relations } from 'drizzle-orm'

// Enums
export const roleEnum = pgEnum('role', ['CUSTOMER', 'ADMIN'])
export const statusEnum = pgEnum('status', ['PENDING', 'ACTIVE', 'CANCELLED'])

// Tables
export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  email: varchar('email', { length: 255 }).notNull().unique(),
  role: roleEnum('role').default('CUSTOMER').notNull(),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
})

export const posts = pgTable('posts', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: text('title').notNull(),
  content: text('content'),
  authorId: uuid('author_id').references(() => users.id, { onDelete: 'cascade' }).notNull(),
  tags: json('tags').$type<string[]>().notNull().default([]),
  price: numeric('price', { precision: 10, scale: 2 }),
  published: boolean('published').notNull().default(false),
  createdAt: timestamp('created_at').defaultNow(),
}, (table) => [
  index('posts_author_id_idx').on(table.authorId),
  index('posts_published_idx').on(table.published),
])

// Type inference
type User = typeof users.$inferSelect
type NewUser = typeof users.$inferInsert
```

See [references/schema-columns.md](references/schema-columns.md) for all column types, constraints, indexes, and enums.

## Relations

```typescript
export const usersRelations = relations(users, ({ many }) => ({
  posts: many(posts),
}))

export const postsRelations = relations(posts, ({ one, many }) => ({
  author: one(users, { fields: [posts.authorId], references: [users.id] }),
  comments: many(comments),
}))

// Self-referencing (category tree)
export const categoriesRelations = relations(categories, ({ one, many }) => ({
  parent: one(categories, {
    fields: [categories.parentId], references: [categories.id],
    relationName: 'categoryTree',
  }),
  children: many(categories, { relationName: 'categoryTree' }),
}))

// Many-to-many (join table)
export const usersToGroups = pgTable('users_to_groups', {
  userId: uuid('user_id').notNull().references(() => users.id),
  groupId: uuid('group_id').notNull().references(() => groups.id),
}, (t) => [primaryKey({ columns: [t.userId, t.groupId] })])
```

## CRUD Queries

```typescript
import { eq, and, or, gt, like, ilike, inArray, between, desc, sql } from 'drizzle-orm'

// Select
const allUsers = await db.select().from(users)
const user = await db.select().from(users).where(eq(users.id, id)).limit(1)

// Partial select
const names = await db.select({ id: users.id, name: users.name }).from(users)

// Insert + returning
const [newUser] = await db.insert(users).values({ name: 'Dan', email: 'dan@test.com' }).returning()

// Update + returning
const [updated] = await db.update(users).set({ name: 'Daniel' }).where(eq(users.id, id)).returning()

// Delete
await db.delete(users).where(eq(users.id, id))

// Filters
await db.select().from(users).where(
  and(
    eq(users.role, 'ADMIN'),
    gt(users.createdAt, new Date('2024-01-01')),
    or(like(users.name, '%Dan%'), ilike(users.email, '%@gmail.com'))
  )
).orderBy(desc(users.createdAt)).limit(10).offset(20)
```

See [references/queries-crud.md](references/queries-crud.md) for all filter operators, ordering, pagination, and aggregations.

## Relational Queries (`db.query`)

```typescript
// findFirst — single record with nested relations
const user = await db.query.users.findFirst({
  where: eq(users.id, userId),
  with: {
    posts: {
      where: eq(posts.published, true),
      orderBy: (posts, { desc }) => [desc(posts.createdAt)],
      limit: 5,
      with: { comments: true },
    },
  },
})

// findMany — multiple records
const activeUsers = await db.query.users.findMany({
  where: eq(users.role, 'ADMIN'),
  columns: { id: true, name: true, email: true },
  with: { posts: { columns: { id: true, title: true } } },
})
```

See [references/relational-queries.md](references/relational-queries.md) for nested with, column selection, ordering, and v2 filter syntax.

## Transactions

```typescript
const result = await db.transaction(async (tx) => {
  const [user] = await tx.insert(users).values(userData).returning()
  await tx.insert(profiles).values({ userId: user.id, ...profileData })
  return user
})
```

**Transaction-or-DB type pattern** (pass `tx` through service layers):
```typescript
type TxCallback = Parameters<typeof db.transaction>[0]
export type Tx = Parameters<TxCallback>[0] | typeof db

async function getUser(id: string, tx: Tx = db) {
  return tx.query.users.findFirst({ where: eq(users.id, id) })
}

// Works with both db and tx:
await getUser(id)                            // uses db
await db.transaction(async (tx) => getUser(id, tx))  // uses tx
```

See [references/advanced-patterns.md](references/advanced-patterns.md) for joins, subqueries, aggregations, raw SQL, and performance patterns.

## Migrations

```typescript
// drizzle.config.ts
import { defineConfig } from 'drizzle-kit'

export default defineConfig({
  dialect: 'postgresql',
  schema: './src/db/schema/index.ts',
  out: './drizzle/migrations',
  dbCredentials: { url: process.env.DATABASE_URL! },
})
```

```bash
bunx drizzle-kit generate   # Generate migration SQL from schema changes
bunx drizzle-kit push        # Push schema directly (dev only, no migration files)
bunx drizzle-kit migrate     # Run pending migrations
bunx drizzle-kit studio      # Visual browser UI for your database
```

See [references/neon-migrations.md](references/neon-migrations.md) for Neon connection patterns, pool config, migration workflows, and drizzle-kit options.

## Common Pitfalls

- **Relations are NOT foreign keys.** `relations()` only defines query-time joins for `db.query`. You still need `.references()` on columns for actual FK constraints.
- **Schema must be passed to `drizzle()`.** Without `{ schema }`, `db.query` will not work — only the SQL builder API.
- **`serial` is deprecated.** Prefer `integer().primaryKey().generatedAlwaysAsIdentity()` or `uuid().defaultRandom()`.
- **JSON columns need `$type<T>()`** to get type safety: `json('data').$type<MyType>()`.
- **`numeric`/`decimal` returns strings.** Drizzle returns `string` for precision types — parse with `Number()` or `parseFloat()` in application code.
- **`defaultNow()` is SQL-level.** It only runs on INSERT, not on UPDATE. Manually set `updatedAt: new Date()` in `.set()`.
- **Neon HTTP vs Pool:** Use `neon-http` for serverless/edge (single queries), `neon-serverless` with `Pool` for long-running servers needing transactions.
- **`onDelete: 'cascade'` goes on `.references()`**, not on relations: `.references(() => users.id, { onDelete: 'cascade' })`.
- **Filters are imported from `drizzle-orm`**, not from `drizzle-orm/pg-core`. Columns/tables come from `pg-core`, operators from the root.
- **Use `DATABASE_URL_UNPOOLED` for migrations.** Neon's pooled connection string doesn't support DDL — use the unpooled URL in `drizzle.config.ts`.

## References

- [references/schema-columns.md](references/schema-columns.md) — All column types, constraints, indexes, enums, composite keys, and type inference
- [references/queries-crud.md](references/queries-crud.md) — Select, insert, update, delete, all filter operators, ordering, pagination, aggregations
- [references/relational-queries.md](references/relational-queries.md) — `db.query` API, findFirst/findMany, nested with, column selection, v2 filters
- [references/advanced-patterns.md](references/advanced-patterns.md) — Joins, subqueries, transactions, raw SQL, Tx pattern, exists, count, performance
- [references/neon-migrations.md](references/neon-migrations.md) — Neon HTTP/Pool setup, drizzle-kit CLI, migration workflows, connection retry, pool stats
