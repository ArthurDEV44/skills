# Relational Queries (`db.query`)

The relational query API provides an intuitive way to load nested relations without manual joins. It requires passing `{ schema }` when initializing drizzle.

## Setup

```typescript
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema' // must export tables AND relations

const db = drizzle(sql, { schema })
// db.query.users, db.query.posts, etc. are now available
```

## findFirst

Returns a single record or `undefined`.

```typescript
const user = await db.query.users.findFirst({
  where: eq(users.id, userId),
})

// With nested relations
const user = await db.query.users.findFirst({
  where: eq(users.id, userId),
  with: {
    posts: true,                    // include all posts
    profile: true,                  // include profile (one-to-one)
  },
})

// With filtered and ordered relations
const user = await db.query.users.findFirst({
  where: eq(users.id, userId),
  with: {
    posts: {
      where: eq(posts.published, true),
      orderBy: (posts, { desc }) => [desc(posts.createdAt)],
      limit: 5,
      columns: { id: true, title: true, createdAt: true },
    },
  },
})
```

## findMany

Returns an array (empty if no matches).

```typescript
// Basic
const allUsers = await db.query.users.findMany()

// With filtering and pagination
const admins = await db.query.users.findMany({
  where: eq(users.role, 'ADMIN'),
  orderBy: [desc(users.createdAt)],
  limit: 20,
  offset: 0,
})

// Column selection (include only specific columns)
const names = await db.query.users.findMany({
  columns: { id: true, name: true, email: true },
})

// Column exclusion
const withoutSecrets = await db.query.users.findMany({
  columns: { passwordHash: false },
})
```

## Deep Nesting

```typescript
const product = await db.query.products.findFirst({
  where: eq(products.id, productId),
  with: {
    category: true,
    variants: {
      orderBy: (variants, { asc }) => [asc(variants.price)],
      with: {
        inventory: true,
      },
    },
    images: {
      orderBy: (images, { asc }) => [asc(images.order)],
    },
    reviews: {
      limit: 10,
      orderBy: (reviews, { desc }) => [desc(reviews.createdAt)],
      with: {
        author: { columns: { id: true, name: true } },
      },
    },
  },
})
```

## Columns Selection in Relations

```typescript
const conversations = await db.query.conversations.findMany({
  where: eq(conversations.userId, userId),
  columns: {
    id: true,
    title: true,
    createdAt: true,
  },
  with: {
    messages: {
      columns: { id: true, content: true, role: true },
      orderBy: (m, { asc }) => [asc(m.createdAt)],
    },
  },
  orderBy: (c, { desc }) => [desc(c.updatedAt)],
})
```

## Where with Callback

```typescript
const users = await db.query.users.findMany({
  where: (users, { eq, and, gt }) =>
    and(eq(users.role, 'ADMIN'), gt(users.age, 18)),
})
```

## Extras (Computed Fields)

```typescript
import { sql } from 'drizzle-orm'

const users = await db.query.users.findMany({
  extras: {
    fullName: sql<string>`${users.firstName} || ' ' || ${users.lastName}`.as('full_name'),
    postCount: sql<number>`(
      SELECT count(*) FROM posts WHERE posts.author_id = users.id
    )::int`.as('post_count'),
  },
})
```

## v2 Relational Queries (Object Syntax)

Drizzle v2 introduces an object-based filter syntax for relational queries:

```typescript
// v2 object syntax
const result = await db.query.users.findMany({
  where: {
    role: { eq: 'ADMIN' },
    age: { gte: 18 },
  },
  with: {
    posts: {
      where: { published: { eq: true } },
    },
  },
})

// Complex v2 filters
const result = await db.query.users.findMany({
  where: {
    AND: [
      { OR: [{ name: { ilike: 'john%' } }, { name: { ilike: 'jane%' } }] },
      { age: { gte: 18, lte: 65 } },
      { NOT: { status: { eq: 'banned' } } },
    ],
  },
})

// Filter by relation existence
const usersWithPosts = await db.query.users.findMany({
  where: {
    posts: { id: { isNotNull: true } },  // users that have at least one post
  },
})

// RAW SQL in v2
const result = await db.query.users.findMany({
  where: {
    RAW: (table) => sql`${table.age} BETWEEN 25 AND 35`,
  },
})
```

## Relation Types Reference

| Relation | Usage | Notes |
|----------|-------|-------|
| `one()` | One-to-one, many-to-one | Requires `fields` and `references` on the side with the FK |
| `many()` | One-to-many | No fields/references needed (inferred from the `one()` side) |
| `relationName` | Disambiguate multiple relations to same table | Required for self-referencing |

## Important Notes

- `db.query` requires `{ schema }` passed to `drizzle()` — both tables AND relations must be exported
- Relations are **query-time only** — they don't create foreign keys in the database
- `findFirst` returns `T | undefined`, not `T | null`
- Ordering in `with` uses a callback: `orderBy: (table, { asc, desc }) => [desc(table.column)]`
- Column selection with `columns: { ... }` uses boolean flags (true to include, false to exclude)
- You cannot mix include and exclude in the same `columns` object
