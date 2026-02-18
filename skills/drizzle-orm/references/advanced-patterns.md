# Advanced Patterns: Joins, Subqueries, Transactions, Raw SQL

## Joins

```typescript
import { eq, sql } from 'drizzle-orm'

// Inner join
const result = await db
  .select({
    postId: posts.id,
    postTitle: posts.title,
    authorName: users.name,
  })
  .from(posts)
  .innerJoin(users, eq(posts.authorId, users.id))

// Left join (nullable joined columns)
const result = await db
  .select({
    userId: users.id,
    userName: users.name,
    postTitle: posts.title, // string | null
  })
  .from(users)
  .leftJoin(posts, eq(users.id, posts.authorId))

// Multiple joins
const result = await db
  .select({
    postTitle: posts.title,
    authorName: users.name,
    categoryName: categories.name,
  })
  .from(posts)
  .innerJoin(users, eq(posts.authorId, users.id))
  .leftJoin(categories, eq(posts.categoryId, categories.id))
```

## Subqueries

```typescript
// Subquery as a derived table
const mainImageSq = db
  .select({
    productId: productImages.productId,
    url: productImages.url,
  })
  .from(productImages)
  .where(eq(productImages.isMain, true))
  .as('main_image')  // .as() creates a named subquery

const variantStatsSq = db
  .select({
    productId: productVariants.productId,
    variantCount: count().as('variant_count'),
    totalStock: sum(productVariants.stock).as('total_stock'),
    minPrice: min(productVariants.price).as('min_price'),
    maxPrice: max(productVariants.price).as('max_price'),
  })
  .from(productVariants)
  .groupBy(productVariants.productId)
  .as('variant_stats')

// Join subqueries
const products = await db
  .select({
    id: productsTable.id,
    name: productsTable.name,
    mainImage: mainImageSq.url,
    variantCount: variantStatsSq.variantCount,
    minPrice: variantStatsSq.minPrice,
  })
  .from(productsTable)
  .leftJoin(mainImageSq, eq(productsTable.id, mainImageSq.productId))
  .leftJoin(variantStatsSq, eq(productsTable.id, variantStatsSq.productId))
  .orderBy(desc(productsTable.createdAt))
  .limit(20)
```

## Exists Subquery

```typescript
import { exists, sql } from 'drizzle-orm'

// Find users who have at least one published post
const usersWithPosts = await db
  .select()
  .from(users)
  .where(
    exists(
      db.select({ one: sql`1` })
        .from(posts)
        .where(
          and(
            eq(posts.authorId, users.id),
            eq(posts.published, true)
          )
        )
    )
  )

// Search across related tables with exists
function buildSearchCondition(query: string): SQL {
  const pattern = `%${escapeLike(query)}%`
  return or(
    ilike(products.name, pattern),
    ilike(products.description, pattern),
    exists(
      db.select({ one: sql`1` })
        .from(productVariants)
        .where(
          and(
            eq(productVariants.productId, products.id),
            or(
              ilike(productVariants.sku, pattern),
              ilike(productVariants.name, pattern),
            )
          )
        )
    ),
  )!
}
```

## Transactions

```typescript
// Basic transaction
const result = await db.transaction(async (tx) => {
  const [user] = await tx.insert(users).values(data).returning()
  await tx.insert(profiles).values({ userId: user.id, bio: '' })
  return user
})

// Transaction with rollback
await db.transaction(async (tx) => {
  await tx.insert(orders).values(orderData)
  const stock = await tx.select().from(inventory).where(eq(inventory.productId, pid))
  if (stock[0].quantity < 1) {
    tx.rollback()  // explicitly rollback — throws, exits transaction
  }
  await tx.update(inventory).set({ quantity: sql`quantity - 1` }).where(eq(inventory.productId, pid))
})

// Nested transactions (savepoints)
await db.transaction(async (tx) => {
  await tx.insert(users).values(userData)
  // savepoint — if inner fails, only inner rolls back
  await tx.transaction(async (tx2) => {
    await tx2.insert(logs).values(logData)
  })
})
```

## Transaction-or-DB Type (Tx Pattern)

Allow services to work with or without a transaction context:

```typescript
// Define the Tx type
type TxCallback = Parameters<typeof db.transaction>[0]
export type Tx = Parameters<TxCallback>[0] | typeof db

// Service methods accept optional tx
export class UserService {
  static async getById(id: string, tx: Tx = db) {
    return tx.query.users.findFirst({ where: eq(users.id, id) })
  }

  static async create(data: NewUser, tx: Tx = db) {
    const [user] = await tx.insert(users).values(data).returning()
    return user
  }

  static async updateEmail(id: string, email: string, tx: Tx = db) {
    const [updated] = await tx.update(users)
      .set({ email, updatedAt: new Date() })
      .where(eq(users.id, id))
      .returning()
    return updated
  }
}

// Standalone
const user = await UserService.getById(id)

// Inside transaction
await db.transaction(async (tx) => {
  const user = await UserService.getById(id, tx)
  await UserService.updateEmail(id, 'new@email.com', tx)
  await AuditService.log('email_changed', user.id, tx)
})
```

## Raw SQL

```typescript
import { sql } from 'drizzle-orm'

// Typed raw query
const result = await db.execute<{ count: number }>(
  sql`SELECT count(*)::int as count FROM users WHERE role = 'ADMIN'`
)

// sql template with table references
await db.execute(sql`
  UPDATE ${products}
  SET ${products.views} = ${products.views} + 1
  WHERE ${products.id} = ${productId}
`)

// sql in select fields
const result = await db.select({
  id: users.id,
  fullName: sql<string>`${users.firstName} || ' ' || ${users.lastName}`,
  daysSinceCreation: sql<number>`extract(day from now() - ${users.createdAt})::int`,
}).from(users)

// Bulk operations with raw SQL
await db.execute(sql`
  DELETE FROM ${sessions}
  WHERE ${sessions.expiresAt} < now() - interval '30 days'
`)

// Array parameters in raw SQL
const ids = [1, 2, 3]
await db.execute(sql`
  UPDATE ${users}
  SET ${users.role} = 'VERIFIED'
  WHERE ${users.id} = ANY(${ids})
`)
```

## Prepared Statements

```typescript
const getUser = db
  .select()
  .from(users)
  .where(eq(users.id, sql.placeholder('id')))
  .prepare('get_user')

const user = await getUser.execute({ id: userId })

// Reusable prepared statement
const searchUsers = db
  .select()
  .from(users)
  .where(ilike(users.name, sql.placeholder('pattern')))
  .limit(sql.placeholder('limit'))
  .prepare('search_users')

const results = await searchUsers.execute({ pattern: '%Dan%', limit: 10 })
```

## Performance Patterns

### Parallel Queries
```typescript
const [products, totalCount, categories] = await Promise.all([
  db.select().from(productsTable)
    .where(whereClause)
    .orderBy(orderBy)
    .limit(pageSize)
    .offset(offset),
  db.select({ count: sql<number>`count(*)::int` }).from(productsTable).where(whereClause),
  db.select().from(categoriesTable).orderBy(asc(categoriesTable.name)),
])
```

### Batch Insert with Chunks
```typescript
async function batchInsert<T>(table: any, data: T[], chunkSize = 500) {
  for (let i = 0; i < data.length; i += chunkSize) {
    const chunk = data.slice(i, i + chunkSize)
    await db.insert(table).values(chunk)
  }
}
```

### Avoiding N+1 with Relational Queries
```typescript
// BAD: N+1 queries
const users = await db.select().from(usersTable)
for (const user of users) {
  const posts = await db.select().from(postsTable).where(eq(postsTable.authorId, user.id))
}

// GOOD: Single query with relations
const users = await db.query.users.findMany({
  with: { posts: true },
})

// GOOD: Two queries with inArray
const users = await db.select().from(usersTable)
const userIds = users.map(u => u.id)
const posts = await db.select().from(postsTable).where(inArray(postsTable.authorId, userIds))
```

### Connection Retry for Neon
```typescript
export async function withRetry<T>(fn: () => Promise<T>, attempts = 2): Promise<T> {
  try {
    return await fn()
  } catch (err: any) {
    if (attempts <= 1 || !isRetryableError(err)) throw err
    await new Promise(r => setTimeout(r, 500))
    return withRetry(fn, attempts - 1)
  }
}

function isRetryableError(err: any): boolean {
  const msg = err?.message ?? ''
  return msg.includes('Connection terminated') ||
         msg.includes('ECONNRESET') ||
         msg.includes('socket hang up')
}
```
