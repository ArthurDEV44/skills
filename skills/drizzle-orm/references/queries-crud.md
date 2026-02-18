# CRUD Queries & Filter Operators

## Select

```typescript
import { eq, ne, gt, gte, lt, lte, and, or, not, like, ilike, notLike,
         inArray, notInArray, between, isNull, isNotNull, exists,
         desc, asc, sql, count, sum, min, max, avg } from 'drizzle-orm'

// All rows
const allUsers = await db.select().from(users)

// Single row
const [user] = await db.select().from(users).where(eq(users.id, id))

// Partial select (only specific columns)
const result = await db.select({
  id: users.id,
  name: users.name,
  lowerName: sql<string>`lower(${users.name})`,
}).from(users)

// Distinct
await db.selectDistinct().from(users).orderBy(users.name)
await db.selectDistinctOn([users.email]).from(users)
```

## Filter Operators

All imported from `drizzle-orm` (NOT from `drizzle-orm/pg-core`).

```typescript
// Comparison
eq(users.id, 1)                       // = 1
ne(users.status, 'inactive')          // != 'inactive'
gt(users.age, 18)                     // > 18
gte(users.age, 21)                    // >= 21
lt(users.age, 65)                     // < 65
lte(users.age, 100)                   // <= 100

// Null checks
isNull(users.deletedAt)               // IS NULL
isNotNull(users.email)                // IS NOT NULL

// Array operations
inArray(users.role, ['admin', 'mod']) // IN ('admin', 'mod')
notInArray(users.status, ['banned'])  // NOT IN ('banned')

// Pattern matching
like(users.name, '%Dan%')             // LIKE '%Dan%'  (case-sensitive)
ilike(users.email, '%@gmail.com')     // ILIKE '%@gmail.com'  (case-insensitive)
notLike(users.name, 'test%')          // NOT LIKE 'test%'

// Range
between(users.age, 18, 65)           // BETWEEN 18 AND 65

// Logical
and(eq(users.role, 'admin'), gt(users.age, 18))
or(eq(users.role, 'admin'), eq(users.role, 'mod'))
not(eq(users.status, 'banned'))

// Exists subquery
exists(
  db.select({ one: sql`1` })
    .from(posts)
    .where(eq(posts.authorId, users.id))
)
```

## Dynamic Filter Building

```typescript
import { type SQL } from 'drizzle-orm'

function buildFilters(params: {
  search?: string
  role?: string
  minAge?: number
}): SQL | undefined {
  const conditions: SQL[] = []

  if (params.search) {
    const pattern = `%${params.search}%`
    conditions.push(or(
      ilike(users.name, pattern),
      ilike(users.email, pattern),
    )!)
  }
  if (params.role) conditions.push(eq(users.role, params.role))
  if (params.minAge) conditions.push(gte(users.age, params.minAge))

  return conditions.length > 0 ? and(...conditions) : undefined
}

// Usage
const where = buildFilters({ search: 'dan', role: 'admin' })
await db.select().from(users).where(where)
```

## Ordering & Pagination

```typescript
// Single column
await db.select().from(users).orderBy(desc(users.createdAt))

// Multiple columns
await db.select().from(users).orderBy(asc(users.name), desc(users.createdAt))

// Pagination
await db.select().from(users)
  .orderBy(desc(users.createdAt))
  .limit(20)
  .offset(40)

// Cursor-based pagination (better perf)
await db.select().from(users)
  .where(lt(users.createdAt, cursorDate))
  .orderBy(desc(users.createdAt))
  .limit(20)
```

## Insert

```typescript
// Single insert
const [newUser] = await db.insert(users)
  .values({ name: 'Dan', email: 'dan@test.com' })
  .returning()

// Bulk insert
await db.insert(users)
  .values([
    { name: 'Alice', email: 'alice@test.com' },
    { name: 'Bob', email: 'bob@test.com' },
  ])
  .returning()

// Upsert (ON CONFLICT)
await db.insert(users)
  .values({ name: 'Dan', email: 'dan@test.com' })
  .onConflictDoUpdate({
    target: users.email,
    set: { name: 'Dan Updated', updatedAt: new Date() },
  })
  .returning()

await db.insert(users)
  .values({ name: 'Dan', email: 'dan@test.com' })
  .onConflictDoNothing({ target: users.email })

// Returning specific columns
const [{ id }] = await db.insert(users)
  .values(data)
  .returning({ id: users.id })
```

## Update

```typescript
// Update by ID
const [updated] = await db.update(users)
  .set({ name: 'Daniel', updatedAt: new Date() })
  .where(eq(users.id, id))
  .returning()

// Conditional update
await db.update(users)
  .set({ role: 'ADMIN' })
  .where(
    and(
      eq(users.email, 'admin@company.com'),
      ne(users.role, 'ADMIN')
    )
  )

// Increment with SQL
await db.update(posts)
  .set({ views: sql`${posts.views} + 1` })
  .where(eq(posts.id, postId))
```

## Delete

```typescript
// Delete by ID
await db.delete(users).where(eq(users.id, id))

// Delete with returning
const [deleted] = await db.delete(users)
  .where(eq(users.id, id))
  .returning({ id: users.id })

// Bulk delete
await db.delete(sessions)
  .where(lt(sessions.expiresAt, new Date()))
```

## Aggregations

```typescript
// Count
const [{ total }] = await db
  .select({ total: count() })
  .from(users)

// Typed count
const [{ total }] = await db
  .select({ total: sql<number>`count(*)::int` })
  .from(users)
  .where(eq(users.role, 'ADMIN'))

// Group by with aggregates
const stats = await db
  .select({
    categoryId: products.categoryId,
    productCount: count(),
    totalStock: sum(productVariants.stock),
    minPrice: min(productVariants.price),
    maxPrice: max(productVariants.price),
    avgPrice: avg(productVariants.price),
  })
  .from(products)
  .leftJoin(productVariants, eq(products.id, productVariants.productId))
  .groupBy(products.categoryId)

// Having clause
const popular = await db
  .select({
    authorId: posts.authorId,
    postCount: count(),
  })
  .from(posts)
  .groupBy(posts.authorId)
  .having(sql`count(*) > 5`)
```

## Like Pattern Escaping

Always escape user input for LIKE queries:

```typescript
function escapeLike(value: string): string {
  return value.replace(/[%_\\]/g, '\\$&')
}

const pattern = `%${escapeLike(userInput)}%`
await db.select().from(users).where(ilike(users.name, pattern))
```
