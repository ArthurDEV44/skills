# Schema, Columns, Constraints & Indexes

## Column Types (PostgreSQL)

### Numeric
```typescript
import { integer, smallint, bigint, serial, smallserial, bigserial,
         numeric, real, doublePrecision } from 'drizzle-orm/pg-core'

integer('age')                                    // int4
smallint('count')                                 // int2
bigint('views', { mode: 'number' })               // int8 as number (safe up to 2^53)
bigint('views', { mode: 'bigint' })               // int8 as BigInt
serial('id').primaryKey()                         // auto-increment (deprecated, prefer identity)
integer('id').primaryKey().generatedAlwaysAsIdentity()  // GENERATED ALWAYS AS IDENTITY (preferred)
numeric('price', { precision: 10, scale: 2 })     // decimal — returns string
real('score')                                     // float4
doublePrecision('precise')                        // float8
```

### String
```typescript
import { text, varchar, char } from 'drizzle-orm/pg-core'

text('bio')                                       // unlimited text
varchar('name', { length: 255 })                  // varchar(255)
char('code', { length: 2 })                       // char(2)
```

### Boolean & Date
```typescript
import { boolean, timestamp, date, time, interval } from 'drizzle-orm/pg-core'

boolean('is_active')                              // bool
timestamp('created_at')                           // timestamp without tz
timestamp('created_at', { withTimezone: true })    // timestamptz
timestamp('created_at', { mode: 'string' })        // returns ISO string instead of Date
date('birth_date')                                // date
date('birth_date', { mode: 'string' })             // returns string
time('start_time')                                // time
interval('duration')                              // interval
```

### UUID
```typescript
import { uuid } from 'drizzle-orm/pg-core'

uuid('id').primaryKey().defaultRandom()           // gen_random_uuid()
uuid('id').primaryKey().default(sql`uuid_generate_v4()`)
```

### JSON
```typescript
import { json, jsonb } from 'drizzle-orm/pg-core'

json('metadata')                                  // json (stored as-is)
jsonb('metadata')                                 // jsonb (indexed, queryable)

// Type-safe JSON — critical for type inference
json('tags').$type<string[]>().notNull().default([])
json('settings').$type<{ theme: string; lang: string }>()
jsonb('citations').$type<Array<{ text: string; url?: string }>>()
```

### Arrays
```typescript
import { text, integer } from 'drizzle-orm/pg-core'

text('tags').array()                              // text[]
integer('scores').array()                         // int4[]
```

### Custom ID Generators
```typescript
import { createId } from '@paralleldrive/cuid2'

// CUID2
id: text('id').primaryKey().$defaultFn(() => createId()),

// UUID
id: uuid('id').primaryKey().defaultRandom(),

// Crypto random
id: text('id').primaryKey().$defaultFn(() => crypto.randomUUID()),
```

## Column Modifiers

```typescript
column.notNull()                                  // NOT NULL
column.default(value)                             // DEFAULT value
column.defaultNow()                               // DEFAULT now() — timestamps only
column.unique()                                   // UNIQUE constraint
column.primaryKey()                               // PRIMARY KEY
column.references(() => otherTable.id)            // FOREIGN KEY
column.references(() => otherTable.id, {
  onDelete: 'cascade',                            // CASCADE | SET NULL | SET DEFAULT | RESTRICT | NO ACTION
  onUpdate: 'cascade',
})
column.generatedAlwaysAs(sql`...`)                // GENERATED ALWAYS AS (expression) STORED
column.$defaultFn(() => value)                    // JS-side default (runs before INSERT)
column.$type<MyType>()                            // Override inferred TypeScript type
```

## Enums

```typescript
import { pgEnum } from 'drizzle-orm/pg-core'

export const roleEnum = pgEnum('role', ['CUSTOMER', 'ADMIN', 'SUPER_ADMIN'])
export const orderStatusEnum = pgEnum('order_status', [
  'PENDING', 'PAID', 'PROCESSING', 'SHIPPED', 'DELIVERED', 'CANCELLED', 'REFUNDED'
])

// Usage
role: roleEnum('role').default('CUSTOMER').notNull(),
status: orderStatusEnum('status').default('PENDING').notNull(),

// TypeScript type from enum
type Role = (typeof roleEnum.enumValues)[number]  // 'CUSTOMER' | 'ADMIN' | 'SUPER_ADMIN'
```

## Indexes & Constraints

```typescript
import { pgTable, index, uniqueIndex, primaryKey, unique } from 'drizzle-orm/pg-core'
import { sql } from 'drizzle-orm'

export const products = pgTable('products', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  slug: varchar('slug', { length: 255 }).notNull(),
  categoryId: uuid('category_id').references(() => categories.id).notNull(),
  price: numeric('price', { precision: 10, scale: 2 }).notNull(),
}, (table) => [
  // Simple index
  index('products_category_id_idx').on(table.categoryId),

  // Unique index
  uniqueIndex('products_slug_idx').on(table.slug),

  // Composite unique constraint
  unique('products_name_category_unique').on(table.name, table.categoryId),

  // Composite primary key (join tables)
  // primaryKey({ columns: [table.userId, table.groupId] }),

  // Case-insensitive index (custom SQL)
  uniqueIndex('products_slug_lower_idx').on(sql`lower(${table.slug})`),

  // Partial index
  index('products_active_idx').on(table.categoryId).where(sql`is_active = true`),
])
```

**Helper for case-insensitive operations:**
```typescript
import { type AnyPgColumn, sql, type SQL } from 'drizzle-orm'

export function lower(column: AnyPgColumn): SQL {
  return sql`lower(${column})`
}

// Usage: uniqueIndex('name_unique').on(lower(table.name))
```

## Composite Primary Keys

```typescript
import { primaryKey } from 'drizzle-orm/pg-core'

// Many-to-many join table
export const usersToGroups = pgTable('users_to_groups', {
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  groupId: uuid('group_id').notNull().references(() => groups.id, { onDelete: 'cascade' }),
}, (table) => [
  primaryKey({ columns: [table.userId, table.groupId] }),
])
```

## Type Inference

```typescript
// Select type — all columns, nullables respected
type User = typeof users.$inferSelect
// { id: string; name: string; email: string; role: 'CUSTOMER' | 'ADMIN'; createdAt: Date | null }

// Insert type — optionals for defaults, required for notNull without default
type NewUser = typeof users.$inferInsert
// { id?: string; name: string; email: string; role?: 'CUSTOMER' | 'ADMIN'; createdAt?: Date }
```

## Schema Organization

Recommended file structure for projects with many tables:

```
src/db/
  index.ts          # drizzle() init + db export
  schema/
    index.ts        # re-exports all tables, relations, enums
    enums.ts        # pgEnum definitions
    users.ts        # users table + usersRelations
    products.ts     # products table + productsRelations
    orders.ts       # orders table + ordersRelations
```

```typescript
// src/db/schema/index.ts
export * from './enums'
export * from './users'
export * from './products'
export * from './orders'
```
