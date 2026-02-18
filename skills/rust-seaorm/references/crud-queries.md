# SeaORM CRUD & Queries Reference

## Table of Contents
- [Basic CRUD](#basic-crud)
- [Custom Select](#custom-select)
- [Partial Models](#partial-models)
- [Conditional Expressions](#conditional-expressions)
- [Aggregate Functions](#aggregate-functions)
- [Join Patterns](#join-patterns)
- [Subqueries](#subqueries)
- [Pagination](#pagination)
- [Streaming](#streaming)
- [Raw SQL](#raw-sql)

---

## Basic CRUD

### Find (SELECT)

```rust
use sea_orm::entity::prelude::*;

// All records
let cakes: Vec<cake::Model> = Cake::find().all(db).await?;

// Single by ID
let cake: Option<cake::Model> = Cake::find_by_id(1).one(db).await?;

// Composite key
let model = CakeFilling::find_by_id((1, 2)).one(db).await?;

// With filter
let cakes = Cake::find()
    .filter(cake::Column::Name.contains("chocolate"))
    .all(db).await?;

// With ordering
let cakes = Cake::find()
    .order_by_asc(cake::Column::Name)
    .order_by_desc(cake::Column::Id)
    .all(db).await?;

// With limit/offset
let cakes = Cake::find()
    .limit(10)
    .offset(20)
    .all(db).await?;

// Count
let count: u64 = Cake::find().count(db).await?;
```

### Insert (INSERT)

```rust
use sea_orm::ActiveValue::{Set, NotSet};

// Single insert
let model = fruit::ActiveModel {
    name: Set("Apple".to_owned()),
    cake_id: Set(Some(1)),
    ..Default::default()  // id: NotSet (auto-generated)
};
let result: fruit::Model = model.insert(db).await?;
// result.id is now populated

// Insert and get last_insert_id only (no RETURNING)
let res: InsertResult = Fruit::insert(model).exec(db).await?;
println!("id: {}", res.last_insert_id);

// Insert with RETURNING (Postgres)
let res: fruit::Model = Fruit::insert(model)
    .exec_with_returning(db).await?;

// Bulk insert
let apple = fruit::ActiveModel { name: Set("Apple".to_owned()), ..Default::default() };
let pear = fruit::ActiveModel { name: Set("Pear".to_owned()), ..Default::default() };
let res = Fruit::insert_many([apple, pear]).exec(db).await?;

// Bulk insert with RETURNING
let models: Vec<fruit::Model> = Fruit::insert_many([apple, pear])
    .exec_with_returning(db).await?;

// Insert from JSON
let mut model: fruit::ActiveModel = Default::default();
model.set_from_json(serde_json::json!({
    "name": "Apple",
    "cake_id": 1
}))?;
let result = model.save(db).await?;
```

### Update (UPDATE)

```rust
// Update single record
let mut pear: fruit::ActiveModel = pear_model.into();
pear.name = Set("Sweet pear".to_owned());
let pear: fruit::Model = pear.update(db).await?;

// Bulk update
let res: UpdateResult = Fruit::update_many()
    .col_expr(fruit::Column::CakeId, Expr::value(1))
    .filter(fruit::Column::Name.contains("Apple"))
    .exec(db).await?;
println!("rows updated: {}", res.rows_affected);

// Update with arithmetic
Fruit::update_many()
    .col_expr(fruit::Column::CakeId, fruit::Column::CakeId.add(2))
    .filter(fruit::Column::Id.gt(10))
    .exec(db).await?;
```

### Save (INSERT or UPDATE)

```rust
// Insert: when PK is NotSet
let banana = fruit::ActiveModel {
    id: NotSet,
    name: Set("Banana".to_owned()),
    ..Default::default()
};
let mut banana: fruit::ActiveModel = banana.save(db).await?;
// banana.id is now Unchanged(new_id)

// Update: when PK is Set/Unchanged
banana.name = Set("Banana Mongo".to_owned());
let banana = banana.save(db).await?;
```

### Delete (DELETE)

```rust
// Delete by model
let res: DeleteResult = orange.delete(db).await?;
println!("rows deleted: {}", res.rows_affected);

// Delete by ID (validated - fails on non-existent)
let res = Fruit::delete_by_id(1).exec(db).await?;

// Bulk delete
let res = Fruit::delete_many()
    .filter(fruit::Column::Name.contains("Orange"))
    .exec(db).await?;
```

---

## Custom Select

### Select Specific Columns

```rust
// Clear defaults with select_only(), then add columns
cake::Entity::find()
    .select_only()
    .column(cake::Column::Name)
    .all(db).await?;

// Multiple columns
cake::Entity::find()
    .select_only()
    .columns([cake::Column::Id, cake::Column::Name])
    .all(db).await?;

// Dynamic column filtering
cake::Entity::find()
    .select_only()
    .columns(cake::Column::iter().filter(|col| match col {
        cake::Column::Id => false,
        _ => true,
    }))
    .all(db).await?;
```

### Custom Expressions

```rust
use sea_query::{Alias, Expr, Func};

cake::Entity::find()
    .column_as(
        Expr::col(cake::Column::Id).max().sub(Expr::col(cake::Column::Id)),
        "id_diff"
    )
    .column_as(Expr::cust("CURRENT_TIMESTAMP"), "current_time")
    .all(db).await?;

// Function-based
cake::Entity::find()
    .expr_as(
        Func::upper(Expr::col((cake::Entity, cake::Column::Name))),
        "name_upper"
    )
    .all(db).await?;
```

### FromQueryResult Custom Struct

```rust
#[derive(FromQueryResult)]
struct CakeAndFillingCount {
    id: i32,
    name: String,
    count: i64,
}

let results: Vec<CakeAndFillingCount> = cake::Entity::find()
    .column_as(filling::Column::Id.count(), "count")
    .join_rev(JoinType::InnerJoin, cake_filling::Relation::Cake.def())
    .join(JoinType::InnerJoin, cake_filling::Relation::Filling.def())
    .group_by(cake::Column::Id)
    .into_model::<CakeAndFillingCount>()
    .all(db).await?;
```

### Unstructured Tuples

```rust
let results: Vec<(String, i64)> = cake::Entity::find()
    .select_only()
    .column(cake::Column::Name)
    .column_as(cake::Column::Id.count(), "count")
    .group_by(cake::Column::Name)
    .into_tuple()
    .all(db).await?;
```

---

## Partial Models

Type-safe partial column selection without raw strings:

```rust
#[derive(DerivePartialModel, FromQueryResult)]
#[sea_orm(entity = "User")]
struct PartialUser {
    pub id: i32,
    pub avatar: String,
    pub unique_id: Uuid,
}

let users: Vec<PartialUser> = User::find()
    .into_partial_model::<PartialUser>()
    .all(db).await?;
// SELECT id, avatar, unique_id FROM user
```

### Column Remapping and Expressions

```rust
#[derive(DerivePartialModel, FromQueryResult)]
#[sea_orm(entity = "User")]
struct PartialRow {
    #[sea_orm(from_col = "id")]
    user_id: i32,
    #[sea_orm(from_expr = "Expr::col(user::Column::Id).add(1)")]
    next_id: i32,
}
```

---

## Conditional Expressions

### AND Conditions

```rust
use sea_orm::Condition;

cake::Entity::find()
    .filter(
        Condition::all()
            .add(cake::Column::Id.gte(1))
            .add(cake::Column::Name.like("%Cheese%"))
    )
    .all(db).await?;
// WHERE id >= 1 AND name LIKE '%Cheese%'
```

### OR Conditions

```rust
cake::Entity::find()
    .filter(
        Condition::any()
            .add(cake::Column::Id.eq(4))
            .add(cake::Column::Id.eq(5))
    )
    .all(db).await?;
// WHERE id = 4 OR id = 5
```

### Nested Conditions

```rust
cake::Entity::find()
    .filter(
        Condition::any()
            .add(
                Condition::all()
                    .add(cake::Column::Id.lte(30))
                    .add(cake::Column::Name.like("%Chocolate%"))
            )
            .add(
                Condition::all()
                    .add(cake::Column::Id.gte(1))
                    .add(cake::Column::Name.like("%Cheese%"))
            )
    )
// WHERE (id <= 30 AND name LIKE '%Chocolate%') OR (id >= 1 AND name LIKE '%Cheese%')
```

### Fluent Conditional Queries (apply_if)

Build queries conditionally based on optional parameters:

```rust
cake::Entity::find()
    .apply_if(Some(3), |mut query, v| {
        query.filter(cake::Column::Id.eq(v))
    })
    .apply_if(Some(100), QuerySelect::limit)
    .apply_if(None::<u64>, QuerySelect::offset)  // skipped since None
    .all(db).await?;
```

### Column Filter Methods

| Method | SQL |
|--------|-----|
| `col.eq(v)` | `= v` |
| `col.ne(v)` | `<> v` |
| `col.gt(v)` | `> v` |
| `col.gte(v)` | `>= v` |
| `col.lt(v)` | `< v` |
| `col.lte(v)` | `<= v` |
| `col.between(a, b)` | `BETWEEN a AND b` |
| `col.not_between(a, b)` | `NOT BETWEEN a AND b` |
| `col.like(s)` | `LIKE s` |
| `col.not_like(s)` | `NOT LIKE s` |
| `col.starts_with(s)` | `LIKE 's%'` |
| `col.ends_with(s)` | `LIKE '%s'` |
| `col.contains(s)` | `LIKE '%s%'` |
| `col.is_in(vec)` | `IN (...)` |
| `col.is_not_in(vec)` | `NOT IN (...)` |
| `col.is_null()` | `IS NULL` |
| `col.is_not_null()` | `IS NOT NULL` |

---

## Aggregate Functions

Available on `ColumnTrait`: `max`, `min`, `sum`, `avg`, `count`.

### Basic Aggregate

```rust
let sum: Decimal = order::Entity::find()
    .select_only()
    .column_as(order::Column::Total.sum(), "sum")
    .into_tuple()
    .one(db).await?.unwrap();
```

### Group By with Multiple Aggregates

```rust
#[derive(Debug, FromQueryResult)]
struct CustomerStats {
    name: String,
    num_orders: i64,
    total_spent: Decimal,
    min_spent: Decimal,
    max_spent: Decimal,
}

let stats: Vec<CustomerStats> = customer::Entity::find()
    .left_join(order::Entity)
    .select_only()
    .column(customer::Column::Name)
    .column_as(order::Column::Total.count(), "num_orders")
    .column_as(order::Column::Total.sum(), "total_spent")
    .column_as(order::Column::Total.min(), "min_spent")
    .column_as(order::Column::Total.max(), "max_spent")
    .group_by(customer::Column::Name)
    .into_model::<CustomerStats>()
    .all(db).await?;
```

### Having Clause

```rust
cake::Entity::find()
    .select_only()
    .column(cake::Column::Name)
    .column_as(cake::Column::Id.count(), "count")
    .group_by(cake::Column::Name)
    .having(cake::Column::Id.count().gt(6))
    .into_tuple::<(String, i64)>()
    .all(db).await?;
```

---

## Join Patterns

### Eager Loading (Relation-Based)

```rust
// 1-1: find_also_related (returns Option for related)
let pairs: Vec<(cake::Model, Option<fruit::Model>)> =
    Cake::find().find_also_related(Fruit).all(db).await?;

// 1-N / M-N: find_with_related (returns Vec for related)
let groups: Vec<(cake::Model, Vec<filling::Model>)> =
    Cake::find().find_with_related(Filling).all(db).await?;

// Lazy loading from model
let fruits: Vec<fruit::Model> = cake_model.find_related(Fruit).all(db).await?;
```

### Manual Join

```rust
use sea_orm::JoinType;

cake::Entity::find()
    .join(JoinType::LeftJoin, cake::Relation::Fruit.def())
    .all(db).await?;

// Reverse join
cake::Entity::find()
    .join_rev(JoinType::InnerJoin, cake_filling::Relation::Cake.def())
    .all(db).await?;
```

### Custom Join Conditions

```rust
// On-the-fly condition
cake::Entity::find()
    .join(
        JoinType::LeftJoin,
        cake_filling::Relation::Filling
            .def()
            .on_condition(|_left, right| {
                Expr::col((right, filling::Column::Name))
                    .like("%lemon%")
                    .into_condition()
            })
    )
    .all(db).await?;
```

### Table Aliases

```rust
#[derive(DeriveIden, Clone, Copy)]
pub struct Base;

// Join with alias
cake::Entity::find()
    .join_as(JoinType::LeftJoin, relation.def(), Base)
    .join(JoinType::LeftJoin, other_relation.def().from_alias(Base))
    .all(db).await?;

// Select from aliased table
cake::Entity::find()
    .select_only()
    .tbl_col_as((Base, base_product::Column::Id), "id")
    .tbl_col_as((Base, base_product::Column::Name), "name")
    .join_as(JoinType::InnerJoin, relation.def(), Base)
    .all(db).await?;
```

### Diamond Topology Joins

When two paths lead to the same table:

```rust
.join_as(JoinType::LeftJoin, complex_product::Relation::BaseProduct.def(), Base)
.join_as(JoinType::LeftJoin, complex_product::Relation::Material.def(), Material)
.join(JoinType::InnerJoin, base_product::Relation::Attribute.def().from_alias(Base))
.join(JoinType::InnerJoin, material::Relation::Attribute.def().from_alias(Material))
```

### Join Methods Summary

| Method | Usage |
|--------|-------|
| `join()` | Basic join using existing relation |
| `join_as()` | Join with table alias |
| `join_rev()` | Reverse join direction |
| `from_alias()` | Chain join from aliased table |

---

## Subqueries

### IN Subquery

```rust
use sea_query::Query;

cake::Entity::find()
    .filter(
        Condition::any().add(
            cake::Column::Id.in_subquery(
                Query::select()
                    .expr(cake::Column::Id.max())
                    .from(cake::Entity)
                    .to_owned()
            )
        )
    )
    .all(db).await?;
// WHERE id IN (SELECT MAX(id) FROM cake)
```

### NOT IN Subquery

```rust
cake::Entity::find()
    .filter(
        cake::Column::Id.not_in_subquery(
            Query::select()
                .column(fruit::Column::CakeId)
                .from(fruit::Entity)
                .to_owned()
        )
    )
    .all(db).await?;
// WHERE id NOT IN (SELECT cake_id FROM fruit)
```

---

## Pagination

### Offset-Based Pagination

```rust
use sea_orm::PaginatorTrait;

let paginator = Cake::find()
    .order_by_asc(cake::Column::Id)
    .paginate(db, 50); // 50 items per page

// Get total pages
let num_pages: u64 = paginator.num_pages().await?;

// Get total items
let num_items: u64 = paginator.num_items().await?;

// Fetch specific page (0-indexed)
let page_0: Vec<cake::Model> = paginator.fetch_page(0).await?;
let page_1: Vec<cake::Model> = paginator.fetch_page(1).await?;

// Iterate all pages
let mut pages = paginator.into_stream();
while let Some(cakes) = pages.try_next().await? {
    for cake in cakes {
        // process each cake
    }
}
```

### Cursor-Based Pagination

More efficient for large datasets (no COUNT/OFFSET):

```rust
use sea_orm::CursorTrait;

// Forward pagination
let mut cursor = Cake::find().cursor_by(cake::Column::Id);
cursor.after(0);  // start after id=0
let batch: Vec<cake::Model> = cursor.first(20).all(db).await?;

// Backward pagination
let mut cursor = Cake::find().cursor_by(cake::Column::Id);
cursor.before(100);  // before id=100
let batch: Vec<cake::Model> = cursor.last(20).all(db).await?;

// Composite cursor
let mut cursor = Cake::find()
    .cursor_by((cake::Column::Name, cake::Column::Id));
cursor.after(("Apple".to_owned(), 1));
let batch = cursor.first(20).all(db).await?;
```

---

## Streaming

Process large result sets without loading all into memory:

### Basic Stream

```rust
use futures::TryStreamExt;

let mut stream = Fruit::find().stream(db).await?;
while let Some(item) = stream.try_next().await? {
    let item: fruit::Model = item;
    // process each item
}
```

### Filtered Stream

```rust
let mut stream = Fruit::find()
    .filter(fruit::Column::Name.contains("a"))
    .order_by_asc(fruit::Column::Name)
    .stream(db).await?;
```

### Connection Pool Warning

Stream objects exclusively hold a connection until dropped. Multiple concurrent streams consume multiple pool connections:

```rust
{
    let s1 = Fruit::find().stream(db).await?;
    let s2 = Fruit::find().stream(db).await?;
    // 2 connections held simultaneously!
}
// connections returned to pool on drop
```

---

## Raw SQL

### Query with FromQueryResult

```rust
use sea_orm::{FromQueryResult, Statement, DbBackend};

#[derive(FromQueryResult)]
struct CakeResult {
    id: i32,
    name: String,
}

let cakes: Vec<CakeResult> = CakeResult::find_by_statement(
    Statement::from_sql_and_values(
        DbBackend::Postgres,
        r#"SELECT "id", "name" FROM "cake" WHERE "id" = $1"#,
        [1.into()],
    )
).all(db).await?;
```

### Execute Raw Statement

```rust
let result = db.execute(Statement::from_string(
    DbBackend::Postgres,
    "TRUNCATE TABLE cake".to_owned(),
)).await?;
```

### Query to JSON

```rust
let results: Vec<serde_json::Value> = JsonValue::find_by_statement(
    Statement::from_sql_and_values(
        DbBackend::Postgres,
        r#"SELECT "id", "name" FROM "cake""#,
        [],
    )
).all(db).await?;
```

### Debug: Inspect Generated SQL

```rust
let query = Cake::find()
    .filter(cake::Column::Name.contains("chocolate"))
    .build(DbBackend::Postgres)
    .to_string();
println!("{}", query);
// SELECT "cake"."id", "cake"."name" FROM "cake" WHERE "cake"."name" LIKE '%chocolate%'
```
