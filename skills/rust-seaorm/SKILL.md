---
name: rust-seaorm
description: "SeaORM 2.0 async ORM for Rust: entity definition, CRUD operations, advanced queries, relations, transactions, and database patterns. Use when writing, reviewing, or refactoring Rust code using SeaORM: (1) Defining entities with DeriveEntityModel and relations (HasOne, HasMany, BelongsTo, M-N), (2) Writing CRUD operations (find, insert, update, delete, save), (3) Building advanced queries (custom select, partial models, aggregates, joins, subqueries), (4) Using conditional expressions and filters (Condition::all, Condition::any, apply_if), (5) Managing transactions (closure-based, explicit begin/commit, nested), (6) Streaming query results, (7) Working with nested ActiveModel for atomic relational persistence, (8) Handling database errors (DbErr, SqlErr), (9) Custom join conditions and table aliases, (10) Entity-first schema sync workflow."
---

# SeaORM 2.0

Async/dynamic ORM for Rust built on SeaQuery. Database-agnostic at compile time (MySQL, PostgreSQL, SQLite selected at runtime).

## Quick Start

### Entity Definition (Dense Format)

```rust
#[sea_orm::model]
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "user")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub name: String,
    pub email: String,
    #[sea_orm(has_one)]
    pub profile: HasOne<profile::Entity>,
    #[sea_orm(has_many)]
    pub posts: HasMany<post::Entity>,
}
```

### Basic CRUD

```rust
// Find all
let cakes: Vec<cake::Model> = Cake::find().all(db).await?;

// Find by ID
let cake: Option<cake::Model> = Cake::find_by_id(1).one(db).await?;

// Find with filter
let cakes = Cake::find()
    .filter(cake::Column::Name.contains("chocolate"))
    .all(db).await?;

// Insert
let model = fruit::ActiveModel {
    name: Set("Apple".to_owned()),
    ..Default::default()
};
let result = model.insert(db).await?;

// Update
let mut pear: fruit::ActiveModel = pear.into();
pear.name = Set("Sweet pear".to_owned());
let pear = pear.update(db).await?;

// Save (upsert: insert if NotSet PK, update if Set)
let banana = fruit::ActiveModel {
    id: NotSet,
    name: Set("Banana".to_owned()),
    ..Default::default()
};
let banana = banana.save(db).await?;

// Delete
orange.delete(db).await?;

// Bulk operations
Fruit::insert_many([apple, pear]).exec_with_returning(db).await?;
Fruit::update_many()
    .col_expr(fruit::Column::CakeId, fruit::Column::CakeId.add(2))
    .filter(fruit::Column::Name.contains("Apple"))
    .exec(db).await?;
Fruit::delete_many()
    .filter(fruit::Column::Name.contains("Orange"))
    .exec(db).await?;
```

### Relations & Eager Loading

```rust
// 1-1 eager load
let cake_with_fruit: Vec<(cake::Model, Option<fruit::Model>)> =
    Cake::find().find_also_related(Fruit).all(db).await?;

// 1-N / M-N eager load
let cake_with_fillings: Vec<(cake::Model, Vec<filling::Model>)> =
    Cake::find().find_with_related(Filling).all(db).await?;

// Smart Entity Loader (auto N+1 prevention)
let user = user::Entity::load()
    .filter_by_id(42)
    .with(profile::Entity)
    .with((post::Entity, tag::Entity))
    .one(db).await?.unwrap();
```

### Transactions

```rust
use sea_orm::TransactionTrait;

// Closure-based (auto commit/rollback)
db.transaction::<_, (), DbErr>(|txn| {
    Box::pin(async move {
        entity_a.save(txn).await?;
        entity_b.save(txn).await?;
        Ok(())
    })
}).await;

// Explicit
let txn = db.begin().await?;
entity_a.save(&txn).await?;
txn.commit().await?; // drop without commit = rollback
```

### Nested ActiveModel (SeaORM 2.0)

```rust
let user = user::ActiveModel::builder()
    .set_name("Bob")
    .set_email("bob@sea-ql.org")
    .set_profile(profile::ActiveModel::builder().set_picture("image.jpg"))
    .add_post(
        post::ActiveModel::builder()
            .set_title("Nice weather")
            .add_tag(tag::ActiveModel::builder().set_tag("sunny")),
    )
    .save(db).await?;
```

## Reference Files

For detailed patterns with full code examples, read the appropriate reference file:

- **Advanced queries** (custom select, partial models, conditions, aggregates, joins, subqueries, streaming): See [references/advanced-queries.md](references/advanced-queries.md)
- **Transactions & error handling** (closure/explicit txn, nested txn, isolation levels, DbErr, SqlErr): See [references/transactions-errors.md](references/transactions-errors.md)
- **Entities & internals** (entity definition, nested ActiveModel, core traits, derive macros, architecture): See [references/entities-internals.md](references/entities-internals.md)
