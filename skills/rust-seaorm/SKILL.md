---
name: rust-seaorm
description: "SeaORM async ORM for Rust (1.x stable + 2.0 RC): entity definition, CRUD operations, relations, migrations, transactions, and database patterns. Use when writing, reviewing, or refactoring Rust code using SeaORM: (1) Defining entities with DeriveEntityModel, column types, and ActiveEnum, (2) Writing relations (HasOne, HasMany, BelongsTo, M-N via junction, Linked), (3) CRUD operations (find, insert, update, delete, save, bulk ops), (4) Building queries (custom select, partial models, conditions, aggregates, joins, subqueries, raw SQL), (5) Managing transactions (closure-based, explicit begin/commit, nested savepoints, isolation levels), (6) Setting up database connections and connection pools, (7) Running migrations with sea-orm-migration and sea-orm-cli, (8) Pagination (offset and cursor-based), streaming results, (9) Handling errors (DbErr, SqlErr), (10) Mock database testing, (11) ActiveModelBehavior lifecycle hooks, (12) SeaORM 2.0 nested ActiveModel and entity-first schema sync."
---

# SeaORM

Async/dynamic ORM for Rust built on SeaQuery. Database-agnostic at compile time (MySQL, PostgreSQL, SQLite selected at runtime).

## Database Connection

```rust
use sea_orm::{Database, DatabaseConnection, ConnectOptions};

// Simple connection
let db: DatabaseConnection = Database::connect("postgres://user:pass@localhost/db").await?;

// With options
let mut opt = ConnectOptions::new("postgres://user:pass@localhost/db");
opt.max_connections(100)
   .min_connections(5)
   .connect_timeout(Duration::from_secs(8))
   .acquire_timeout(Duration::from_secs(8))
   .idle_timeout(Duration::from_secs(8))
   .max_lifetime(Duration::from_secs(8))
   .sqlx_logging(true)
   .sqlx_logging_level(log::LevelFilter::Info);
let db = Database::connect(opt).await?;
```

## Entity Definition

### Traditional Format (Stable 1.x)

```rust
use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "cake")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub name: String,
    #[sea_orm(column_type = "Text", nullable)]
    pub description: Option<String>,
    #[sea_orm(column_name = "type")]
    pub r#type: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::fruit::Entity")]
    Fruit,
}

impl Related<super::fruit::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Fruit.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
```

### Dense Format (SeaORM 2.0)

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

### ActiveEnum

```rust
#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum)]
#[sea_orm(rs_type = "String", db_type = "Enum", enum_name = "tea")]
pub enum Tea {
    #[sea_orm(string_value = "EverydayTea")]
    EverydayTea,
    #[sea_orm(string_value = "BreakfastTea")]
    BreakfastTea,
}

// Integer-backed enum
#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum)]
#[sea_orm(rs_type = "i32", db_type = "Integer")]
pub enum Color {
    #[sea_orm(num_value = 0)]
    Black,
    #[sea_orm(num_value = 1)]
    White,
}
```

## Basic CRUD

```rust
use sea_orm::{entity::*, query::*, ActiveValue::{Set, NotSet, Unchanged}};

// Find
let cakes: Vec<cake::Model> = Cake::find().all(db).await?;
let cake: Option<cake::Model> = Cake::find_by_id(1).one(db).await?;
let cakes = Cake::find()
    .filter(cake::Column::Name.contains("chocolate"))
    .order_by_asc(cake::Column::Name)
    .all(db).await?;

// Insert
let model = fruit::ActiveModel {
    name: Set("Apple".to_owned()),
    ..Default::default()
};
let result: fruit::Model = model.insert(db).await?;

// Update
let mut pear: fruit::ActiveModel = pear.into();
pear.name = Set("Sweet pear".to_owned());
let pear: fruit::Model = pear.update(db).await?;

// Save (insert if PK NotSet, update if Set/Unchanged)
let banana = fruit::ActiveModel {
    id: NotSet,
    name: Set("Banana".to_owned()),
    ..Default::default()
};
let banana: fruit::ActiveModel = banana.save(db).await?;

// Delete
let res: DeleteResult = orange.delete(db).await?;
let res = Fruit::delete_by_id(1).exec(db).await?;

// Bulk insert
let res = Fruit::insert_many([apple, pear]).exec(db).await?;

// Bulk update
Fruit::update_many()
    .col_expr(fruit::Column::CakeId, Expr::value(1))
    .filter(fruit::Column::Name.contains("Apple"))
    .exec(db).await?;

// Bulk delete
Fruit::delete_many()
    .filter(fruit::Column::Name.contains("Orange"))
    .exec(db).await?;
```

## Relations & Eager Loading

```rust
// 1-1 eager load (find_also_related)
let cake_with_fruit: Vec<(cake::Model, Option<fruit::Model>)> =
    Cake::find().find_also_related(Fruit).all(db).await?;

// 1-N / M-N eager load (find_with_related)
let cake_with_fillings: Vec<(cake::Model, Vec<filling::Model>)> =
    Cake::find().find_with_related(Filling).all(db).await?;

// Lazy loading from model instance
let fruits: Vec<fruit::Model> = cake_model.find_related(Fruit).all(db).await?;

// Chained relations via Linked
let fillings: Vec<filling::Model> = cake_model.find_linked(CakeToFilling).all(db).await?;
```

### Defining Many-to-Many

```rust
// In cake entity - Related impl with via()
impl Related<super::filling::Entity> for Entity {
    fn to() -> RelationDef {
        super::cake_filling::Relation::Filling.def()
    }
    fn via() -> Option<RelationDef> {
        Some(super::cake_filling::Relation::Cake.def().rev())
    }
}
```

## Transactions

```rust
use sea_orm::TransactionTrait;

// Closure-based (auto commit/rollback)
db.transaction::<_, (), DbErr>(|txn| {
    Box::pin(async move {
        entity_a.save(txn).await?;
        entity_b.save(txn).await?;
        Ok(())
    })
}).await?;

// Explicit begin/commit (drop without commit = rollback)
let txn = db.begin().await?;
entity_a.save(&txn).await?;
entity_b.save(&txn).await?;
txn.commit().await?;
```

## Pagination

```rust
use sea_orm::PaginatorTrait;

// Offset-based pagination
let paginator = Cake::find()
    .order_by_asc(cake::Column::Id)
    .paginate(db, 50); // 50 per page
let num_pages = paginator.num_pages().await?;
let page_2: Vec<cake::Model> = paginator.fetch_page(1).await?; // 0-indexed

// Cursor-based pagination
use sea_orm::CursorTrait;
let mut cursor = Cake::find().cursor_by(cake::Column::Id);
cursor.after(10); // after id=10
let next_batch: Vec<cake::Model> = cursor.first(20).all(db).await?;
```

## Migrations (sea-orm-cli)

```bash
# Install CLI
cargo install sea-orm-cli

# Generate entities from existing database
sea-orm-cli generate entity -u postgres://user:pass@localhost/db -o src/entity

# Create migration
sea-orm-cli migrate generate create_users_table

# Run migrations
sea-orm-cli migrate up -u postgres://user:pass@localhost/db

# Rollback
sea-orm-cli migrate down -u postgres://user:pass@localhost/db
```

### Migration Definition

```rust
use sea_orm_migration::{prelude::*, async_trait};

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager.create_table(
            Table::create()
                .table(Users::Table)
                .if_not_exists()
                .col(ColumnDef::new(Users::Id).integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(Users::Name).string().not_null())
                .col(ColumnDef::new(Users::Email).string().not_null().unique_key())
                .to_owned(),
        ).await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager.drop_table(Table::drop().table(Users::Table).to_owned()).await
    }
}

#[derive(DeriveIden)]
enum Users {
    Table,
    Id,
    Name,
    Email,
}
```

## Raw SQL

```rust
use sea_orm::{FromQueryResult, Statement, DbBackend};

#[derive(FromQueryResult)]
struct CakeResult { id: i32, name: String }

let cakes: Vec<CakeResult> = CakeResult::find_by_statement(
    Statement::from_sql_and_values(
        DbBackend::Postgres,
        r#"SELECT "id", "name" FROM "cake" WHERE "id" = $1"#,
        [1.into()],
    )
).all(db).await?;

// Execute raw statement
db.execute(Statement::from_string(
    DbBackend::Postgres,
    "TRUNCATE TABLE cake".to_owned(),
)).await?;
```

## Reference Files

For detailed patterns with full code examples, consult:

- **Entity definition, relations, ActiveEnum, lifecycle hooks, core traits**: See [references/entities-relations.md](references/entities-relations.md)
- **CRUD, custom select, conditions, aggregates, joins, subqueries, pagination, streaming**: See [references/crud-queries.md](references/crud-queries.md)
- **Transactions, isolation levels, error handling (DbErr, SqlErr)**: See [references/transactions-errors.md](references/transactions-errors.md)
- **Database setup, CLI, migrations, mock testing**: See [references/setup-cli-testing.md](references/setup-cli-testing.md)
