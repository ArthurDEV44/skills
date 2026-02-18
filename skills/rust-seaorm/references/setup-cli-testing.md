# SeaORM Setup, CLI & Testing Reference

## Table of Contents
- [Database Connection](#database-connection)
- [Cargo Dependencies](#cargo-dependencies)
- [sea-orm-cli](#sea-orm-cli)
- [Migrations](#migrations)
- [Entity Generation](#entity-generation)
- [Mock Database Testing](#mock-database-testing)

---

## Database Connection

### Basic Connection

```rust
use sea_orm::{Database, DatabaseConnection};

let db: DatabaseConnection = Database::connect("postgres://user:pass@localhost/mydb").await?;
```

### Connection String Formats

| Database | Format |
|----------|--------|
| PostgreSQL | `postgres://user:pass@host:5432/db` |
| MySQL | `mysql://user:pass@host:3306/db` |
| SQLite | `sqlite:./path/to/db.sqlite?mode=rwc` |
| SQLite (memory) | `sqlite::memory:` |

### Connection Options

```rust
use sea_orm::ConnectOptions;
use std::time::Duration;

let mut opt = ConnectOptions::new("postgres://user:pass@localhost/db");
opt.max_connections(100)
   .min_connections(5)
   .connect_timeout(Duration::from_secs(8))
   .acquire_timeout(Duration::from_secs(8))
   .idle_timeout(Duration::from_secs(8))
   .max_lifetime(Duration::from_secs(8))
   .sqlx_logging(true)
   .sqlx_logging_level(log::LevelFilter::Info)
   .set_schema_search_path("my_schema");

let db = Database::connect(opt).await?;
```

### Connection Healthcheck

```rust
db.ping().await?;
```

### Closing Connection

```rust
db.close().await?;
```

---

## Cargo Dependencies

### Minimal Setup

```toml
[dependencies]
sea-orm = { version = "1.1", features = [
    "sqlx-postgres",        # or sqlx-mysql, sqlx-sqlite
    "runtime-tokio-rustls", # or runtime-actix-rustls, runtime-async-std-rustls
    "macros",               # DeriveEntityModel, DeriveActiveEnum, etc.
] }
```

### Common Feature Flags

| Feature | Purpose |
|---------|---------|
| `sqlx-postgres` | PostgreSQL backend |
| `sqlx-mysql` | MySQL/MariaDB backend |
| `sqlx-sqlite` | SQLite backend |
| `runtime-tokio-rustls` | Tokio runtime with rustls TLS |
| `runtime-tokio-native-tls` | Tokio runtime with native TLS |
| `runtime-actix-rustls` | Actix runtime with rustls |
| `macros` | Derive macros (DeriveEntityModel, etc.) |
| `with-json` | JSON column support + set_from_json |
| `with-chrono` | chrono DateTime support |
| `with-time` | time crate DateTime support |
| `with-rust_decimal` | Decimal support |
| `with-bigdecimal` | BigDecimal support |
| `with-uuid` | UUID support |
| `mock` | MockDatabase for testing |
| `debug-print` | Print SQL statements to stdout |

### Migration Crate Dependencies

```toml
[dependencies]
sea-orm-migration = { version = "1.1", features = [
    "sqlx-postgres",
    "runtime-tokio-rustls",
] }
```

---

## sea-orm-cli

### Installation

```bash
cargo install sea-orm-cli
```

### Entity Generation (Schema-First)

Generate Rust entity files from an existing database:

```bash
# Basic usage
sea-orm-cli generate entity \
    -u postgres://user:pass@localhost/mydb \
    -o src/entity

# With options
sea-orm-cli generate entity \
    -u postgres://user:pass@localhost/mydb \
    -o src/entity \
    --expanded-format \          # traditional format (not DeriveEntityModel)
    --with-serde both \          # add Serialize + Deserialize
    --with-copy-enums \          # derive Copy for enums
    --date-time-crate chrono \   # chrono instead of default time
    --tables users,posts \       # specific tables only
    --ignore-tables migrations   # exclude tables
```

### Migration Commands

```bash
# Initialize migration directory
sea-orm-cli migrate init

# Create new migration
sea-orm-cli migrate generate create_users_table

# Run pending migrations
sea-orm-cli migrate up -u postgres://user:pass@localhost/mydb

# Rollback last migration
sea-orm-cli migrate down -u postgres://user:pass@localhost/mydb

# Rollback N migrations
sea-orm-cli migrate down -n 3 -u postgres://user:pass@localhost/mydb

# Check migration status
sea-orm-cli migrate status -u postgres://user:pass@localhost/mydb

# Reset (rollback all, then run all)
sea-orm-cli migrate fresh -u postgres://user:pass@localhost/mydb

# Refresh (rollback all, then run all)
sea-orm-cli migrate refresh -u postgres://user:pass@localhost/mydb
```

---

## Migrations

### Project Structure

```
migration/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── m20220101_000001_create_table.rs
│   └── m20220201_000001_add_column.rs
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
                .col(ColumnDef::new(Users::Id)
                    .integer()
                    .not_null()
                    .auto_increment()
                    .primary_key())
                .col(ColumnDef::new(Users::Email)
                    .string()
                    .not_null()
                    .unique_key())
                .col(ColumnDef::new(Users::Name)
                    .string()
                    .not_null())
                .col(ColumnDef::new(Users::CreatedAt)
                    .timestamp_with_time_zone()
                    .not_null()
                    .default(Expr::current_timestamp()))
                .to_owned(),
        ).await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager.drop_table(
            Table::drop().table(Users::Table).to_owned()
        ).await
    }
}

#[derive(DeriveIden)]
enum Users {
    Table,
    Id,
    Email,
    Name,
    CreatedAt,
}
```

### Common Migration Operations

```rust
// Add column
manager.alter_table(
    Table::alter()
        .table(Users::Table)
        .add_column(ColumnDef::new(Users::Bio).text().null())
        .to_owned(),
).await?;

// Drop column
manager.alter_table(
    Table::alter()
        .table(Users::Table)
        .drop_column(Users::Bio)
        .to_owned(),
).await?;

// Create index
manager.create_index(
    Index::create()
        .name("idx_users_email")
        .table(Users::Table)
        .col(Users::Email)
        .unique()
        .to_owned(),
).await?;

// Add foreign key
manager.create_foreign_key(
    ForeignKey::create()
        .name("fk_post_user")
        .from(Posts::Table, Posts::UserId)
        .to(Users::Table, Users::Id)
        .on_delete(ForeignKeyAction::Cascade)
        .on_update(ForeignKeyAction::Cascade)
        .to_owned(),
).await?;

// Create enum type (PostgreSQL)
manager.create_type(
    Type::create()
        .as_enum(Alias::new("status"))
        .values([Alias::new("active"), Alias::new("inactive")])
        .to_owned(),
).await?;

// Execute raw SQL in migration
manager.get_connection()
    .execute_unprepared("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
    .await?;
```

### Registering Migrations

```rust
// src/lib.rs
pub use sea_orm_migration::prelude::*;

mod m20220101_000001_create_users;
mod m20220201_000001_create_posts;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20220101_000001_create_users::Migration),
            Box::new(m20220201_000001_create_posts::Migration),
        ]
    }
}
```

### Running Migrations Programmatically

```rust
use migration::{Migrator, MigratorTrait};

// Run all pending migrations
Migrator::up(db, None).await?;

// Run N migrations
Migrator::up(db, Some(2)).await?;

// Rollback last migration
Migrator::down(db, Some(1)).await?;

// Fresh: drop all tables and re-run all migrations
Migrator::fresh(db).await?;

// Check status
Migrator::status(db).await?;
```

---

## Entity Generation

### Generated Entity Structure

`sea-orm-cli generate entity` produces one file per table:

```
src/entity/
├── mod.rs          # re-exports all entities
├── prelude.rs      # convenient imports
├── cake.rs         # cake entity
├── fruit.rs        # fruit entity
└── cake_filling.rs # junction table entity
```

### Customizing Generated Code

After generation, you typically:
1. Add custom `Related` impls for M-N relations (not auto-generated)
2. Add `ActiveModelBehavior` hooks
3. Add `impl` blocks with custom business logic
4. Adjust types (e.g., use chrono instead of time)

### Entity Module Setup

```rust
// src/entity/mod.rs
pub mod prelude;
pub mod cake;
pub mod fruit;
pub mod cake_filling;

// src/entity/prelude.rs
pub use super::cake::Entity as Cake;
pub use super::fruit::Entity as Fruit;
pub use super::cake_filling::Entity as CakeFilling;
```

---

## Mock Database Testing

Enable the `mock` feature for testing without a real database.

### MockDatabase Setup

```rust
#[cfg(test)]
mod tests {
    use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult};

    #[tokio::test]
    async fn test_find_cake() {
        // Setup mock with expected query results
        let db = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([
                // First query result
                vec![cake::Model {
                    id: 1,
                    name: "Chocolate".to_owned(),
                }],
                // Second query result
                vec![cake::Model {
                    id: 2,
                    name: "Vanilla".to_owned(),
                }],
            ])
            .into_connection();

        // Execute operations
        let cake = Cake::find_by_id(1).one(&db).await.unwrap();
        assert_eq!(cake.unwrap().name, "Chocolate");

        // Verify executed queries
        let log = db.into_transaction_log();
        assert_eq!(log.len(), 1);
    }
}
```

### Mocking Exec Results (INSERT/UPDATE/DELETE)

```rust
let db = MockDatabase::new(DatabaseBackend::Postgres)
    .append_exec_results([
        MockExecResult {
            last_insert_id: 1,
            rows_affected: 1,
        },
    ])
    .into_connection();

let result = fruit::ActiveModel {
    name: Set("Apple".to_owned()),
    ..Default::default()
}.insert(&db).await.unwrap();

assert_eq!(result.id, 1);
```

### Mocking Multiple Operations

```rust
let db = MockDatabase::new(DatabaseBackend::Postgres)
    .append_query_results([
        // Result for first query
        vec![cake::Model { id: 1, name: "Cake 1".to_owned() }],
    ])
    .append_exec_results([
        // Result for insert
        MockExecResult { last_insert_id: 2, rows_affected: 1 },
    ])
    .append_query_results([
        // Result for second query (after insert)
        vec![
            cake::Model { id: 1, name: "Cake 1".to_owned() },
            cake::Model { id: 2, name: "Cake 2".to_owned() },
        ],
    ])
    .into_connection();
```

### Verifying Transaction Log

```rust
use sea_orm::Transaction;

let log = db.into_transaction_log();
assert_eq!(
    log[0],
    Transaction::from_sql_and_values(
        DatabaseBackend::Postgres,
        r#"SELECT "cake"."id", "cake"."name" FROM "cake" WHERE "cake"."id" = $1 LIMIT $2"#,
        [1.into(), 1u64.into()]
    )
);
```

### Testing Transactions

```rust
let db = MockDatabase::new(DatabaseBackend::Postgres)
    .append_exec_results([
        MockExecResult { last_insert_id: 1, rows_affected: 1 },
        MockExecResult { last_insert_id: 2, rows_affected: 1 },
    ])
    .into_connection();

db.transaction::<_, (), DbErr>(|txn| {
    Box::pin(async move {
        cake_a.save(txn).await?;
        cake_b.save(txn).await?;
        Ok(())
    })
}).await.unwrap();

// Transaction log includes BEGIN and COMMIT
let log = db.into_transaction_log();
```

### Testing with Real Database (Integration Tests)

```rust
#[cfg(test)]
mod integration_tests {
    use sea_orm::{Database, DatabaseConnection};

    async fn setup() -> DatabaseConnection {
        let db = Database::connect("postgres://test:test@localhost/test_db").await.unwrap();
        Migrator::fresh(&db).await.unwrap();
        db
    }

    #[tokio::test]
    async fn test_create_cake() {
        let db = setup().await;

        let cake = cake::ActiveModel {
            name: Set("Test Cake".to_owned()),
            ..Default::default()
        }.insert(&db).await.unwrap();

        assert!(cake.id > 0);
        assert_eq!(cake.name, "Test Cake");

        db.close().await.unwrap();
    }
}
```
