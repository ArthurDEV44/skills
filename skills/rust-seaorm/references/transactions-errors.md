# SeaORM Transactions & Error Handling Reference

## Table of Contents
- [Closure-Based Transactions](#closure-based-transactions)
- [Explicit Begin/Commit](#explicit-begincommit)
- [Nested Transactions](#nested-transactions)
- [Isolation Levels & Access Modes](#isolation-levels--access-modes)
- [Error Handling](#error-handling)

---

## Closure-Based Transactions

Auto-commits on `Ok`, auto-rolls back on `Err`. Use `Pin<Box<_>>` wrapper since async closures are not stabilized:

```rust
use sea_orm::TransactionTrait;

db.transaction::<_, (), DbErr>(|txn| {
    Box::pin(async move {
        bakery::ActiveModel {
            name: Set("SeaSide Bakery".to_owned()),
            profit_margin: Set(10.4),
            ..Default::default()
        }
        .save(txn)
        .await?;

        bakery::ActiveModel {
            name: Set("Top Bakery".to_owned()),
            profit_margin: Set(15.0),
            ..Default::default()
        }
        .save(txn)
        .await?;

        Ok(())
    })
}).await?;
```

### Returning Values from Transactions

```rust
let (bakery, chef) = db.transaction::<_, (bakery::Model, chef::Model), DbErr>(|txn| {
    Box::pin(async move {
        let bakery = bakery::ActiveModel {
            name: Set("SeaSide Bakery".to_owned()),
            ..Default::default()
        }.insert(txn).await?;

        let chef = chef::ActiveModel {
            name: Set("Baker Bob".to_owned()),
            bakery_id: Set(bakery.id),
            ..Default::default()
        }.insert(txn).await?;

        Ok((bakery, chef))
    })
}).await?;
```

---

## Explicit Begin/Commit

For complex lifetime scenarios or when closure-based approach is awkward:

```rust
let txn = db.begin().await?;

bakery::ActiveModel {
    name: Set("SeaSide Bakery".to_owned()),
    profit_margin: Set(10.4),
    ..Default::default()
}.save(&txn).await?;

bakery::ActiveModel {
    name: Set("Top Bakery".to_owned()),
    profit_margin: Set(15.0),
    ..Default::default()
}.save(&txn).await?;

txn.commit().await?;
```

**Auto-rollback**: Dropping `txn` without calling `commit()` automatically rolls back.

```rust
let txn = db.begin().await?;
some_operation(&txn).await?;
// If error occurs here, txn is dropped -> rollback
txn.commit().await?;
```

---

## Nested Transactions

Uses database SAVEPOINTs. Nested transactions can independently commit or rollback without affecting the parent.

### Closure Nesting

```rust
db.transaction::<_, _, DbErr>(|txn| {
    Box::pin(async move {
        let _ = bakery_a().save(txn).await?;

        // Nested transaction (committed)
        txn.transaction::<_, _, DbErr>(|txn| {
            Box::pin(async move {
                let _ = bakery_b().save(txn).await?;

                // Deeply nested (rollbacked)
                assert!(txn.transaction::<_, _, DbErr>(|txn| {
                    Box::pin(async move {
                        let _ = bakery_c().save(txn).await?;
                        Err(DbErr::Query(RuntimeErr::Internal(
                            "Force Rollback!".to_owned(),
                        )))
                    })
                }).await.is_err());

                // bakery_c is rolled back, bakery_b persists
                Ok(())
            })
        }).await?;

        Ok(())
    })
}).await?;
// Result: bakery_a + bakery_b committed, bakery_c rolled back
```

### Begin/Commit Nesting

```rust
let txn = db.begin().await?;
bakery_a().save(&txn).await?;

{
    let nested = txn.begin().await?;
    bakery_b().save(&nested).await?;

    {
        let deep = nested.begin().await?;
        bakery_c().save(&deep).await?;
        // dropped without commit -> bakery_c rolled back
    }

    nested.commit().await?; // bakery_b persists
}

txn.commit().await?; // bakery_a + bakery_b persist
```

---

## Isolation Levels & Access Modes

Use `transaction_with_config` or `begin_with_config` (MySQL/PostgreSQL):

```rust
use sea_orm::{IsolationLevel, AccessMode};

// Closure-based with config
db.transaction_with_config::<_, (), DbErr>(
    |txn| {
        Box::pin(async move {
            // operations...
            Ok(())
        })
    },
    Some(IsolationLevel::RepeatableRead),
    Some(AccessMode::ReadOnly),
).await?;

// Explicit with config
let txn = db.begin_with_config(
    Some(IsolationLevel::Serializable),
    Some(AccessMode::ReadWrite),
).await?;
```

### Isolation Levels

| Level | Behavior |
|-------|----------|
| `RepeatableRead` | Reads snapshot from first read in txn |
| `ReadCommitted` | Each read gets fresh snapshot |
| `ReadUncommitted` | Non-locking SELECT, may read uncommitted rows |
| `Serializable` | Only sees rows committed before first statement |

### Access Modes

| Mode | Behavior |
|------|----------|
| `ReadOnly` | Prevents data modifications in txn |
| `ReadWrite` | Allows modifications (default) |

**Note**: MySQL executes `SET TRANSACTION` before `BEGIN`; PostgreSQL after `BEGIN`.

---

## Error Handling

### Core Error Type: `DbErr`

All runtime errors are represented as `DbErr`. Common variants:

| Variant | Description |
|---------|-------------|
| `DbErr::ConnectionAcquire` | Failed to acquire connection from pool |
| `DbErr::Conn(RuntimeErr)` | Connection error |
| `DbErr::Exec(RuntimeErr)` | Execution error |
| `DbErr::Query(RuntimeErr)` | Query error |
| `DbErr::RecordNotFound(String)` | Record not found |
| `DbErr::AttrNotSet(String)` | Attribute not set on ActiveModel |
| `DbErr::Custom(String)` | Custom error message |
| `DbErr::Type(String)` | Type conversion error |
| `DbErr::Json(String)` | JSON operation error |
| `DbErr::Migration(String)` | Migration error |

### Standard SQL Errors via `sql_err()`

Converts to cross-database `SqlErr` for portable error handling:

```rust
// Unique constraint violation
match cake.into_active_model().insert(db).await {
    Err(err) => match err.sql_err() {
        Some(SqlErr::UniqueConstraintViolation(msg)) => {
            println!("Duplicate: {}", msg);
        }
        Some(SqlErr::ForeignKeyConstraintViolation(msg)) => {
            println!("FK error: {}", msg);
        }
        _ => return Err(err),
    },
    Ok(model) => { /* success */ }
}
```

### Database-Specific Error Codes

When you need the raw database error code:

```rust
let error: DbErr = cake.into_active_model().insert(db).await
    .expect_err("duplicate key");

match error {
    DbErr::Exec(RuntimeErr::SqlxError(error)) => match error {
        sqlx::Error::Database(e) => {
            // MySQL: "23000" (ER_DUP_KEY)
            // PostgreSQL: "23505" (unique_violation)
            let code = e.code().unwrap();
            let message = e.message();
        }
        _ => panic!("Unexpected sqlx error"),
    },
    _ => panic!("Unexpected DbErr"),
}
```

### RecordNotFound Handling

```rust
let cake = Cake::find_by_id(999).one(db).await?;
match cake {
    Some(model) => { /* found */ }
    None => { /* not found - no error, just None */ }
}

// When using one() vs one_or_err():
// .one(db) -> Result<Option<Model>, DbErr>
// For delete_by_id, non-existent returns error
```

### Custom Error Integration

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    Db(#[from] sea_orm::DbErr),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Validation: {0}")]
    Validation(String),
}

// Use in service layer
async fn get_cake(db: &DatabaseConnection, id: i32) -> Result<cake::Model, AppError> {
    Cake::find_by_id(id)
        .one(db)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Cake {} not found", id)))
}
```
