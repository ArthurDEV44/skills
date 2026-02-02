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
}).await;
```

---

## Explicit Begin/Commit

For complex lifetime scenarios:

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

**Auto-rollback**: Dropping `txn` without `commit()` automatically rolls back.

---

## Nested Transactions

Uses database SAVEPOINTs. Nested transactions can independently commit or rollback.

### Closure Nesting

```rust
ctx.db.transaction::<_, _, DbErr>(|txn| {
    Box::pin(async move {
        let _ = bakery::ActiveModel {..}.save(txn).await?;
        assert_eq!(Bakery::find().all(txn).await?.len(), 1);

        // Nested transaction (committed)
        txn.transaction::<_, _, DbErr>(|txn| {
            Box::pin(async move {
                let _ = bakery::ActiveModel {..}.save(txn).await?;

                // Deeply nested (rollbacked)
                assert!(txn.transaction::<_, _, DbErr>(|txn| {
                    Box::pin(async move {
                        let _ = bakery::ActiveModel {..}.save(txn).await?;
                        Err(DbErr::Query(RuntimeErr::Internal(
                            "Force Rollback!".to_owned(),
                        )))
                    })
                }).await.is_err());

                Ok(())
            })
        }).await;
        Ok(())
    })
}).await;
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
        // dropped without commit -> rollback
    }

    nested.commit().await?; // bakery_b persists
}

txn.commit().await?; // bakery_a + bakery_b persist
```

---

## Isolation Levels & Access Modes

Use `transaction_with_config` or `begin_with_config` (since v0.10.5, MySQL/PostgreSQL):

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
| `ReadOnly` | Prevents data modifications |
| `ReadWrite` | Allows modifications (default) |

**Note**: MySQL executes `SET TRANSACTION` before `BEGIN`; PostgreSQL after `BEGIN`.

---

## Error Handling

### Core Error Type: `DbErr`

All runtime errors are represented as `DbErr`.

### Standard SQL Errors via `sql_err()`

Converts to cross-database `SqlErr`:

```rust
// Unique constraint violation
assert!(matches!(
    cake.into_active_model().insert(db).await
        .expect_err("Duplicate primary key")
        .sql_err(),
    Some(SqlErr::UniqueConstraintViolation(_))
));

// Foreign key constraint violation
assert!(matches!(
    fk_cake.insert(db).await
        .expect_err("Invalid foreign key")
        .sql_err(),
    Some(SqlErr::ForeignKeyConstraintViolation(_))
));
```

### Database-Specific Error Codes

```rust
let error: DbErr = cake.into_active_model().insert(db).await
    .expect_err("duplicate key");

match error {
    DbErr::Exec(RuntimeErr::SqlxError(error)) => match error {
        sqlx::Error::Database(e) => {
            // MySQL: "23000" (ER_DUP_KEY)
            // PostgreSQL: "23505" (unique_violation)
            assert_eq!(e.code().unwrap(), "23000");
        }
        _ => panic!("Unexpected sqlx error"),
    },
    _ => panic!("Unexpected DbErr"),
}
```
