# SeaORM Entities & Internal Design Reference

## Table of Contents
- [Entity Definition (SeaORM 2.0)](#entity-definition-seaorm-20)
- [Nested ActiveModel](#nested-activemodel)
- [Core Traits](#core-traits)
- [Derive Macros](#derive-macros)
- [Architecture](#architecture)

---

## Entity Definition (SeaORM 2.0)

### Dense Format with `#[sea_orm::model]`

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

### Relationship Types

| Type | Attribute | Description |
|------|-----------|-------------|
| 1-1 | `has_one` | Single related entity |
| 1-N | `has_many` | Multiple related entities |
| M-N | `has_many, via = "junction"` | Many-to-many via junction table |
| Belongs To | `belongs_to, from = "col", to = "col"` | Inverse of has_one/has_many |

### ActiveValue Tri-State

```rust
pub enum ActiveValue<V> {
    Set(V),        // Explicitly set value -> included in INSERT/UPDATE
    Unchanged(V),  // Existing value -> excluded from UPDATE
    NotSet,        // Uninitialized -> excluded from INSERT/UPDATE
}
```

---

## Nested ActiveModel

### Save Related Entities Atomically (SeaORM 2.0)

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

### Has One / Belongs To (1-1)

```rust
// User side
#[sea_orm(table_name = "user")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    #[sea_orm(has_one)]
    pub profile: HasOne<super::profile::Entity>,
}

// Profile side
#[sea_orm(table_name = "profile")]
pub struct Model {
    #[sea_orm(unique)]
    pub user_id: i32,
    #[sea_orm(belongs_to, from = "user_id", to = "id")]
    pub user: HasOne<super::user::Entity>,
}
```

### Has Many (1-N) - Append vs Replace

```rust
// Append mode (default): non-destructive, adds new children
let mut bob = user::Entity::load()
    .filter_by_email("bob@sea-ql.org")
    .one(db).await?.unwrap();
bob.posts.push(post::ActiveModel::builder().set_title("Another weekend"));
bob.save(db).await?; // INSERT INTO post ..

// Replace mode: specifies exact child set, deletes others
bob.posts.replace_all([post_1]); // retain only post_1
```

### Many to Many (M-N)

```rust
#[sea_orm(table_name = "post")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub title: String,
    #[sea_orm(has_many, via = "post_tag")]
    pub tags: HasMany<super::tag::Entity>,
}

// Junction table
#[sea_orm(table_name = "post_tag")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub post_id: i32,
    #[sea_orm(primary_key, auto_increment = false)]
    pub tag_id: i32,
    #[sea_orm(belongs_to, from = "post_id", to = "id")]
    pub post: Option<super::post::Entity>,
    #[sea_orm(belongs_to, from = "tag_id", to = "id")]
    pub tag: Option<super::tag::Entity>,
}
```

### Cascade Delete

```rust
let user_4 = user::Entity::find_by_id(4).one(db).await?.unwrap();
user_4.cascade_delete(db).await?;
// Respects FK dependencies: deletes children before parents
```

### Weak Belongs To (nullable FK)

```rust
#[sea_orm(table_name = "attachment")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub post_id: Option<i32>,  // nullable FK
    #[sea_orm(belongs_to, from = "post_id", to = "id")]
    pub post: HasOne<super::post::Entity>,
}
// Deleting parent sets FK to NULL instead of cascade delete
```

### Change Detection

Only changed columns trigger SQL statements. Saving unchanged models is a no-op:

```rust
let mut post = post.into_active_model();
post.title = Set("The weather changed!");
post.save(db).await?;
// UPDATE post SET title = '..' WHERE id = 22

// Nested changes tracked automatically:
bob.posts[0].title = Set("Lorem ipsum".into());
bob.posts[0].comments[0].comment = Set("nice post!".into());
bob.posts[1].comments.push(comment::ActiveModel::builder().set_comment("interesting!"));
bob.save(db).await?;
```

---

## Core Traits

| Trait | Purpose |
|-------|---------|
| `EntityTrait` | Database table as unit struct. Provides CRUD methods (`find`, `insert`, `update`, `delete`) |
| `ColumnTrait` | Enum of all table columns with types/attributes. Implements `IdenStatic` + `Iterable` |
| `PrimaryKeyTrait` | Enum for primary keys. Each variant maps to a column variant |
| `ModelTrait` | Read-only in-memory query result. Implements `FromQueryResult` |
| `ActiveModelTrait` | Insert/update operations. Implements `ActiveModelBehavior` for lifecycle hooks |
| `RelationTrait` | Defines relationships with other entities. Implements `Iterable` |
| `ActiveEnum` | Maps Rust enum variants to database values |
| `Related` | Defines join paths for querying related entities (especially M-N) |
| `Linked` | Complex join paths: chained relations, self-referencing, multiple relations between two entities |
| `ConnectionTrait` | Generic database connection API |
| `TransactionTrait` | Spawn database transactions |
| `StreamTrait` | Stream query results |
| `PaginatorTrait` | Paginate result sets |
| `CursorTrait` | Cursor-based pagination |

---

## Derive Macros

| Macro | Purpose |
|-------|---------|
| `DeriveEntityModel` | Generates `Entity`, `Column`, and `PrimaryKey` from a `Model` |
| `DeriveEntity` | Implements `EntityTrait` + `Iden` + `IdenStatic` |
| `DeriveModel` | Implements `ModelTrait` + `FromQueryResult` |
| `DeriveActiveModel` | Implements `ActiveModelTrait` with setters/getters |
| `DeriveColumn` | Implements `ColumnTrait` + column identifiers + `EnumIter` |
| `DerivePrimaryKey` | Implements `PrimaryKeyToColumn` + `EnumIter` |
| `DeriveRelation` | Implements `RelationTrait` |
| `DeriveActiveEnum` | Maps Rust enums to database types |
| `DerivePartialModel` | Selective column queries via `PartialModelTrait` |
| `DeriveValueType` | Value conversion traits for custom types |
| `FromQueryResult` | Manual result mapping from raw queries |
| `DeriveIden` | Custom identifiers for table/column aliases |

### Common Attributes

| Attribute | Context | Purpose |
|-----------|---------|---------|
| `table_name = "name"` | Entity | Maps struct to table |
| `primary_key` | Column | Marks primary key |
| `auto_increment = false` | Column | Disables auto-increment |
| `unique` | Column | Unique constraint |
| `has_one` | Field | 1-1 relation |
| `has_many` | Field | 1-N relation |
| `has_many, via = "junction"` | Field | M-N relation |
| `belongs_to, from = "col", to = "col"` | Field | Inverse relation |
| `on_condition = "expr"` | Relation | Custom join condition |
| `condition_type = "any"` | Relation | OR join logic |
| `from_col = "col"` | PartialModel | Remap column |
| `from_expr = "expr"` | PartialModel | Custom expression |
| `skip` | FromQueryResult | Skip field in mapping |

---

## Architecture

### Design Philosophy

- **Layered abstractions**: "You'd dig one layer beneath if you want to"
- **Database-agnostic at compile time**: Entities don't know the DB until runtime
- **SeaQuery** is the standalone SQL builder underneath

### Four Processing Stages

1. **Declaration**: Entities/relations via `EntityTrait`, `ColumnTrait`, `RelationTrait`
2. **Query Building** (3 layers):
   - Layer 1: Entity's `find*`, `insert`, `update`, `delete` (basic CRUD)
   - Layer 2: `Select`, `Insert`, `Update`, `Delete` structs (advanced)
   - Layer 3: SeaQuery's `SelectStatement`, `InsertStatement` etc. (raw SQL tree)
3. **Execution**: `Selector`, `Inserter`, `Updater`, `Deleter` execute against DB
4. **Resolution**: Query results convert to Rust types per defined relations

### Smart Entity Loader (N+1 Prevention)

Uses join for 1-1 and data loader for 1-N automatically:

```rust
let smart_user = user::Entity::load()
    .filter_by_id(42)
    .with(profile::Entity)
    .with((post::Entity, tag::Entity))
    .one(db).await?.unwrap();
// Only 3 queries despite nested relations
```

### Entity-First Workflow (SeaORM 2.0)

Auto-detects changes and creates tables/columns/keys:

```rust
db.get_schema_registry("my_crate::entity::*").sync(db).await;
```
