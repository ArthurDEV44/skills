# SeaORM Entities & Relations Reference

## Table of Contents
- [Entity Definition](#entity-definition)
- [Column Types & Attributes](#column-types--attributes)
- [ActiveValue Tri-State](#activevalue-tri-state)
- [ActiveEnum](#activeenum)
- [Relations](#relations)
- [Linked Trait](#linked-trait)
- [ActiveModelBehavior Hooks](#activemodelbehavior-hooks)
- [Nested ActiveModel (2.0)](#nested-activemodel-20)
- [Core Traits](#core-traits)
- [Derive Macros](#derive-macros)
- [Architecture](#architecture)

---

## Entity Definition

### Traditional Expanded Format (Stable 1.x)

Each entity consists of: Entity struct, Model struct, Column enum, PrimaryKey enum, Relation enum, Related impl, and ActiveModelBehavior impl.

```rust
use sea_orm::entity::prelude::*;

// Entity unit struct
#[derive(Copy, Clone, Default, Debug, DeriveEntity)]
pub struct Entity;

impl EntityName for Entity {
    fn table_name(&self) -> &'static str {
        "cake"
    }
}

// Model (read-only query result)
#[derive(Clone, Debug, PartialEq, Eq, DeriveModel, DeriveActiveModel)]
pub struct Model {
    pub id: i32,
    pub name: String,
}

// Column enum
#[derive(Copy, Clone, Debug, EnumIter, DeriveColumn)]
pub enum Column {
    Id,
    Name,
}

impl ColumnTrait for Column {
    type EntityName = Entity;
    fn def(&self) -> ColumnDef {
        match self {
            Self::Id => ColumnType::Integer.def(),
            Self::Name => ColumnType::String(StringLen::None).def(),
        }
    }
}

// Primary key
#[derive(Copy, Clone, Debug, EnumIter, DerivePrimaryKey)]
pub enum PrimaryKey {
    Id,
}

impl PrimaryKeyTrait for PrimaryKey {
    type ValueType = i32;
    fn auto_increment() -> bool {
        true
    }
}

// Relations
#[derive(Copy, Clone, Debug, EnumIter)]
pub enum Relation {}

impl RelationTrait for Relation {
    fn def(&self) -> RelationDef {
        unimplemented!()
    }
}

impl ActiveModelBehavior for ActiveModel {}
```

### Compact DeriveEntityModel (Preferred)

Generates Entity, Column, PrimaryKey automatically from a single Model struct:

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

### With Schema Name

```rust
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "cake", schema_name = "public")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
}
```

### Explicit Column Naming

```rust
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "my_entity")]
pub struct Model {
    #[sea_orm(primary_key, enum_name = "IdentityColumn", column_name = "id")]
    pub id: i32,
    #[sea_orm(column_name = "type")]  // reserved word escape
    pub r#type: String,
}
```

### Composite Primary Key

```rust
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "cake_filling")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub cake_id: i32,
    #[sea_orm(primary_key, auto_increment = false)]
    pub filling_id: i32,
}
```

### Dense Format (SeaORM 2.0)

The `#[sea_orm::model]` attribute allows defining relations inline with the model:

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

---

## Column Types & Attributes

### Common Column Types

| Rust Type | SeaORM column_type | Database Type |
|-----------|-------------------|---------------|
| `String` | `String(StringLen::None)` | VARCHAR |
| `String` | `Text` | TEXT |
| `i32` | `Integer` | INT |
| `i64` | `BigInteger` | BIGINT |
| `f32` | `Float` | FLOAT |
| `f64` | `Double` | DOUBLE |
| `bool` | `Boolean` | BOOL |
| `Vec<u8>` | `Binary(BlobSize::Blob(None))` | BLOB |
| `Decimal` | `Decimal(Some((p, s)))` | DECIMAL(p,s) |
| `Uuid` | `Uuid` | UUID |
| `DateTime` | `DateTime` | TIMESTAMP |
| `DateTimeWithTimeZone` | `TimestampWithTimeZone` | TIMESTAMPTZ |
| `Date` | `Date` | DATE |
| `Time` | `Time` | TIME |
| `Json` / `serde_json::Value` | `Json` / `JsonBinary` | JSON / JSONB |
| `Option<T>` | any (nullable) | column + NULL |

### Column Attributes

| Attribute | Purpose |
|-----------|---------|
| `primary_key` | Mark as primary key |
| `auto_increment = false` | Disable auto-increment (default true for PK) |
| `column_name = "name"` | Override database column name |
| `column_type = "Text"` | Override inferred column type |
| `enum_name = "Name"` | Override Column enum variant name |
| `nullable` | Mark column as nullable |
| `unique` | Add unique constraint |
| `default_value = "expr"` | Default value |
| `select_as = "expr"` | Custom expression when selecting |
| `save_as = "expr"` | Custom expression when saving |

---

## ActiveValue Tri-State

```rust
pub enum ActiveValue<V> {
    Set(V),        // Explicitly set -> included in INSERT/UPDATE
    Unchanged(V),  // Existing DB value -> used in WHERE, excluded from SET
    NotSet,        // Undefined -> excluded entirely
}
```

### Behavior in Statements

| Operation | Set | Unchanged | NotSet |
|-----------|-----|-----------|--------|
| INSERT | Included in VALUES | Included in VALUES | Excluded (DB default) |
| UPDATE | Included in SET | Used in WHERE (PK) | Excluded from SET |

### Usage Patterns

```rust
use sea_orm::ActiveValue::{Set, NotSet, Unchanged};

// New record: let DB generate id
let new = fruit::ActiveModel {
    id: NotSet,
    name: Set("Apple".to_owned()),
    cake_id: Set(None),  // explicitly NULL
    ..Default::default()
};

// Update: only change name, keep other fields
let update = fruit::ActiveModel {
    id: Unchanged(1),    // WHERE id = 1
    name: Set("Pear".to_owned()),
    cake_id: NotSet,     // don't touch
    ..Default::default()
};

// Convert Model to ActiveModel for editing
let mut active: fruit::ActiveModel = model.into();
active.name = Set("Updated".to_owned());

// Conditional set (preserves Unchanged state if value didn't change)
active.name.set_if_not_equals("Same Name".to_owned());
```

### Save Behavior

`save()` inspects the primary key:
- PK is `NotSet` -> performs INSERT
- PK is `Set` or `Unchanged` -> performs UPDATE

---

## ActiveEnum

Maps Rust enums to database values with type safety.

### String-Backed Enum

```rust
#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum)]
#[sea_orm(rs_type = "String", db_type = "Enum", enum_name = "tea")]
pub enum Tea {
    #[sea_orm(string_value = "EverydayTea")]
    EverydayTea,
    #[sea_orm(string_value = "BreakfastTea")]
    BreakfastTea,
}
```

### Integer-Backed Enum

```rust
#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum)]
#[sea_orm(rs_type = "i32", db_type = "Integer")]
pub enum Color {
    #[sea_orm(num_value = 0)]
    Black,
    #[sea_orm(num_value = 1)]
    White,
}
```

### Using in Entity

```rust
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "order")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub status: OrderStatus,  // uses ActiveEnum automatically
}
```

---

## Relations

### Has One (1-1)

```rust
#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_one = "super::profile::Entity")]
    Profile,
}

impl Related<super::profile::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Profile.def()
    }
}
```

### Has Many (1-N)

```rust
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
```

### Belongs To (Inverse)

```rust
// In fruit entity - belongs to cake
#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::cake::Entity",
        from = "Column::CakeId",
        to = "super::cake::Column::Id"
    )]
    Cake,
}

impl Related<super::cake::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Cake.def()
    }
}
```

### Many-to-Many (via Junction Table)

Requires a junction/pivot entity and `Related` impls with `via()`:

```rust
// Junction entity: cake_filling
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "cake_filling")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub cake_id: i32,
    #[sea_orm(primary_key, auto_increment = false)]
    pub filling_id: i32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::cake::Entity",
        from = "Column::CakeId",
        to = "super::cake::Column::Id"
    )]
    Cake,
    #[sea_orm(
        belongs_to = "super::filling::Entity",
        from = "Column::FillingId",
        to = "super::filling::Column::Id"
    )]
    Filling,
}

// In cake entity:
impl Related<super::filling::Entity> for Entity {
    fn to() -> RelationDef {
        super::cake_filling::Relation::Filling.def()
    }
    fn via() -> Option<RelationDef> {
        Some(super::cake_filling::Relation::Cake.def().rev())
    }
}

// In filling entity:
impl Related<super::cake::Entity> for Entity {
    fn to() -> RelationDef {
        super::cake_filling::Relation::Cake.def()
    }
    fn via() -> Option<RelationDef> {
        Some(super::cake_filling::Relation::Filling.def().rev())
    }
}
```

### Custom Join Conditions

```rust
#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::fruit::Entity")]
    Fruit,
    #[sea_orm(
        has_many = "super::fruit::Entity",
        on_condition = r#"super::fruit::Column::Name.like("%tropical%")"#
    )]
    TropicalFruit,
    #[sea_orm(
        has_many = "super::fruit::Entity",
        on_condition = r#"super::fruit::Column::Name.like("%tropical%")"#,
        condition_type = "any"  // OR instead of AND
    )]
    AnyTropicalFruit,
}
```

---

## Linked Trait

For complex join paths (chained relations, self-referencing, multiple relations between two entities):

```rust
pub struct CakeToFilling;

impl Linked for CakeToFilling {
    type FromEntity = cake::Entity;
    type ToEntity = filling::Entity;

    fn link(&self) -> Vec<RelationDef> {
        vec![
            cake_filling::Relation::Cake.def().rev(),  // Cake -> CakeFilling
            cake_filling::Relation::Filling.def(),       // CakeFilling -> Filling
        ]
    }
}

// Usage
let fillings: Vec<filling::Model> = cake_model.find_linked(CakeToFilling).all(db).await?;
```

### With Custom Conditions

```rust
impl Linked for CheeseCakeToFillingVendor {
    type FromEntity = cake::Entity;
    type ToEntity = vendor::Entity;

    fn link(&self) -> Vec<RelationDef> {
        vec![
            cake_filling::Relation::Cake
                .def()
                .on_condition(|left, _right| {
                    Expr::col((left, cake::Column::Name))
                        .like("%cheese%")
                        .into_condition()
                })
                .rev(),
            cake_filling::Relation::Filling.def(),
            filling::Relation::Vendor.def(),
        ]
    }
}
```

---

## ActiveModelBehavior Hooks

Lifecycle hooks for validation and side effects:

```rust
#[async_trait]
impl ActiveModelBehavior for ActiveModel {
    /// Called before insert
    async fn before_save(self, db: &DatabaseConnection, insert: bool) -> Result<Self, DbErr> {
        // Validate, transform, or enrich data
        if insert {
            // new record
        }
        Ok(self)
    }

    /// Called after insert/update
    async fn after_save(model: Model, db: &DatabaseConnection, insert: bool) -> Result<Model, DbErr> {
        // Trigger side effects, cache invalidation, etc.
        Ok(model)
    }

    /// Called before delete
    async fn before_delete(self, db: &DatabaseConnection) -> Result<Self, DbErr> {
        // Validate deletion is allowed
        Ok(self)
    }

    /// Called after delete
    async fn after_delete(self, db: &DatabaseConnection) -> Result<Self, DbErr> {
        // Cleanup related resources
        Ok(self)
    }
}
```

---

## Nested ActiveModel (2.0)

### Save Related Entities Atomically

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

### Append vs Replace Children

```rust
// Append mode (default): adds new children, leaves existing untouched
let mut bob = user::Entity::load()
    .filter_by_email("bob@sea-ql.org")
    .one(db).await?.unwrap();
bob.posts.push(post::ActiveModel::builder().set_title("Another post"));
bob.save(db).await?; // INSERT INTO post ..

// Replace mode: specifies exact child set, deletes others
bob.posts.replace_all([post_1]); // retain only post_1
bob.save(db).await?; // DELETE removed posts, keep post_1
```

### Change Detection

Only changed columns trigger SQL statements. Saving unchanged models is a no-op:

```rust
bob.posts[0].title = Set("Updated title".into());
bob.posts[0].comments[0].comment = Set("nice!".into());
bob.save(db).await?;
// Only generates UPDATE for changed fields
```

### Cascade Delete

```rust
let user = user::Entity::find_by_id(4).one(db).await?.unwrap();
user.cascade_delete(db).await?;
// Respects FK dependencies: deletes children before parents
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

### Smart Entity Loader (2.0 - N+1 Prevention)

Uses join for 1-1 and data loader for 1-N automatically:

```rust
let smart_user = user::Entity::load()
    .filter_by_id(42)
    .with(profile::Entity)
    .with((post::Entity, tag::Entity))
    .one(db).await?.unwrap();
// Only 3 queries despite nested relations
```

### Entity-First Workflow (2.0)

Auto-detects changes and creates tables/columns/keys:

```rust
db.get_schema_registry("my_crate::entity::*").sync(db).await;
```
