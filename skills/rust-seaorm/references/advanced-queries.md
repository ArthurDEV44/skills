# SeaORM Advanced Queries Reference

## Table of Contents
- [Custom Select](#custom-select)
- [Conditional Expressions](#conditional-expressions)
- [Aggregate Functions](#aggregate-functions)
- [Custom Join Conditions](#custom-join-conditions)
- [Advanced Joins](#advanced-joins)
- [Subqueries](#subqueries)
- [Streaming](#streaming)

---

## Custom Select

### Select Partial Attributes

Clear defaults with `select_only()`, then specify columns:

```rust
cake::Entity::find()
    .select_only()
    .column(cake::Column::Name)
    .build(DbBackend::Postgres)
    .to_string()
// SELECT "cake"."name" FROM "cake"
```

Multiple columns:

```rust
cake::Entity::find()
    .select_only()
    .columns([cake::Column::Id, cake::Column::Name])
```

Dynamic column filtering:

```rust
cake::Entity::find()
    .select_only()
    .columns(cake::Column::iter().filter(|col| match col {
        cake::Column::Id => false,
        _ => true,
    }))
```

### Optional Fields (Since v0.12)

Partial selection of `Option<T>` fields returns `None` instead of errors:

```rust
let customer = Customer::find()
    .select_only()
    .column(customer::Column::Id)
    .column(customer::Column::Name)
    .one(db).await?;
// notes: Option<String> returns None if not selected
```

### Custom Expressions

```rust
use sea_query::{Alias, Expr, Func};

cake::Entity::find()
    .column_as(Expr::col(cake::Column::Id).max().sub(Expr::col(cake::Column::Id)), "id_diff")
    .column_as(Expr::cust("CURRENT_TIMESTAMP"), "current_time")

// Function-based:
cake::Entity::find()
    .expr_as(Func::upper(Expr::col((cake::Entity, cake::Column::Name))), "name_upper")
```

### FromQueryResult Custom Struct

```rust
#[derive(FromQueryResult)]
struct CakeAndFillingCount {
    id: i32,
    name: String,
    count: i32,
}

let cake_counts: Vec<CakeAndFillingCount> = cake::Entity::find()
    .column_as(filling::Column::Id.count(), "count")
    .join_rev(JoinType::InnerJoin, cake_filling::Relation::Cake.def())
    .join(JoinType::InnerJoin, cake_filling::Relation::Filling.def())
    .group_by(cake::Column::Id)
    .into_model::<CakeAndFillingCount>()
    .all(db).await?;
```

### Unstructured Tuples

```rust
let res: Vec<(String, i64)> = cake::Entity::find()
    .select_only()
    .column(cake::Column::Name)
    .column(cake::Column::Id.count())
    .group_by(cake::Column::Name)
    .into_tuple()
    .all(&db).await?;
```

### Partial Models (Since v0.12)

```rust
#[derive(DerivePartialModel)]
#[sea_orm(entity = "User")]
struct PartialUser {
    pub id: i32,
    pub avatar: String,
    pub unique_id: Uuid,
}

let query = User::find().into_partial_model::<PartialUser>();
```

Column remapping and custom expressions:

```rust
#[derive(DerivePartialModel)]
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
cake::Entity::find()
    .filter(
        Condition::all()
            .add(cake::Column::Id.gte(1))
            .add(cake::Column::Name.like("%Cheese%"))
    )
// WHERE cake.id >= 1 AND cake.name LIKE '%Cheese%'
```

### OR Conditions

```rust
cake::Entity::find()
    .filter(
        Condition::any()
            .add(cake::Column::Id.eq(4))
            .add(cake::Column::Id.eq(5))
    )
// WHERE cake.id = 4 OR cake.id = 5
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
```

### Fluent Conditional Queries (apply_if)

```rust
cake::Entity::find()
    .apply_if(Some(3), |mut query, v| {
        query.filter(cake::Column::Id.eq(v))
    })
    .apply_if(Some(100), QuerySelect::limit)
    .apply_if(None, QuerySelect::offset::<Option<u64>>)
```

---

## Aggregate Functions

Available on `ColumnTrait`: `max`, `min`, `sum`, `avg`, `count`.

### Sum

```rust
let sum_total: Decimal = order::Entity::find()
    .select_only()
    .column_as(order::Column::Total.sum(), "sum")
    .into_tuple()
    .one(db).await?.unwrap();
```

### Sum with Group By

```rust
let (customer, total_spent): (String, Decimal) = customer::Entity::find()
    .left_join(order::Entity)
    .select_only()
    .column(customer::Column::Name)
    .column_as(order::Column::Total.sum(), "sum")
    .group_by(customer::Column::Name)
    .into_tuple()
    .one(db).await?.unwrap();
```

### Multiple Aggregates

```rust
#[derive(Debug, FromQueryResult)]
struct SelectResult {
    name: String,
    num_orders: i64,
    total_spent: Decimal,
    min_spent: Decimal,
    max_spent: Decimal,
}

let select = customer::Entity::find()
    .left_join(order::Entity)
    .select_only()
    .column(customer::Column::Name)
    .column_as(order::Column::Total.count(), "num_orders")
    .column_as(order::Column::Total.sum(), "total_spent")
    .column_as(order::Column::Total.min(), "min_spent")
    .column_as(order::Column::Total.max(), "max_spent")
    .group_by(customer::Column::Name);
```

### Having Clause

```rust
cake::Entity::find()
    .select_only()
    .column_as(cake::Column::Id.count(), "count")
    .group_by(cake::Column::Name)
    .having(Expr::col("count").gt(6))
```

---

## Custom Join Conditions

### 1. Relation Method (on_condition attribute)

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
}
```

### 2. Linked Method

```rust
impl Linked for CheeseCakeToFillingVendor {
    type FromEntity = super::cake::Entity;
    type ToEntity = super::vendor::Entity;

    fn link(&self) -> Vec<RelationDef> {
        vec![
            super::cake_filling::Relation::Cake
                .def()
                .on_condition(|left, _right| {
                    Expr::col((left, super::cake::Column::Name))
                        .like("%cheese%")
                        .into_condition()
                })
                .rev(),
            super::cake_filling::Relation::Filling.def(),
            super::filling::Relation::Vendor.def(),
        ]
    }
}
```

### 3. On-the-Fly Method

```rust
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
```

### Table Aliases

```rust
cake::Entity::find()
    .join_as(JoinType::LeftJoin, relation.def(), "alias_name")
    .join(JoinType::LeftJoin, relation.def().from_alias("alias_name"))
```

### OR Conditions in Joins

Use `condition_type = "any"` attribute or `.condition_type(ConditionType::Any)` for OR logic.

---

## Advanced Joins

### Custom Result Struct

```rust
#[derive(Clone, Debug, PartialEq, Eq, FromQueryResult, Serialize)]
pub struct ComplexProduct {
    pub id: i64,
    pub name: String,
    pub r#type: String,
    pub price: Decimal,
    #[sea_orm(skip)]
    pub history: Vec<product_history::Model>,
}
```

### Helper Aliases

```rust
#[derive(DeriveIden, Clone, Copy)]
pub struct Base;
use complex_product::Entity as Prod;
pub type ProdCol = <Prod as EntityTrait>::Column;
type ProdRel = <Prod as EntityTrait>::Relation;
```

### Multi-Table Join with Aliases

```rust
pub fn query() -> Select<complex_product::Entity> {
    complex_product::Entity::find()
        .select_only()
        .tbl_col_as((Base, Id), "id")
        .tbl_col_as((Base, Name), "name")
        .column_as(product_type::Column::Name, "type")
        .column_as(ProdCol::Price, "price")
        .join_as(JoinType::InnerJoin, ProdRel::BaseProduct.def(), Base)
        .join(JoinType::InnerJoin,
              base_product::Relation::ProductType.def().from_alias(Base))
        .order_by_asc(Expr::col((Base, Id)))
}
```

### Diamond Topology Joins

```rust
.join_as(JoinType::LeftJoin, complex_product::Relation::BaseProduct.def(), Base)
.join_as(JoinType::LeftJoin, complex_product::Relation::Material.def(), Material)
.join(JoinType::InnerJoin, base_product::Relation::Attribute.def().from_alias(Base))
.join(JoinType::InnerJoin, material::Relation::Attribute.def().from_alias(Material))
```

### Data Association Pattern (N+1 manual)

```rust
pub fn history_of(ids: Vec<i64>) -> Select<product_history::Entity> {
    product_history::Entity::find()
        .filter(product_history::Column::ProductId.is_in(ids))
        .order_by_asc(product_history::Column::Id)
}

pub fn associate(
    mut parent: Vec<ComplexProduct>,
    children: Vec<product_history::Model>,
) -> Vec<ComplexProduct> {
    let parent_id_map: HashMap<i64, usize> =
        parent.iter().enumerate().map(|(i, s)| (s.id, i)).collect();
    for item in children {
        if let Some(index) = parent_id_map.get(&item.product_id) {
            parent[*index].history.push(item);
        }
    }
    parent
}
```

### Join Methods Summary

| Method | Usage |
|--------|-------|
| `join()` | Basic join using existing relation |
| `join_as()` | Join with table alias |
| `join_rev()` | Reverse join direction (belongs_to) |
| `from_alias()` | Chain join from aliased table |

---

## Subqueries

### in_subquery / not_in_subquery

```rust
use sea_orm::Condition;
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
// SELECT ... FROM cake WHERE cake.id IN (SELECT MAX(cake.id) FROM cake)
```

---

## Streaming

### Basic Stream

```rust
use futures::TryStreamExt;

let mut stream = Fruit::find().stream(db).await?;
while let Some(item) = stream.try_next().await? {
    let item: fruit::ActiveModel = item.into();
    // process item
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
    // 2 connections held simultaneously
}
// connections returned to pool on drop
```
