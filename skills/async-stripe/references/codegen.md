# Code Generation Reference

## Pipeline Overview

```
spec3.sdk.json → Parse (openapiv3) → Component extraction → Crate inference
  → Overrides → Deduplication → Code generation → rustfmt (2 passes) → rsync to generated/
```

Entry point: `openapi/src/main.rs`. CLI flags: `--fetch [latest|current|vNNN]`, `--graph`, `--dry-run`.

Orchestration: `CodeGen::write_files()` in `openapi/src/codegen.rs`:
1. `write_crate_base()` -- Cargo.toml + mod.rs per crate
2. `write_components()` -- structs, enums, requests
3. `write_api_version_file()` -- API version constant
4. `write_generated_for_webhooks()` -- EventObject enum
5. `write_crate_table()` -- crate_info.md
6. `write_object_info_for_testing()` -- test fixtures

Output goes to `out/` first, then `rsync --delete-during` copies to `generated/`, `async-stripe-webhook/src/generated/`, and `tests/tests/it/generated/`.

## Key Source Files

| File | Purpose |
|------|---------|
| `openapi/gen_crates.toml` | Crate assignment config (paths, packages per crate) |
| `openapi/version.json` | Pinned OpenAPI spec version |
| `openapi/src/codegen.rs` | Main orchestrator |
| `openapi/src/components.rs` | Schema resolution, cycle detection |
| `openapi/src/crate_inference.rs` | Type-to-crate assignment algorithm |
| `openapi/src/requests.rs` | Request struct generation |
| `openapi/src/rust_object.rs` | Rust object model (Struct, Enum, FieldlessEnum) |
| `openapi/src/rust_type.rs` | Rust type model (Object, Simple, Path, Container) |
| `openapi/src/spec_inference.rs` | OpenAPI-to-Rust type inference |
| `openapi/src/deduplication.rs` | Deduplicate repeated inline types |
| `openapi/src/overrides.rs` | Extract/promote specific types (e.g., ApiVersion) |
| `openapi/src/webhook.rs` | Webhook event type generation |
| `openapi/src/visitor.rs` | `Visit`/`VisitMut` traits for type tree traversal |
| `openapi/src/templates/` | All code templates (see below) |

## Crate Configuration (gen_crates.toml)

Each crate entry has:
- `paths`: Component paths from the OpenAPI schema (e.g., `"charge"`, `"customer"`)
- `packages`: Stripe `x-stripeResource.in_package` values (e.g., `"billing"`)
- `description`: Crate doc comment

**Critical invariant**: `shared` crate has NO dependencies on other generated crates. The crate dependency graph must be acyclic.

## Crate Inference Algorithm

When `gen_crates.toml` doesn't explicitly assign a component:

1. **Naming**: If component path starts with a known component's prefix (e.g., `charge_fraud_details` matches `charge`), assign to same crate. Also checks if all requests nest under another component's URL.
2. **Reverse deps**: If all dependants are in the same crate, assign there.
3. **Forward deps**: If all dependencies are in the same crate, assign there.
4. **Shared promotion**: `assign_paths_required_to_share_types()` moves type defs to `async-stripe-shared` when a type is depended upon by multiple crates.

When `types_split_from_requests()` is true: type defs go to `async-stripe-shared/src/{component}.rs`, requests go to `async-stripe-{crate}/src/{component}/requests.rs`, and domain crate re-exports the shared types.

## Type Model

### RustType (4 variants)
- `Object(RustObject, ObjectMetadata)` -- inline struct/enum
- `Simple(SimpleType)` -- Bool, Float, Str, String, Int(IntType), Ext(ExtType)
- `Path { path: PathToType, is_ref }` -- reference to type defined elsewhere
- `Container(Container)` -- List, SearchList, Vec, Slice, Expandable, Option, Box, Map

### PathToType
- `Component(ComponentPath)` -- top-level component
- `ObjectId(ComponentPath)` -- ID type for a component
- `Shared(RustIdent)` -- type in shared/extra types
- `Deduplicated { path, ident }` -- deduplicated type within a component

### RustObject (3 variants)
- `Struct(Struct)` -- fields, optional `object_field` sentinel, visibility
- `Enum(Vec<EnumVariant>)` -- variants with data
- `FieldlessEnum(Vec<FieldlessVariant>)` -- C-like string-backed enum

### ObjectUsage
- `kind: ObjectKind` -- `RequestParam`, `RequestReturned`, or `Type`
- Drives serialization: request params always serialize (never deser), return types always deser via miniserde (serde deser feature-gated)

## Spec Inference Rules

Key rules in `openapi/src/spec_inference.rs`:

| Condition | Inferred Type |
|-----------|---------------|
| Field name is/ends with `_currency` | `Currency` |
| Description starts with "Unique identifier" | ID type |
| Integer with `unix-time` format, or name contains `date` | `Timestamp` |
| Name contains `days` | `u32` |
| Name contains `count`/`size`/`quantity` | `u64` |
| Boolean with "Always true for" in description | `AlwaysTrue` |
| Object with `"object"` field enum `"list"` | `List<T>` |
| Object with `"object"` field enum `"search_result"` | `SearchList<T>` |
| `anyOf`/`oneOf` with `x-expansionResources` | `Expandable<T>` |
| First option title `range_query_specs` | `RangeQueryTs` |
| Empty properties object | `serde_json::Value` |

## Templates

All in `openapi/src/templates/`, methods on `ObjectWriter`.

### Struct Template (`structs.rs`)
1. Derives: `Clone, Debug` (optionally `Copy`, `Eq, PartialEq`)
2. Feature-gated serde: `#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]`
3. Fields with `serde(rename)` and `serde(skip_serializing_if = "Option::is_none")`
4. Miniserde deser impl (if type is deserialized)
5. Custom `Serialize` for structs with `object_field` (adds constant `"object"` field)

### Fieldless Enum Template (`enums.rs`)
1. `#[non_exhaustive]` with `Unknown(String)` variant for forward compatibility
2. `as_str()` method mapping variants to wire names
3. Infallible `FromStr` (falls back to `Unknown`)
4. `Display`/`Debug` delegate to `as_str()`
5. `serde::Serialize` via `as_str()`, `miniserde::Deserialize` via `FromStr`
6. When Stripe has a literal `"unknown"` variant, codegen uses `_Unknown(String)` to avoid conflict

### Request Template (`requests.rs`)
1. Private `{Name}Builder` struct with `serde::Serialize` (always, not gated)
2. Public `{Name}` struct wrapping `inner: {Name}Builder` (+ path params)
3. `new(required_params...)` constructor
4. Builder methods: `fn field(mut self, val: impl Into<T>) -> Self`
5. `send()` / `send_blocking()` convenience methods
6. `StripeRequest` impl: `build()` returns `RequestBuilder` -- GET uses `.query()`, POST/DELETE uses `.form()`
7. `paginate()` if return type is `List<T>` or `SearchList<T>`

### Cargo.toml Template (`cargo_toml.rs`)
- Workspace-inherited metadata
- `[lib]` with `path = "src/mod.rs"`, `name = "stripe_{name}"`
- Dependencies: `async-stripe-types`, `async-stripe-shared`, inter-crate deps
- `serialize`/`deserialize` features propagate transitively
- Per-component feature gates + `full` meta-feature

### Miniserde Template (`miniserde.rs`)
Generates `const _: () = { ... }` blocks containing:
1. `{Name}Builder` with `Option<T>` fields
2. `Deserialize` impl for the type
3. `Visitor` impl creating a Builder
4. `MapBuilder` impl with `key()` dispatch, `deser_default()`, `take_out()`
5. `Map` impl, `ObjectDeser` impl, `FromValueOpt` impl

Copy optimization: fields with `Copy` types use direct access instead of `.take()`.

### mod.rs Entry Pattern
```rust
#![recursion_limit = "256"]
#![deny(clippy::large_stack_frames)]
#![allow(clippy::large_enum_variant)]
extern crate self as stripe_{name};
miniserde::make_place!(Place);
```
The `extern crate self as stripe_{name}` enables absolute paths across crate boundaries.

## Deduplication

Problem: OpenAPI spec repeats identical type definitions across fields. Algorithm in `openapi/src/deduplication.rs`:

1. **Collect**: Walk objects via `Visit`, map `RustObject -> Vec<(ObjectMetadata, ObjectUsage)>`
2. **Name**: For objects appearing 2+ times, infer name from `title`, doc comment "The {X} of", or field names
3. **Replace**: Swap inline `RustType::Object` with `RustType::Path(PathToType::Deduplicated {...})`
4. **Iterate**: Multiple passes since dedup enables further dedup

Deduplicated types stored in `StripeObject::deduplicated_objects`.

## Overrides

In `openapi/src/overrides.rs`. Currently one override: extracts `api_version` enum from POST `/webhook_endpoints` request and promotes it to shared `ApiVersion` type. Uses `VisitMut` to replace all matching inline objects with `PathToType::Shared` references.

## Webhook Generation

In `openapi/src/webhook.rs`. Detects events via `x-stripeEvent` extension. Generates:
- `EventObject` enum with feature-gated variants (one per Stripe event type)
- `Box<T>` wrapping for each variant
- `from_raw_data()` (miniserde) and `from_json_value()` (serde, feature-gated) dispatchers
- `Unknown(miniserde::json::Value)` fallback
- Polymorphic per-event enums using `serde(tag = "object")` / `ObjectBuilderInner`

## Naming Conventions

- `RustIdent::create("payment_intent")` -> `PaymentIntent` (UpperCamelCase)
- Object renames: `"invoiceitem"` -> `"invoice_item"`, `"item"` -> `"checkout_session_item"`, `"line_item"` -> `"invoice_line_item"`, `"fee_refund"` -> `"application_fee_refund"`
- Reserved keywords (`type`, `as`, `use`, `struct`, `enum`, `const`, `async`, `await`, `in`) get underscore suffix (e.g., `type_`)

## Dependency Graph

Uses `petgraph::DiGraphMap` in `openapi/src/graph.rs`:
- Component graph: which components depend on which
- Crate graph: aggregated to crate level
- Used for crate inference, Cargo.toml deps, cycle detection, unused component filtering
