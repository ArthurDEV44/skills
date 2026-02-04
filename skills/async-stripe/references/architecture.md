# Hand-Written Architecture Reference

## Crate Dependency Graph

```
async-stripe-types          (hand-written)  -- core type primitives
      |
async-stripe-shared         (generated)     -- shared Stripe object types
      |
async-stripe-{domain}       (generated)     -- domain-specific requests/types
      |
async-stripe-client-core    (hand-written)  -- client abstraction, pagination
      |
async-stripe                (hand-written)  -- concrete HTTP clients
      |
async-stripe-webhook        (mixed)         -- verification + generated EventObject
```

## async-stripe-types

### def_id! Macro (`ids.rs`)

`def_id!(AccountId)` generates a newtype around `smol_str::SmolStr` with:
- `Clone`, `Debug`, `Eq`, `PartialEq`, `Hash`, `Ord`, `Display`, `FromStr`, `AsRef<str>`, `Deref<Target=str>`
- `serde::Serialize` (always), `serde::Deserialize` (under `"deserialize"` feature)
- `miniserde::Deserialize` (always), `FromValueOpt` via `impl_from_val_with_from_str!`
- `AsCursor`/`FromCursor` for pagination

Generated crates invoke: `stripe_types::def_id!(AccountId);`

### Expandable<T> (`expandable.rs`)

```rust
pub enum Expandable<T: Object> {
    Id(T::Id),
    Object(Box<T>),
}
```

Miniserde deser: JSON string -> `Expandable::Id(...)`, JSON object -> `Expandable::Object(...)`.

Also defines `MapBuilder` and `ObjectDeser` traits used by all generated miniserde impls.

### Object Trait, List<T>, SearchList<T> (`pagination.rs`)

```rust
pub trait Object {
    type Id: AsCursorOpt + FromCursor;
    fn id(&self) -> &Self::Id;
    fn into_id(self) -> Self::Id;
}
```

`List<T>`: `data: Vec<T>`, `has_more: bool`, `url: String`. Custom `Serialize` adds `"object": "list"`.

`SearchList<T>`: extends with `next_page: Option<String>`, `total_count: Option<u64>`.

Both have hand-written miniserde `Deserialize` impls.

### Params (`params.rs`)

- `AlwaysTrue` -- sentinel for discriminating deleted objects in `serde(untagged)` enums
- `Timestamp` -- `i64` alias
- `RangeQueryTs` / `RangeBoundsTs` -- gt/gte/lt/lte filters, serialized via `serde_qs` to `created[gte]=1501598702` format

### Helper Traits (`miniserde_helpers.rs`)

- `FromValueOpt` -- converts `miniserde::json::Value` to typed values. Implemented for primitives, `Option<T>`, `Vec<T>`, `HashMap`, `Box<T>`
- `ObjectBuilderInner` -- builds polymorphic objects from raw JSON maps (used by webhook event enums)
- `MaybeDeletedBuilderInner` -- checks `"deleted": true` for deleted-or-not discrimination
- `extract_object_discr` -- extracts `"object"` discriminator from JSON
- `impl_from_val_with_from_str!` -- implements `FromValueOpt` for `FromStr` types

### extern crate self Pattern

Both `async-stripe-types` and `async-stripe-shared` use:
```rust
extern crate self as stripe_types;  // or stripe_shared
```
This allows generated code to use absolute paths (`stripe_types::List<T>`) across crate boundaries.

## async-stripe-client-core

### Client Traits (`stripe_request.rs`)

```rust
pub trait StripeClient {
    type Err: StripeClientErr;
    fn execute(&self, req: CustomizedStripeRequest) -> impl Future<Output = Result<Bytes, Self::Err>>;
}

pub trait StripeBlockingClient {
    type Err: StripeClientErr;
    fn execute(&self, req: CustomizedStripeRequest) -> Result<Bytes, Self::Err>;
}

pub trait StripeRequest {
    type Output;
    fn build(&self) -> RequestBuilder;
    fn customize(&self) -> CustomizableStripeRequest<Self::Output>;
}
```

`RequestBuilder` holds: `query`, `body`, `path`, `method`. Uses `serde_qs` for serialization.

`CustomizableStripeRequest<T>`: wraps `RequestBuilder` + `ConfigOverride`, provides `send()`/`send_blocking()`. **All deserialization uses `miniserde::json::from_str`** in the primary path.

### Pagination (`pagination.rs`)

`PaginableList` trait unifies `List<T>` and `SearchList<T>`:
- `List` pagination: sets `starting_after` to last element's ID
- `SearchList` pagination: sets `page` to `next_page` cursor

`ListPaginator<T>` provides:
- `stream(client)` -- async `Stream<Item = Result<T, Err>>` via `futures_util::stream::unfold`. Lazy page fetching.
- `get_all(client)` -- blocking, collects all pages

Generated list requests provide `paginate()` method returning `ListPaginator::new_list(url, params)`.

### Request Strategy (`request_strategy.rs`)

- `Once` -- single request
- `Idempotent(key)` -- with idempotency key (validated: non-empty, max 255 chars)
- `Retry(n)` -- retry up to n times
- `ExponentialBackoff(n)` -- 2^n second backoff

Retry logic: `Stripe-Should-Retry` header overrides; 429 (rate limit) triggers retry; 400-499 (except 429) stops retries.

### Configuration (`config.rs`)

`SharedConfigBuilder`: secret key (must start with `sk_` or `rk_`), API version, app info, client ID, account ID, request strategy, base URL.

`ConfigOverride`: per-request overrides for `account_id` and `request_strategy`. `#[non_exhaustive]`.

## async-stripe (Main Client)

### Hyper Client (`hyper/`)

- `client.rs`: wraps `hyper_util::client::legacy::Client`. `construct_request()` builds HTTP with headers (Authorization, User-Agent, stripe-version, etc.). `send_inner()` implements retry loop with tracing.
- `client_builder.rs`: `ClientBuilder` wrapping `SharedConfigBuilder`. Default base URL `https://api.stripe.com/`, User-Agent `Stripe/v1 RustBindings/{version}`.
- `blocking.rs`: wraps async client + `tokio::runtime::Runtime` in `Arc`. Current-thread runtime with 30s timeout. Panics if used inside async runtime.
- `connector.rs`: compile-time TLS selection between `hyper-tls` and `hyper-rustls`.

### async-std Client (`async_std/`)

Parallel implementation using `surf::Client`. Same retry pattern with `async_std::task::sleep`.

### Error Types (`error.rs`)

```rust
pub enum StripeError {
    Stripe(Box<ApiErrors>, u16),   // API error + status code
    JSONDeserialize(String),        // Parse failure
    ClientError(String),            // Network error
    ConfigError(String),            // Invalid config
    Timeout,                        // Blocking request timeout
}
```

## async-stripe-webhook

### Hand-Written (`webhook.rs`)

`Webhook` struct provides:
- `construct_event(payload, sig, secret)` -- HMAC-SHA256 verification, 5-minute tolerance
- `insecure(payload)` -- bypasses verification (logs warning)
- `generate_test_header(payload, secret, timestamp)` -- for testing

Parsing: deserializes `stripe_shared::Event` via miniserde, then `EventObject::from_raw_data(type_str, data_value)` for typed dispatch.

`Event` struct (hand-written) wraps the generated event data. Custom `serde::Deserialize` uses "Shadow Struct" pattern (`EventProxy`) to first deserialize `data` as raw `serde_json::Value`, then dispatches via `EventObject::from_json_value()`.

### Generated (`generated/mod.rs`)

`EventObject` enum: one variant per Stripe event type, feature-gated (e.g., `#[cfg(feature = "async-stripe-billing")]`). Two dispatch methods:
- `from_raw_data(typ, data)` -- miniserde path
- `from_json_value(typ, data)` -- serde path (feature-gated)

Polymorphic events use `serde(tag = "object")` / `ObjectBuilderInner` with `"object"` field as discriminator.

## Testing

### Setup

Test crate: `tests/` (package `async-stripe-tests`). Connects to stripe-mock at `http://localhost:12111` with key `sk_test_123`.

```sh
docker run --rm -d -it -p 12111-12112:12111-12112 stripe/stripe-mock:v0.197.0
cargo test -p async-stripe-tests --no-default-features --features "default-tls"
```

### Test Categories

| Location | What |
|----------|------|
| `tests/tests/it/deser.rs` | Miniserde parsing of complex JSON (nested objects, expandable fields, polymorphic types) |
| `tests/tests/it/deserialization_fixture.rs` | Round-trip: miniserde deser -> serde ser -> JSON comparison for all fixtures |
| `tests/tests/it/blocking/` | CRUD against stripe-mock (blocking client) |
| `tests/tests/it/async_tests/` | Same with async hyper + async-std clients via `test_with_all_clients()` |
| `tests/tests/it/async_tests/pagination.rs` | Uses `wiremock` for mock pagination (List + SearchList) |
| `tests/tests/it/generated/` | Auto-generated fixture tests |
| In-crate `#[cfg(test)]` | Unit tests in ids.rs, params.rs, currency.rs, request_strategy.rs, webhook.rs |

### Fixture Tests

Loaded from `openapi/fixtures.json`. For each fixture: miniserde deserialize -> serde serialize -> recursive JSON comparison (handles missing optionals, float precision).

### Pagination Tests

`PaginationMock` creates a `wiremock` mock server simulating paginated responses. Tests cover: empty lists, full pagination, cursor sequences, partial consumption via `.take(n)`, both `List` and `SearchList`.
