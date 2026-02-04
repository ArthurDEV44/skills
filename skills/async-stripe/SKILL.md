---
name: async-stripe
description: "Development guide for the async-stripe Rust crate: strongly-typed async Stripe API bindings generated from OpenAPI spec. Use when working in the async-stripe repository for: (1) Modifying the code generation system in openapi/, (2) Adding or changing Stripe API types, requests, or enums, (3) Working with the hand-written foundation crates (async-stripe-types, async-stripe-client-core, async-stripe, async-stripe-webhook), (4) Fixing codegen bugs or adding codegen features, (5) Writing tests against stripe-mock, (6) Understanding the workspace architecture and crate relationships, (7) Debugging serialization issues (serde vs miniserde), (8) Adding new webhook event types, (9) Modifying feature flags or crate assignments."
---

# async-stripe Development

## Decision Tree

1. **Changing generated output?** (types, requests, enums in `generated/`)
   - Never edit `generated/` directly. Modify the codegen in `openapi/`. See [references/codegen.md](references/codegen.md).
2. **Changing hand-written foundation crates?** (`async-stripe-types`, `async-stripe-client-core`, `async-stripe`, `async-stripe-webhook`)
   - Edit those crates directly. See [references/architecture.md](references/architecture.md).
3. **Adding extension logic to a generated type?**
   - Create or edit `*_ext.rs` in `async-stripe-types/src/`.
4. **Assigning a new type to a crate?**
   - Edit `openapi/gen_crates.toml`. See [references/codegen.md](references/codegen.md).

## Critical Rules

- **Never edit files under `generated/`** -- they are overwritten by codegen. Change `openapi/` instead.
- **Miniserde is the primary deserializer**, not serde. Serde `Deserialize` is behind `"deserialize"` feature flags. Serde `Serialize` is always available for request params.
- **The `openapi/` directory is excluded from the workspace.** It has its own `Cargo.lock`. Build/run it with `cd openapi && cargo run --release`.
- **Two `cargo +nightly fmt` passes** are needed after codegen (single pass doesn't converge).
- Use **conventional commits** (`feat:`, `fix:`, `feat!:`) for automated versioning via release-plz.

## Commands

```sh
# Build (default members, excludes examples)
cargo build

# Regenerate from latest / pinned spec
cd openapi && cargo run --release -- --fetch latest
cd openapi && cargo run --release -- --fetch current

# Test (unit, no external deps)
cargo test --no-default-features --features "full serialize deserialize default-tls"

# Test (integration, requires stripe-mock on port 12111)
docker run --rm -d -it -p 12111-12112:12111-12112 stripe/stripe-mock:v0.197.0
cargo test -p async-stripe-tests --no-default-features --features "default-tls"

# Lint & format
cargo clippy --no-default-features --features "full serialize deserialize default-tls"
cargo +nightly fmt --all -- --check
```

## Workspace Layout

```
async-stripe/              Main client (hyper, async-std, blocking backends)
async-stripe-types/        Foundation: def_id!, Expandable<T>, List<T>, Object trait
async-stripe-client-core/  Client traits, pagination, request strategy
async-stripe-webhook/      Webhook verification + generated EventObject enum
generated/                 12 auto-generated domain crates (NEVER EDIT)
openapi/                   Code generator (EXCLUDED from workspace, own Cargo.lock)
tests/                     Integration tests (stripe-mock)
examples/                  Excluded from default build
```

## Serialization Model

| Direction | Library | Derives/Impls | Feature-gated? |
|-----------|---------|---------------|----------------|
| Outgoing (request params) | serde | `#[derive(serde::Serialize)]` | No, always on |
| Incoming (responses) | miniserde | Hand-generated `const _: () = { ... }` blocks | No, always on |
| Incoming (optional serde) | serde | `#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]` | Yes, `"deserialize"` |

## Key Patterns

**Request builder**: Each API operation is a struct wrapping a private `Builder` with `serde::Serialize`. Public builder methods use `impl Into<T>`. The struct implements `StripeRequest` with `build()` returning a `RequestBuilder`. GET uses `.query()`, POST/DELETE uses `.form()`.

**Miniserde `const` blocks**: Each response type gets a `const _: () = { ... }` anonymous scope containing a `Builder` struct, `Deserialize`/`MapBuilder`/`Map`/`ObjectDeser`/`FromValueOpt` impls. The anonymous scope avoids name collisions.

**Object field stripping**: Stripe's constant `"object": "balance"` fields are stripped from struct definitions, stored in `struct_.object_field`, and re-added via custom `Serialize` impls.

**Cyclic deps**: Types in dependency cycles live in `async-stripe-shared` (no inter-crate deps). Requests (which don't reference each other) stay in domain crates behind feature gates.

**Feature flags**: Each resource in a generated crate has its own feature (e.g., `charge = []`). A `full` feature enables all. Only requests are gated; type definitions are always compiled.

## References

- **[references/codegen.md](references/codegen.md)** -- Code generation pipeline, templates, crate inference, type model, deduplication, overrides. Read when modifying anything in `openapi/`.
- **[references/architecture.md](references/architecture.md)** -- Hand-written crate internals, client abstraction, pagination, webhook system, testing. Read when modifying foundation crates or writing tests.
