---
name: clerk-rs-sdk
description: "clerk-rs Rust SDK for Clerk authentication: client setup, typed API calls, JWT validation, and framework middleware (Axum, Actix, Rocket, Poem). Use when writing, reviewing, or refactoring Rust code using clerk-rs: (1) Setting up ClerkConfiguration and Clerk client, (2) Calling Clerk Backend API endpoints (users, organizations, sessions, invitations, etc.), (3) Adding JWT authentication middleware to Axum, Actix-web, Rocket, or Poem, (4) Validating JWTs with ClerkAuthorizer and JwksProvider, (5) Working with ClerkJwt claims, ActiveOrganization permissions, or Actor tokens, (6) Configuring JWKS caching with MemoryCacheJwksProvider, (7) Using clerk_rs::apis or clerk_rs::models types, (8) Adding clerk-rs feature flags to Cargo.toml, (9) Protecting routes with ClerkLayer, ClerkMiddleware, ClerkGuard, or ClerkPoemMiddleware."
---

# clerk-rs SDK

Community Rust SDK for the Clerk Backend API. Single crate with typed API modules (OpenAPI-generated), data models, and optional framework middleware for JWT validation.

## Quick Start

```rust
use clerk_rs::{clerk::Clerk, ClerkConfiguration};

let config = ClerkConfiguration::new(None, None, Some("sk_live_xxx".to_string()), None);
let client = Clerk::new(config);
```

Cargo.toml:
```toml
clerk-rs = "0.4"                                    # default: rustls-tls
clerk-rs = { version = "0.4", features = ["axum"] } # with framework middleware
```

## API Usage

### Typed API (recommended)

```rust
use clerk_rs::apis::users_api::User;
let users = User::get_user_list(&client, None, None, None).await?;

use clerk_rs::apis::organizations_api::Organization;
let org = Organization::get_organization(&client, "org_123").await?;
```

### Raw Endpoints

```rust
use clerk_rs::endpoints::{ClerkGetEndpoint, ClerkDynamicGetEndpoint};

let res = client.get(ClerkGetEndpoint::GetUserList).await?;
let user = client.get_with_params(ClerkDynamicGetEndpoint::GetUser, vec!["user_123"]).await?;
```

## Framework Middleware

All middleware takes: `JwksProvider`, excluded routes (`Option<Vec<String>>`), and `validate_session_cookie: bool`.

### Axum

```rust
use clerk_rs::validators::{axum::ClerkLayer, jwks::MemoryCacheJwksProvider};

let app = Router::new()
    .route("/protected", get(handler))
    .layer(ClerkLayer::new(MemoryCacheJwksProvider::new(clerk), None, true));

// Extract JWT: Extension(jwt): Extension<ClerkJwt>
```

### Actix-web

```rust
use clerk_rs::validators::{actix::ClerkMiddleware, jwks::MemoryCacheJwksProvider};

App::new()
    .wrap(ClerkMiddleware::new(MemoryCacheJwksProvider::new(clerk), None, true))
```

### Rocket

```rust
use clerk_rs::validators::rocket::{ClerkGuard, ClerkGuardConfig};

rocket::build()
    .manage(ClerkGuardConfig::new(MemoryCacheJwksProvider::new(clerk), None, true))

// Route guard: fn handler(jwt: ClerkGuard<MemoryCacheJwksProvider>)
```

### Poem

```rust
use clerk_rs::validators::poem::ClerkPoemMiddleware;

Route::new().at("/protected", get(handler))
    .with(ClerkPoemMiddleware::new(MemoryCacheJwksProvider::new(clerk), true, None));

// Extract JWT: Data<&ClerkJwt>
```

## Key Types

- `ClerkJwt` — decoded JWT with `sub`, `sid`, `org: Option<ActiveOrganization>`, `act: Option<Actor>`, `other: Map<String, Value>` (custom claims via `#[serde(flatten)]`)
- `ActiveOrganization` — `id`, `slug`, `role`, `permissions` with `has_permission()` / `has_role()`
- `ClerkError` — `Unauthorized(String)` | `InternalServerError(String)`
- `MemoryCacheJwksProvider` — cached JWKS with configurable lifetime (default 1h) and rate-limited refresh on unknown key

## Codebase Notes

- `src/apis/` and `src/models/` are **OpenAPI-generated** — may be overwritten on regeneration
- `src/validators/` is **hand-written** — core auth logic and framework middleware
- Formatting: `max_width = 150`, hard tabs, `imports_granularity = "Crate"`
- Tests use `#[tokio::test]`, `mockito` for HTTP mocking, `rsa`/`base64` for JWT fixtures
- Only RS256 algorithm is supported for JWT validation

## Resources

For complete API method listing, endpoint enums, JWKS provider configuration, and full framework middleware examples, see [references/api_reference.md](references/api_reference.md).
