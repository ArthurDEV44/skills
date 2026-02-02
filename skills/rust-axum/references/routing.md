# Axum Routing Reference

Source: https://docs.rs/axum/0.8.8/axum/routing/index.html

## Router

Primary struct for composing handlers and services.

### Creating and Adding Routes

```rust
use axum::{Router, routing::get};

let app = Router::new()
    .route("/", get(root))
    .route("/users/{id}", get(show_user))
    .route("/assets/{*path}", get(serve_asset));
```

Path syntax:
- Static: `/foo/bar`
- Capture: `/{key}` — extracts a single segment
- Wildcard: `/{*key}` — captures remaining path

### route_service

Route to a custom `Service` instead of a handler:

```rust
use axum::routing::any_service;
use tower::service_fn;

let app = Router::new()
    .route_service("/", any_service(service_fn(|_: Request| async {
        Ok::<_, Infallible>(Response::new(Body::from("Hi")))
    })))
    .route_service("/static/Cargo.toml", ServeFile::new("Cargo.toml"));
```

### Nesting

Compose applications from smaller routers. Nested router sees URI with matched prefix stripped:

```rust
let user_routes = Router::new().route("/{id}", get(|| async {}));
let api_routes = Router::new().nest("/users", user_routes);
let app = Router::new().nest("/api", api_routes);
// Accepts: GET /api/users/{id}
```

### nest_service

Same as `nest` but accepts an arbitrary `Service` instead of a `Router`.

### Merging

Combine paths and fallbacks of two routers:

```rust
let user_routes = Router::new()
    .route("/users", get(users_list))
    .route("/users/{id}", get(users_show));
let team_routes = Router::new()
    .route("/teams", get(teams_list));
let app = Router::new().merge(user_routes).merge(team_routes);
```

Panics if both routers define a fallback.

### Middleware

**`layer()`** — apply Tower `Layer` to all routes (runs after routing):

```rust
use tower_http::trace::TraceLayer;

let app = Router::new()
    .route("/foo", get(|| async {}))
    .route("/bar", get(|| async {}))
    .layer(TraceLayer::new_for_http());
```

Only affects routes added before `.layer()`.

**`route_layer()`** — apply middleware only to matched routes (unmatched returns 404):

```rust
use tower_http::validate_request::ValidateRequestHeaderLayer;

let app = Router::new()
    .route("/foo", get(|| async {}))
    .route_layer(ValidateRequestHeaderLayer::bearer("password"));
// GET /foo without token → 401
// GET /not-found without token → 404
```

### Fallbacks

**`fallback(handler)`** — handler for unmatched routes:

```rust
async fn fallback(uri: Uri) -> (StatusCode, String) {
    (StatusCode::NOT_FOUND, format!("No route for {uri}"))
}
let app = Router::new()
    .route("/foo", get(|| async {}))
    .fallback(fallback);
```

**`method_not_allowed_fallback(handler)`** — handler when route exists but method doesn't match:

```rust
let router = Router::new()
    .route("/", get(hello_world))
    .fallback(default_fallback)
    .method_not_allowed_fallback(handle_405);
// POST / → calls handle_405
// GET /hello → calls default_fallback
```

**`reset_fallback()`** — clear fallback before merging routers that both have fallbacks.

### State Management

```rust
#[derive(Clone)]
struct AppState { /* ... */ }

let app = Router::new()
    .route("/", get(|State(state): State<AppState>| async { /* ... */ }))
    .with_state(AppState { /* ... */ });
```

`with_state(state)` returns `Router<()>` (no missing state). Only `Router<()>` has `into_make_service`.

### Server Conversion

**`into_make_service()`** — for use with `axum::serve`:

```rust
let app = Router::new().route("/", get(|| async { "Hi!" }));
let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
axum::serve(listener, app).await?;
```

**`into_make_service_with_connect_info::<SocketAddr>()`** — enables `ConnectInfo` extraction:

```rust
async fn handler(ConnectInfo(addr): ConnectInfo<SocketAddr>) -> String {
    format!("Hello {addr}")
}
let app = Router::new().route("/", get(handler));
let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
```

### Utility Methods

- `Router::new()` — empty router, responds 404 by default
- `has_routes()` — whether at least one route is registered
- `as_service()` / `into_service()` — convert to Tower `Service` (useful in tests)
- `without_v07_checks()` — allow legacy `:param` and `*wildcard` syntax

## MethodRouter

Routes based on HTTP method. Functions: `get`, `post`, `put`, `delete`, `patch`, `head`, `options`, `trace`, `connect`, `any`, `on`.

Variants: `get_service`, `post_service`, etc. for routing to `Service` implementations.

## Middleware Execution Order

With `Router::layer()`, middleware wraps bottom-to-top:

```
requests → layer_three → layer_two → layer_one → handler → response
```

With `ServiceBuilder`, layers execute top-to-bottom (more intuitive).
