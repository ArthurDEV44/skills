---
name: rust-axum
description: "Axum 0.8 web framework for Rust: routing, extractors, handlers, middleware, responses, state, WebSockets, SSE, error handling, testing, Tower. Use when writing or reviewing Axum code: (1) Router and routes, (2) Extractors (Path, Query, Json, Form, State, Host, Multipart), (3) Responses (Json, Html, Redirect, SSE, NoContent, IntoResponse), (4) Middleware (from_fn, Tower layers, CORS, tracing), (5) State with FromRef substates and Arc, (6) WebSocket, (7) Error handling with anyhow/thiserror, (8) Nesting/merging routers, (9) Graceful shutdown, (10) Custom extractors (FromRequest, FromRequestParts), (11) Testing handlers, (12) debug_handler macro."
---

# Rust Axum 0.8

Axum is a web framework focused on ergonomics and modularity, built on Tokio, Hyper, and Tower.

Official docs: https://docs.rs/axum/0.8/axum/

## Quick Start

```rust
use axum::{routing::get, Router};

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(|| async { "Hello, World!" }));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

Cargo.toml:

```toml
[dependencies]
axum = { version = "0.8", features = ["json", "form", "query", "macros"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors", "compression-gzip"] }
tracing = "0.1"
tracing-subscriber = "0.3"
```

## Feature Flags

Core: `http1`, `json`, `form`, `query`, `matched-path`, `original-uri`, `tokio`, `tracing`

Optional: `http2`, `macros` (`#[debug_handler]`, `#[debug_middleware]`), `multipart`, `ws`

## Handlers

Async functions accepting extractors (max 16) and returning `impl IntoResponse`. Body-consuming extractors must be last.

Use `#[debug_handler]` (requires `macros` feature) for better compile-time error messages:

```rust
use axum::{extract::{Path, Query, Json, State}, http::StatusCode, debug_handler};

#[debug_handler]
async fn create_user(
    State(db): State<DbPool>,
    Json(payload): Json<CreateUser>,
) -> Result<Json<User>, AppError> {
    let user = db.create(payload).await?;
    Ok(Json(user))
}
```

## Routing

```rust
use axum::{routing::{get, post, put, delete, any}, Router};

let app = Router::new()
    .route("/", get(root))
    .route("/users", get(list_users).post(create_user))
    .route("/users/{id}", get(show_user).put(update_user).delete(delete_user))
    .route("/ws", any(ws_handler))
    .nest("/api", api_routes())
    .merge(health_routes())
    .fallback(not_found_handler)
    .method_not_allowed_fallback(method_not_allowed)
    .with_state(app_state);
```

Path syntax: `/{key}` (single segment), `/{*key}` (wildcard/rest). For complete routing API: see [references/routing.md](references/routing.md).

## Extractors

| Extractor | Source | Feature | Body? |
|-----------|--------|---------|-------|
| `Path<T>` | URL segments | - | No |
| `Query<T>` | Query string | `query` | No |
| `Json<T>` | JSON body | `json` | Yes |
| `Form<T>` | Form body | `form` | Yes |
| `State<S>` | App state | - | No |
| `Host` | Host header/URI | - | No |
| `Multipart` | File uploads | `multipart` | Yes |
| `HeaderMap` | Headers | - | No |
| `Extension<T>` | Extensions | - | No |
| `ConnectInfo<T>` | Connection | `tokio` | No |
| `WebSocketUpgrade` | WS upgrade | `ws` | Yes |
| `MatchedPath` | Route pattern | `matched-path` | No |
| `OriginalUri` | Original URI | `original-uri` | No |
| `Request` | Full request | - | Yes |
| `String` / `Bytes` | Raw body | - | Yes |

For custom extractors, rejection handling, DefaultBodyLimit, and all extractor details: see [references/extractors.md](references/extractors.md).

## Responses

```rust
// Tuple: (StatusCode, headers, body)
async fn handler() -> (StatusCode, [(HeaderName, &'static str); 1], Json<Data>) { /* ... */ }

// Custom error type
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.message).into_response()
    }
}
```

Built-in: `()`, `String`, `Json<T>`, `Html<T>`, `Form<T>`, `Redirect`, `Sse<S>`, `NoContent`, `StatusCode`, tuples.

For SSE, Redirect variants, custom IntoResponse, AppendHeaders: see [references/responses.md](references/responses.md).

## Error Handling

The standard anyhow wrapper pattern for handlers:

```rust
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", self.0)).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(err: E) -> Self { Self(err.into()) }
}

async fn handler(State(db): State<DbPool>) -> Result<Json<User>, AppError> {
    let user = db.get_user(1).await?; // ? auto-converts any error
    Ok(Json(user))
}
```

For comprehensive error handling patterns (thiserror, rejections, error JSON responses): see [references/error-handling.md](references/error-handling.md).

## Middleware

```rust
use axum::middleware::{self, Next};

async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // validate headers...
    Ok(next.run(request).await)
}

let app = Router::new()
    .route("/protected", get(handler))
    .route_layer(middleware::from_fn(auth_middleware));
```

Use `from_fn_with_state()` for state access. Layer execution order: bottom-to-top with `.layer()`, top-to-bottom with `ServiceBuilder`.

For from_extractor, custom Tower Service, HandleErrorLayer, tower-http layers, CorsLayer config: see [references/middleware.md](references/middleware.md).

## State Management

```rust
#[derive(Clone)]
struct AppState {
    db: PgPool,
    redis: RedisPool,
}

// Substates via FromRef
#[derive(Clone, FromRef)]
struct AppState {
    db: DbState,
    cache: CacheState,
}

// Mutable shared state
#[derive(Clone)]
struct AppState {
    counter: Arc<Mutex<u64>>,
}
```

Use `Arc<AppState>` when state is not cheaply cloneable. Use `tokio::sync::Mutex` when holding lock across `.await`.

## Graceful Shutdown

```rust
use tokio::signal;

let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal())
    .await?;

async fn shutdown_signal() {
    signal::ctrl_c().await.expect("install handler");
}
```

## Common Patterns

### CRUD REST API

```rust
let app = Router::new()
    .route("/items", get(list_items).post(create_item))
    .route("/items/{id}", get(get_item).put(update_item).delete(delete_item))
    .layer(TraceLayer::new_for_http())
    .layer(CorsLayer::permissive())
    .with_state(state);
```

### WebSocket

```rust
async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            socket.send(Message::Text(format!("Echo: {text}").into())).await.ok();
        }
    }
}
```

### Static Files (tower-http)

```rust
use tower_http::services::{ServeDir, ServeFile};

let app = Router::new()
    .route("/api/health", get(|| async { "ok" }))
    .nest_service("/static", ServeDir::new("static"))
    .fallback_service(ServeFile::new("static/index.html"));
```

### Testing

```rust
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

#[tokio::test]
async fn test_handler() {
    let app = create_app();
    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
```

For comprehensive testing patterns: see [references/testing.md](references/testing.md).

## References

- **Routing** (Router, nest, merge, fallback, route_layer): [references/routing.md](references/routing.md)
- **Extractors** (Path, Query, Json, State, Host, custom extractors): [references/extractors.md](references/extractors.md)
- **Middleware** (from_fn, Tower layers, CorsLayer, error handling): [references/middleware.md](references/middleware.md)
- **Responses** (IntoResponse, SSE, Redirect, NoContent, error types): [references/responses.md](references/responses.md)
- **Error Handling** (anyhow, thiserror, rejections, JSON errors): [references/error-handling.md](references/error-handling.md)
- **Testing** (oneshot, TestClient, integration tests): [references/testing.md](references/testing.md)
