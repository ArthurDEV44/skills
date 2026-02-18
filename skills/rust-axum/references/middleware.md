# Axum Middleware Reference

Source: https://docs.rs/axum/0.8/axum/middleware/index.html

## Overview

Axum uses the Tower middleware ecosystem. Middleware from `tower` and `tower-http` works directly.

## Applying Middleware

Three levels:
- `Router::layer()` / `Router::route_layer()` — router-level
- `MethodRouter::layer()` / `MethodRouter::route_layer()` — method router-level
- `Handler::layer()` — handler-level

Use `tower::ServiceBuilder` for multiple layers (top-to-bottom execution order):

```rust
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;

let app = Router::new()
    .route("/", get(handler))
    .layer(
        ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(TimeoutLayer::new(Duration::from_secs(10)))
    );
```

**`layer()` vs `route_layer()`:**
- `layer()` — applies to ALL requests (including unmatched routes)
- `route_layer()` — applies ONLY to matched routes (unmatched get 404 without running middleware)

Use `route_layer()` for auth middleware so unauthenticated users still get 404 for non-existent routes.

## Writing Middleware

### 1. from_fn — Quick async middleware

```rust
use axum::middleware::{self, Next};

async fn my_middleware(request: Request, next: Next) -> Response {
    // pre-processing
    let response = next.run(request).await;
    // post-processing
    response
}

let app = Router::new()
    .route("/", get(handler))
    .layer(middleware::from_fn(my_middleware));
```

With extractors (extractors run before the inner service):

```rust
async fn auth(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    match get_token(&headers) {
        Some(token) if token_is_valid(token) => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
```

**With state** — use `from_fn_with_state()` (`from_fn` does NOT support `State`):

```rust
async fn check_state(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    // use state...
    next.run(request).await
}

let state = AppState { /* ... */ };
let app = Router::new()
    .route("/", get(handler))
    .route_layer(middleware::from_fn_with_state(state.clone(), check_state))
    .with_state(state);
```

### 2. from_extractor — Extractor as middleware

Runs extractor before inner service. Rejects if extraction fails:

```rust
struct RequireAuth;

impl<S: Send + Sync> FromRequestParts<S> for RequireAuth {
    type Rejection = StatusCode;
    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // validate auth header
        Ok(RequireAuth)
    }
}

let app = Router::new()
    .route("/", get(handler))
    .route_layer(middleware::from_extractor::<RequireAuth>());
```

### 3. map_request / map_response

Simple request/response transformations:

```rust
use tower::ServiceBuilder;

let app = Router::new()
    .route("/", get(handler))
    .layer(
        ServiceBuilder::new()
            .map_request(|req: Request| { /* transform request */ req })
            .map_response(|res: Response| { /* transform response */ res })
    );
```

**With state** — `map_request_with_state` / `map_response_with_state`:

```rust
use axum::middleware::map_request_with_state;

async fn add_request_id<B>(
    State(state): State<AppState>,
    mut request: Request<B>,
) -> Request<B> {
    request.headers_mut().insert("x-request-id", /* ... */);
    request
}

let state = AppState { /* ... */ };
let app = Router::new()
    .route("/", get(handler))
    .route_layer(map_request_with_state(state.clone(), add_request_id))
    .with_state(state);
```

### 4. Custom tower::Service

For maximum control and publishable middleware:

```rust
use tower::{Layer, Service};
use std::task::{Context, Poll};
use futures_util::future::BoxFuture;

#[derive(Clone)]
struct MyLayer;

impl<S> Layer<S> for MyLayer {
    type Service = MyMiddleware<S>;
    fn layer(&self, inner: S) -> Self::Service {
        MyMiddleware { inner }
    }
}

#[derive(Clone)]
struct MyMiddleware<S> { inner: S }

impl<S> Service<Request> for MyMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let future = self.inner.call(request);
        Box::pin(async move {
            let response = future.await?;
            Ok(response)
        })
    }
}
```

**With state** in a custom Layer:

```rust
#[derive(Clone)]
struct MyLayer { state: AppState }

impl<S> Layer<S> for MyLayer {
    type Service = MyService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        MyService { inner, state: self.state.clone() }
    }
}

#[derive(Clone)]
struct MyService<S> { inner: S, state: AppState }

let app = Router::new()
    .route("/", get(handler))
    .layer(MyLayer { state: state.clone() })
    .with_state(state);
```

## Passing Data from Middleware to Handlers

Use request extensions:

```rust
// In middleware:
req.extensions_mut().insert(CurrentUser { id: 42 });

// In handler:
async fn handler(Extension(user): Extension<CurrentUser>) { /* ... */ }
```

## Error Handling in Middleware

Use `HandleErrorLayer` for middleware that may fail:

```rust
use axum::error_handling::HandleErrorLayer;

let app = Router::new()
    .route("/", get(handler))
    .layer(
        ServiceBuilder::new()
            .layer(HandleErrorLayer::new(|err: BoxError| async move {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {err}"))
            }))
            .layer(TimeoutLayer::new(Duration::from_secs(10)))
    );
```

With extractors in error handler:

```rust
async fn handle_error(method: Method, uri: Uri, err: BoxError) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, format!("{method} {uri} failed: {err}"))
}
```

## Common tower-http Middleware

| Layer | Purpose | Feature |
|-------|---------|---------|
| `TraceLayer` | Tracing/logging | `trace` |
| `CorsLayer` | CORS headers | `cors` |
| `CompressionLayer` | Response compression | `compression-gzip` / `compression-br` |
| `RequestIdLayer` | Request ID generation | `request-id` |
| `PropagateRequestIdLayer` | Request ID forwarding | `request-id` |
| `TimeoutLayer` | Request timeouts | `timeout` |
| `RequestBodyLimitLayer` | Body size enforcement | `limit` |
| `SetSensitiveHeadersLayer` | Redact sensitive headers in logs | `sensitive-headers` |
| `ValidateRequestHeaderLayer` | Header validation / bearer auth | `validate-request` |

### CorsLayer Configuration

```rust
use tower_http::cors::{CorsLayer, Any};
use axum::http::{Method, HeaderValue};

// Permissive (development)
let cors = CorsLayer::permissive();

// Restrictive (production)
let cors = CorsLayer::new()
    .allow_origin("https://example.com".parse::<HeaderValue>().unwrap())
    .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
    .allow_headers(Any)
    .max_age(Duration::from_secs(3600));

let app = Router::new()
    .route("/api/data", get(handler))
    .layer(cors);
```

### TraceLayer Setup

```rust
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer())
    .init();

let app = Router::new()
    .route("/", get(handler))
    .layer(TraceLayer::new_for_http());
```

## Key Notes

- `Router::layer()` runs AFTER routing — URI rewrites don't affect route matching
- Middleware execution: bottom-to-top with chained `.layer()`, top-to-bottom with `ServiceBuilder`
- Axum assumes services don't implement backpressure (always ready)
- Request extensions are NOT auto-copied to response extensions
- Use `#[debug_middleware]` (requires `macros` feature) for better compile-time error messages
