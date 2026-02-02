# Axum Middleware Reference

Source: https://docs.rs/axum/0.8.8/axum/middleware/index.html

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

With extractors:

```rust
async fn auth(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    match headers.get("authorization") {
        Some(_) => Ok(next.run(request).await),
        None => Err(StatusCode::UNAUTHORIZED),
    }
}
```

**With state** — use `from_fn_with_state()`:

```rust
async fn check_state(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    next.run(request).await
}

let app = Router::new()
    .route("/", get(handler))
    .layer(middleware::from_fn_with_state(state.clone(), check_state))
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

### 3. Tower Combinators

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

### 4. Custom tower::Service

For maximum control and publishable middleware:

```rust
use tower::{Layer, Service};
use std::task::{Context, Poll};
use std::pin::Pin;
use std::future::Future;

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

impl<S, B> Service<Request<B>> for MyMiddleware<S>
where
    S: Service<Request<B>, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<B>) -> Self::Future {
        let future = self.inner.call(request);
        Box::pin(async move {
            let response = future.await?;
            Ok(response)
        })
    }
}
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

| Layer | Purpose |
|-------|---------|
| `TraceLayer` | Tracing/logging |
| `CorsLayer` | CORS headers |
| `CompressionLayer` | Response compression |
| `RequestIdLayer` | Request ID generation |
| `PropagateRequestIdLayer` | Request ID forwarding |
| `TimeoutLayer` | Request timeouts |
| `RequestBodyLimitLayer` | Body size enforcement |

## Key Notes

- `Router::layer()` runs AFTER routing — URI rewrites don't affect route matching
- Middleware execution: bottom-to-top with chained `.layer()`, top-to-bottom with `ServiceBuilder`
- Axum assumes services don't implement backpressure (always ready)
- Request extensions are NOT auto-copied to response extensions
