# Axum Responses Reference

Source: https://docs.rs/axum/0.8/axum/response/index.html

## IntoResponse Trait

Anything implementing `IntoResponse` can be returned from a handler.

### Built-in Implementations

| Type | Content-Type | Notes |
|------|-------------|-------|
| `()` | — | Empty 200 response |
| `String` / `&'static str` | `text/plain; charset=utf-8` | |
| `Vec<u8>` / `Bytes` | `application/octet-stream` | |
| `Json<T>` | `application/json` | T: Serialize |
| `Html<T>` | `text/html` | |
| `Form<T>` | `application/x-www-form-urlencoded` | T: Serialize |
| `StatusCode` | — | Empty body with status |
| `HeaderMap` | — | Headers only |
| `Redirect` | — | 3xx redirect |
| `Sse<S>` | `text/event-stream` | Server-Sent Events |
| `NoContent` | — | 204 No Content |
| `Response<Body>` | — | Full control |

### Tuple Responses

Combine status, headers, and body:

```rust
// Status + body
async fn handler() -> (StatusCode, String) {
    (StatusCode::NOT_FOUND, "not found".to_string())
}

// Status + headers + body
async fn handler() -> (StatusCode, [(HeaderName, &'static str); 1], String) {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain")],
        "hello".to_string(),
    )
}

// Headers + body
async fn handler() -> (HeaderMap, String) {
    let mut headers = HeaderMap::new();
    headers.insert("x-custom", "value".parse().unwrap());
    (headers, "hello".to_string())
}

// Extension + body (sets response extensions)
async fn handler() -> (Extension<MyData>, String) {
    (Extension(MyData { /* ... */ }), "hello".to_string())
}
```

### Custom IntoResponse

```rust
enum AppError {
    NotFound,
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            AppError::NotFound => StatusCode::NOT_FOUND.into_response(),
            AppError::Internal(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response()
            }
        }
    }
}

async fn handler() -> Result<Json<User>, AppError> {
    let user = find_user().map_err(|e| AppError::Internal(e.to_string()))?;
    Ok(Json(user))
}
```

## Response Types

### Redirect

```rust
// 303 See Other (changes method to GET, use after form POST)
Redirect::to("/new-location")

// 307 Temporary (preserves method and body)
Redirect::temporary("/temp")

// 308 Permanent (preserves method and body)
Redirect::permanent("/new-permanent")
```

### Html

```rust
async fn handler() -> Html<&'static str> {
    Html("<h1>Hello</h1>")
}
```

### NoContent

```rust
use axum::response::NoContent;

async fn delete_item(Path(id): Path<u64>) -> NoContent {
    // delete from database...
    NoContent
}
```

### Server-Sent Events (SSE)

```rust
use axum::response::sse::{Event, Sse, KeepAlive};
use tokio_stream::StreamExt as _;
use futures_util::stream::{self, Stream};
use std::convert::Infallible;
use std::time::Duration;

async fn sse_handler() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::repeat_with(|| Event::default().data("hello"))
        .map(Ok)
        .throttle(Duration::from_secs(1));
    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

Event builder methods:
- `.data("payload")` — event data
- `.event("event-name")` — event type
- `.id("unique-id")` — event ID for reconnection
- `.retry(Duration)` — client reconnection interval
- `.comment("text")` — SSE comment

### Low-Level Response Builder

```rust
use axum::response::Response;
use axum::body::Body;

async fn handler() -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header("x-custom", "value")
        .body(Body::from("hello"))
        .unwrap()
}
```

## Error Handling Model

Axum requires `Infallible` error type — services always produce responses, never connection-terminating errors.

Handlers returning `Result<T, E>` where both T and E implement `IntoResponse` satisfy this:

```rust
async fn handler() -> Result<String, StatusCode> {
    // Err(StatusCode::BAD_REQUEST) is still a valid HTTP response
    Ok("ok".to_string())
}
```

For fallible services, use `HandleError`:

```rust
use axum::error_handling::HandleError;

let app = Router::new().route_service(
    "/",
    HandleError::new(fallible_service, |err: anyhow::Error| async {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {err}"))
    }),
);
```

## AppendHeaders

Add headers without overriding existing ones:

```rust
use axum::response::AppendHeaders;

async fn handler() -> (AppendHeaders<[(HeaderName, &'static str); 1]>, String) {
    (AppendHeaders([(header::CONTENT_TYPE, "text/plain")]), "hello".to_string())
}
```

## Rejection Types

All built-in extractors have typed rejection types that implement `IntoResponse`:

| Rejection | Status | Source |
|-----------|--------|--------|
| `JsonRejection` | 400/415/422 | `Json` extractor |
| `PathRejection` | 400 | `Path` extractor |
| `QueryRejection` | 400 | `Query` extractor |
| `FormRejection` | 400/415 | `Form` extractor |
| `ExtensionRejection` | 500 | `Extension` extractor |
| `MultipartRejection` | 400 | `Multipart` extractor |
| `WebSocketUpgradeRejection` | 400 | `WebSocketUpgrade` |

Customize rejection responses by implementing `From<OriginalRejection>` for your error type.
