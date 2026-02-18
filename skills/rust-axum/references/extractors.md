# Axum Extractors Reference

Source: https://docs.rs/axum/0.8/axum/extract/index.html

## Overview

Extractors automatically parse typed data from HTTP requests. Handler functions accept extractors as parameters; axum processes them left-to-right.

**Critical rules:**
- Only ONE extractor may consume the request body per handler
- Body-consuming extractors MUST be the last parameter
- Extractors run sequentially in parameter order
- Maximum 16 extractors per handler

## Body Extractors (must be last)

### Json\<T\> (feature: `json`)

Deserializes JSON body. Requires `Content-Type: application/json`.

```rust
#[derive(Deserialize)]
struct CreateUser { email: String, password: String }

async fn create_user(Json(payload): Json<CreateUser>) {
    // payload is CreateUser
}
```

Also works as response (sets `Content-Type: application/json`):

```rust
async fn get_user(Path(id): Path<Uuid>) -> Json<User> {
    Json(find_user(id).await)
}
```

Rejection: `JsonRejection` (subtypes: `JsonDataError`, `JsonSyntaxError`, `MissingJsonContentType`).

### Form\<T\> (feature: `form`)

URL-encoded form data. GET/HEAD: from query string. Other methods: from body with `content-type: application/x-www-form-urlencoded`.

```rust
#[derive(Deserialize)]
struct SignUp { username: String, password: String }

async fn accept_form(Form(sign_up): Form<SignUp>) { /* ... */ }
```

Also works as response (sets `application/x-www-form-urlencoded`).

Rejection: `FormRejection` (subtypes: `FailedToDeserializeForm`, `FailedToDeserializeFormBody`, `InvalidFormContentType`).

### Multipart (feature: `multipart`)

Parses `multipart/form-data` for file uploads:

```rust
async fn upload(mut multipart: Multipart) {
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        let file_name = field.file_name().map(|f| f.to_string());
        let content_type = field.content_type().map(|c| c.to_string());
        let data = field.bytes().await.unwrap();
        println!("Length of `{name}` is {} bytes", data.len());
    }
}
```

Default body limit: 2MB. Adjust via `DefaultBodyLimit`.

### String / Bytes

- `String` — UTF-8 validated body
- `Bytes` — raw body bytes

### Request

Full `http::Request<Body>` — consumes entire request.

### RawForm

Unparsed form body as `Bytes`.

## Non-Body Extractors (any position)

### Path\<T\>

Captures URL path parameters via serde. Percent-encoded values auto-decoded.

```rust
// Single param
async fn user(Path(user_id): Path<Uuid>) { /* ... */ }

// Tuple
async fn show(Path((user_id, team_id)): Path<(Uuid, Uuid)>) { /* ... */ }

// Struct (field names must match path labels)
#[derive(Deserialize)]
struct Params { user_id: Uuid, team_id: Uuid }
async fn show(Path(params): Path<Params>) { /* ... */ }

// HashMap or Vec
async fn params(Path(params): Path<HashMap<String, String>>) { /* ... */ }
```

`Option<Path<T>>` for optional path params. Rejection: `PathRejection` (subtypes: `FailedToDeserializePathParams`, `InvalidUtf8InPathParam`).

### Query\<T\> (feature: `query`)

Deserializes query string (`?page=2&per_page=30`):

```rust
#[derive(Deserialize)]
struct Pagination { page: usize, per_page: usize }

async fn list(Query(pagination): Query<Pagination>) { /* ... */ }
```

For duplicate keys (`?foo=1&foo=2`), use `axum_extra::extract::Query`.

Rejection: `QueryRejection` (subtypes: `FailedToDeserializeQueryString`).

### State\<S\>

Access global application state:

```rust
#[derive(Clone)]
struct AppState { db: Pool }

async fn handler(State(state): State<AppState>) { /* ... */ }

let app = Router::new()
    .route("/", get(handler))
    .with_state(AppState { db: pool });
```

**Substates** via `FromRef`:

```rust
#[derive(Clone, FromRef)]
struct AppState { api: ApiState, db: DbState }

// Handlers can extract sub-states directly:
async fn handler(State(api): State<ApiState>) { /* ... */ }
```

**Shared mutable state** — use `Arc<Mutex<T>>`:

```rust
#[derive(Clone)]
struct AppState { data: Arc<Mutex<String>> }

async fn handler(State(state): State<AppState>) {
    let mut data = state.data.lock().expect("poisoned");
    *data = "updated".to_owned();
}
```

Use `tokio::sync::Mutex` if holding across `.await` points.

### Host

Extracts the host from the request (checks `Host` header first, then request URI):

```rust
use axum::extract::Host;

async fn handler(Host(hostname): Host) -> String {
    format!("Requested host: {hostname}")
}
```

### HeaderMap

All request headers:

```rust
async fn handler(headers: HeaderMap) {
    if let Some(auth) = headers.get("authorization") {
        // ...
    }
}
```

### Extension\<T\>

Retrieve data from request extensions (typically set by middleware):

```rust
async fn handler(Extension(user): Extension<CurrentUser>) { /* ... */ }
```

### MatchedPath

The matched route pattern string (e.g., `/users/{id}`):

```rust
async fn handler(MatchedPath(path): MatchedPath) {
    println!("Matched: {path}"); // "/users/{id}"
}
```

### NestedPath

The path prefix from nesting context.

### OriginalUri (feature: `original-uri`)

Original request URI before any rewrites.

### ConnectInfo\<T\> (feature: `tokio`)

Connection metadata. Requires `into_make_service_with_connect_info`:

```rust
async fn handler(ConnectInfo(addr): ConnectInfo<SocketAddr>) -> String {
    format!("Hello {addr}")
}
```

### WebSocketUpgrade (feature: `ws`)

Upgrades HTTP connection to WebSocket:

```rust
async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    while let Some(msg) = socket.recv().await {
        if let Ok(msg) = msg {
            socket.send(msg).await.ok();
        }
    }
}
```

Split for concurrent read/write: `socket.split()` via `StreamExt`.

### RawQuery / RawPathParams

Unparsed query string or path parameters.

## DefaultBodyLimit

Configure body size limits (default 2MB):

```rust
// Custom limit
let app = Router::new()
    .route("/", post(handler))
    .layer(DefaultBodyLimit::max(1024));

// Disable default limit (pair with RequestBodyLimitLayer)
let app = Router::new()
    .route("/", post(handler))
    .layer(DefaultBodyLimit::disable())
    .layer(RequestBodyLimitLayer::new(10 * 1_000_000));

// Route-specific limit
let app = Router::new()
    .route("/", post(handler).layer(DefaultBodyLimit::max(1024)))
    .route("/foo", post(other_handler));
```

## Custom Extractors

### FromRequestParts (no body)

```rust
struct MyExtractor { /* ... */ }

impl<S: Send + Sync> FromRequestParts<S> for MyExtractor {
    type Rejection = StatusCode;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        // extract from parts.headers, parts.uri, etc.
        Ok(MyExtractor { /* ... */ })
    }
}
```

### FromRequest (consumes body)

```rust
struct JsonBody<T>(T);

impl<S, T> FromRequest<S> for JsonBody<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = JsonRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state).await?;
        Ok(JsonBody(value))
    }
}
```

### RequestExt — Manual Extraction in Middleware

Use `RequestExt` to run extractors manually from a `Request`:

```rust
use axum::RequestExt;

async fn my_middleware(mut req: Request, next: Next) -> Response {
    // Extract parts without consuming body
    let path = req.extract_parts::<Path<HashMap<String, String>>>().await.unwrap();

    // Extract with body consumption
    let Json(body) = req.extract::<Json<Value>, _>().await.unwrap();

    next.run(req).await
}
```

### Optional Extractors

Wrap in `Option<T>` or `Result<T, T::Rejection>` for graceful failure handling:

```rust
// Returns None instead of rejecting
async fn handler(user: Option<Extension<CurrentUser>>) { /* ... */ }

// Returns Err with the rejection instead of rejecting
async fn handler(user: Result<Extension<CurrentUser>, ExtensionRejection>) { /* ... */ }
```
