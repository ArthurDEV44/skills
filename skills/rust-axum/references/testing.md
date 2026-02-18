# Axum Testing Reference

Source: https://docs.rs/axum/0.8/axum/

## Tower oneshot Testing (No Extra Dependencies)

The built-in approach using Tower's `ServiceExt::oneshot`:

```rust
use axum::{
    body::Body,
    http::{Request, StatusCode},
    routing::get,
    Router,
};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn create_app() -> Router {
    Router::new().route("/", get(|| async { "Hello, World!" }))
}

#[tokio::test]
async fn test_hello() {
    let app = create_app();

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], b"Hello, World!");
}
```

### GET with Query Parameters

```rust
#[tokio::test]
async fn test_query() {
    let app = create_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/search?q=rust&page=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
```

### POST with JSON Body

```rust
use axum::http::header;

#[tokio::test]
async fn test_create_user() {
    let app = create_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/users")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_string(&serde_json::json!({
                        "email": "test@example.com",
                        "name": "Test User"
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
}
```

### Reading JSON Response Body

```rust
#[tokio::test]
async fn test_json_response() {
    let app = create_app();

    let response = app
        .oneshot(Request::builder().uri("/users/1").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let user: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(user["name"], "Test User");
}
```

## Testing with State

Inject test state (e.g., test database) into the router:

```rust
fn create_app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/users", get(list_users).post(create_user))
        .with_state(state)
}

#[tokio::test]
async fn test_with_mock_db() {
    let state = AppState {
        db: setup_test_database().await,
    };
    let app = create_app_with_state(state);

    let response = app
        .oneshot(Request::builder().uri("/users").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
```

## Multiple Requests (into_service)

`oneshot` consumes the service. For multiple requests, use `as_service()` or `into_service()`:

```rust
#[tokio::test]
async fn test_multiple_requests() {
    let mut app = create_app().into_service();

    // First request
    let response = ServiceExt::<Request<Body>>::ready(&mut app)
        .await
        .unwrap()
        .call(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Second request
    let response = ServiceExt::<Request<Body>>::ready(&mut app)
        .await
        .unwrap()
        .call(Request::builder().uri("/users").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
```

## Testing with Headers

```rust
#[tokio::test]
async fn test_auth_header() {
    let app = create_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .header("Authorization", "Bearer test-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_missing_auth() {
    let app = create_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}
```

## Testing Response Headers

```rust
#[tokio::test]
async fn test_response_headers() {
    let app = create_app();

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "application/json"
    );
}
```

## Helper Function Pattern

Extract a reusable test helper:

```rust
async fn send_request(app: Router, method: &str, uri: &str, body: Option<serde_json::Value>) -> (StatusCode, serde_json::Value) {
    let mut builder = Request::builder().method(method).uri(uri);

    let body = if let Some(json) = body {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
        Body::from(serde_json::to_string(&json).unwrap())
    } else {
        Body::empty()
    };

    let response = app.oneshot(builder.body(body).unwrap()).await.unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&body).unwrap_or(serde_json::json!(null));

    (status, json)
}

#[tokio::test]
async fn test_crud() {
    let app = create_app();

    let (status, body) = send_request(
        app,
        "POST",
        "/users",
        Some(serde_json::json!({"name": "Alice"})),
    ).await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["name"], "Alice");
}
```

## Test Dependencies

```toml
[dev-dependencies]
tokio = { version = "1", features = ["macros", "rt"] }
tower = { version = "0.5", features = ["util"] }
http-body-util = "0.1"
serde_json = "1"
```

## Key Guidelines

- Always test both success and error paths
- Test 404 for non-existent routes and 405 for wrong methods
- Build app from a factory function to isolate test state
- Use `oneshot` for single-request tests (most common)
- Use `into_service` + `ready().call()` when testing sequences
- Test middleware by checking both authenticated and unauthenticated requests
- For integration tests with a real server, use `tokio::net::TcpListener::bind("127.0.0.1:0")` for random port assignment
