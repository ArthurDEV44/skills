# Axum Error Handling Reference

Source: https://docs.rs/axum/0.8/axum/error_handling/index.html

## Core Principle

Axum uses `Infallible` error types — handlers always produce HTTP responses, never connection-terminating errors. Return `Result<T, E>` where both `T` and `E` implement `IntoResponse`.

## Pattern 1: anyhow Wrapper (Recommended)

The standard pattern for converting any error into a 500 response:

```rust
use axum::{http::StatusCode, response::{IntoResponse, Response}};

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        tracing::error!("Application error: {:#}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on any error type that converts to anyhow::Error
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

// Now handlers can use `?` freely:
async fn handler(
    State(db): State<PgPool>,
) -> Result<Json<Vec<User>>, AppError> {
    let users = sqlx::query_as("SELECT * FROM users")
        .fetch_all(&db)
        .await?; // sqlx::Error → anyhow::Error → AppError
    Ok(Json(users))
}
```

## Pattern 2: Typed Errors with thiserror

For APIs that return structured error responses:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
enum ApiError {
    #[error("resource not found")]
    NotFound,
    #[error("unauthorized")]
    Unauthorized,
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("internal error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, self.to_string()),
            ApiError::Validation(msg) => (StatusCode::UNPROCESSABLE_ENTITY, msg.clone()),
            ApiError::Internal(err) => {
                tracing::error!("Internal error: {err:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error".to_string())
            }
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
```

## Pattern 3: JSON Error Responses

Consistent JSON error bodies for REST APIs:

```rust
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<serde_json::Value>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error, details) = match self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, "not_found", None),
            ApiError::Validation(d) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "validation_error",
                Some(serde_json::json!({ "fields": d })),
            ),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", None),
        };

        (
            status,
            Json(ErrorResponse {
                error: error.to_string(),
                details,
            }),
        )
            .into_response()
    }
}
```

## Pattern 4: Customizing Extractor Rejections

Override default rejection responses from built-in extractors:

```rust
use axum::extract::rejection::JsonRejection;

// Wrap Json to customize its rejection
struct AppJson<T>(T);

impl<S, T> FromRequest<S> for AppJson<T>
where
    S: Send + Sync,
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(AppJson(value)),
            Err(rejection) => Err(ApiError::Validation(rejection.body_text())),
        }
    }
}
```

## Pattern 5: Result with StatusCode

Simplest pattern — return `StatusCode` as the error:

```rust
async fn handler() -> Result<Json<User>, StatusCode> {
    let user = find_user().ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(user))
}

// With tuple for custom message
async fn handler() -> Result<Json<User>, (StatusCode, String)> {
    let user = find_user()
        .ok_or_else(|| (StatusCode::NOT_FOUND, "User not found".to_string()))?;
    Ok(Json(user))
}
```

## HandleErrorLayer — Middleware Errors

For Tower middleware that returns errors (e.g., timeouts):

```rust
use axum::error_handling::HandleErrorLayer;
use tower::ServiceBuilder;

let app = Router::new()
    .route("/", get(handler))
    .layer(
        ServiceBuilder::new()
            .layer(HandleErrorLayer::new(|err: BoxError| async move {
                if err.is::<tower::timeout::error::Elapsed>() {
                    (StatusCode::REQUEST_TIMEOUT, "Request timed out")
                } else {
                    (StatusCode::INTERNAL_SERVER_ERROR, "Internal error")
                }
            }))
            .layer(TimeoutLayer::new(Duration::from_secs(10)))
    );
```

With extractors in error handler for richer context:

```rust
async fn handle_error(
    method: Method,
    uri: Uri,
    err: BoxError,
) -> (StatusCode, String) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("{method} {uri} failed: {err}"),
    )
}
```

## HandleError — Fallible Services

For routing to services that may fail:

```rust
use axum::error_handling::HandleError;

let app = Router::new().route_service(
    "/",
    HandleError::new(fallible_service, |err: anyhow::Error| async {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {err}"))
    }),
);
```

## Key Guidelines

- Never expose internal error details to clients in production (log them, return generic messages)
- Use `tracing::error!` to log the full error chain before converting to response
- `#[from]` attribute in thiserror enables `?` operator for automatic conversion
- Prefer typed errors (`ApiError`) over generic `anyhow` for public APIs with varied status codes
- Use `anyhow` wrapper for internal services where all errors are 500s
