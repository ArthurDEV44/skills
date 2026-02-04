# clerk-rs API Reference

## Table of Contents

- [Client Setup](#client-setup)
- [Two Usage Patterns](#two-usage-patterns)
- [Typed API Modules](#typed-api-modules)
- [Endpoint Enums](#endpoint-enums)
- [JWT Validation & Middleware](#jwt-validation--middleware)
- [JWKS Providers](#jwks-providers)
- [Key Types](#key-types)
- [Feature Flags](#feature-flags)
- [Framework Middleware](#framework-middleware)

## Client Setup

```rust
use clerk_rs::{clerk::Clerk, ClerkConfiguration};

// With bearer token (most common)
let config = ClerkConfiguration::new(None, None, Some("sk_live_xxx".to_string()), None);
let client = Clerk::new(config);

// Parameters: (basic_auth, oauth_access_token, bearer_access_token, api_key)
```

Base path defaults to `https://api.clerk.dev/v1`. The reqwest client is pre-configured with `Authorization: Bearer <token>` and a custom User-Agent header.

## Two Usage Patterns

### Pattern 1: Typed API Methods (recommended)

Each API module provides static async methods with typed parameters and responses:

```rust
use clerk_rs::apis::users_api::User;

// All parameters are Option<T> except the client reference
let users = User::get_user_list(
    &client,       // &Clerk
    None,          // limit: Option<f32>
    None,          // offset: Option<f32>
    None,          // order_by: Option<&str>
).await?;
```

### Pattern 2: Raw Endpoint Methods

Use Clerk struct methods with endpoint enums, returning `serde_json::Value`:

```rust
use clerk_rs::endpoints::ClerkGetEndpoint;

let res = client.get(ClerkGetEndpoint::GetUserList).await?;

// For endpoints with path parameters:
use clerk_rs::endpoints::ClerkDynamicGetEndpoint;
let user = client.get_with_params(
    ClerkDynamicGetEndpoint::GetUser,
    vec!["user_123"]
).await?;
```

## Typed API Modules

All in `clerk_rs::apis::*`:

| Module | Struct | Key Methods |
|--------|--------|-------------|
| `users_api` | `User` | `get_user_list`, `get_user`, `create_user`, `update_user`, `delete_user`, `ban_user`, `unban_user`, `verify_password`, `verify_totp`, `disable_mfa`, `update_user_metadata`, `get_users_count`, `users_get_organization_memberships` |
| `organizations_api` | `Organization` | `list_organizations`, `create_organization`, `get_organization`, `update_organization`, `delete_organization`, `merge_organization_metadata`, `upload_organization_logo` |
| `sessions_api` | `Session` | `get_session_list`, `get_session`, `revoke_session`, `verify_session`, `create_session_token_from_template` |
| `clients_api` | `Client` | `get_client_list`, `get_client`, `get_client_last_active_session`, `verify_client` |
| `invitations_api` | `Invitation` | `create_invitation`, `list_invitations`, `revoke_invitation` |
| `organization_memberships_api` | `OrganizationMembership` | `create_organization_membership`, `delete_organization_membership`, `list_organization_memberships`, `update_organization_membership`, `update_organization_membership_metadata` |
| `organization_invitations_api` | `OrganizationInvitation` | `create_organization_invitation`, `list_pending_organization_invitations`, `revoke_organization_invitation` |
| `email_addresses_api` | `EmailAddress` | `create_email_address`, `get_email_address`, `delete_email_address`, `update_email_address` |
| `phone_numbers_api` | `PhoneNumber` | `create_phone_number`, `get_phone_number`, `delete_phone_number`, `update_phone_number` |
| `jwt_templates_api` | `JwtTemplate` | `create_jwt_template`, `get_jwt_template`, `list_jwt_templates`, `update_jwt_template`, `delete_jwt_template` |
| `sign_in_tokens_api` | `SignInToken` | `create_sign_in_token`, `revoke_sign_in_token` |
| `webhooks_api` | `Webhook` | `create_svix_app`, `delete_svix_app`, `generate_svix_auth_url` |
| `actor_tokens_api` | `ActorToken` | `create_actor_token`, `revoke_actor_token` |
| `allow_list_block_list_api` | `AllowListBlockList` | `create_allowlist_identifier`, `delete_allowlist_identifier`, `list_allowlist_identifiers`, `create_blocklist_identifier`, `delete_blocklist_identifier`, `list_blocklist_identifiers` |
| `emails_api` | `Email` | `create_email` |
| `jwks_api` | `Jwks` | `get_jwks` |
| `instance_settings_api` | `InstanceSettings` | `update_instance`, `update_instance_organization_settings`, `update_instance_restrictions` |
| `redirect_urls_api` | `RedirectUrl` | `create_redirect_url`, `get_redirect_url`, `delete_redirect_url`, `list_redirect_urls` |
| `beta_features_api` | `BetaFeatures` | `update_instance_auth_config`, `update_production_instance_domain` |

Each method returns `Result<ModelType, Error<MethodErrorEnum>>`. Error enums map HTTP status codes (400, 401, 404, 422, 500) to specific model types.

## Endpoint Enums

Static endpoints (no path parameters):

- `ClerkGetEndpoint`: `ListAllowlistIdentifiers`, `ListBlocklistIdentifiers`, `GetClientList`, `ListInvitations`, `ListJwtTemplates`, `GetPublicInterstitial`, `ListOrganizations`, `ListRedirectUrls`, `GetSessionList`, `GetUserList`, `GetUsersCount`
- `ClerkPostEndpoint`: `CreateActorToken`, `CreateAllowlistIdentifier`, `CreateBlocklistIdentifier`, `VerifyClient`, `CreateEmailAddress`, `CreateEmail`, `CreateInvitation`, `CreateJwtTemplate`, `CreateOrganization`, `CreatePhoneNumber`, `CreateRedirectUrl`, `CreateSignInToken`, `CreateUser`, `CreateSvixApp`, `GenerateSvixAuthUrl`
- `ClerkDeleteEndpoint`: `DeleteSvixApp`
- `ClerkPatchEndpoint`: `UpdateInstanceAuthConfig`, `UpdateInstance`, `UpdateInstanceOrganizationSettings`, `UpdateInstanceRestrictions`

Dynamic endpoints (with `{param}` placeholders, use `*_with_params` methods):

- `ClerkDynamicGetEndpoint`: `GetUser`, `GetOrganization`, `GetSession`, `GetClient`, `GetEmailAddress`, `GetPhoneNumber`, `GetJwks`, `GetJwtTemplate`, `GetRedirectUrl`, `GetSignUp`, `GetTemplate`, `GetTemplateList`, `GetOAuthAccessToken`, `ListOrganizationMemberships`, `ListPendingOrganizationInvitations`, `GetClientLastActiveSession`, `UsersGetOrganizationMemberships`
- `ClerkDynamicPostEndpoint`: `BanUser`, `UnbanUser`, `VerifyPassword`, `VerifyTotp`, `RevokeSession`, `VerifySession`, `RevokeActorToken`, `RevokeInvitation`, `RevokeSignInToken`, `CreateOrganizationInvitation`, `RevokeOrganizationInvitation`, `CreateOrganizationMembership`, `CreateSessionTokenFromTemplate`, `DeleteBlocklistIdentifier`, `PreviewTemplate`, `RevertTemplate`
- `ClerkDynamicDeleteEndpoint`: `DeleteUser`, `DeleteOrganization`, `DeleteEmailAddress`, `DeletePhoneNumber`, `DeleteJwtTemplate`, `DeleteAllowlistIdentifier`, `DeleteRedirectUrl`, `DeleteOrganizationMembership`, `DisableMfa`

## JWT Validation & Middleware

### ClerkAuthorizer

```rust
use clerk_rs::validators::authorizer::ClerkAuthorizer;
use clerk_rs::validators::jwks::MemoryCacheJwksProvider;

let jwks_provider = MemoryCacheJwksProvider::new(clerk);
let authorizer = ClerkAuthorizer::new(jwks_provider, true); // true = also check __session cookie

let jwt: ClerkJwt = authorizer.authorize(&request).await?;
```

Authorization flow:
1. Extract token from `Authorization: Bearer <token>` header
2. Fall back to `__session` cookie if `validate_session_cookie` is true
3. Decode JWT header to get `kid`
4. Fetch matching JWK from provider
5. Validate RS256 signature, `exp`, and `nbf`
6. Return `ClerkJwt` claims

### Standalone validation

```rust
use clerk_rs::validators::authorizer::{validate_jwt, validate_jwt_with_key};

// With JwksProvider (async, fetches key by kid)
let jwt = validate_jwt(token, jwks_provider_arc).await?;

// With a specific JwksKey (sync)
let jwt = validate_jwt_with_key(token, &jwks_key)?;
```

## JWKS Providers

```rust
use clerk_rs::validators::jwks::{MemoryCacheJwksProvider, JwksProviderNoCache};

// Cached provider (recommended for production)
let provider = MemoryCacheJwksProvider::new(clerk);

// With custom options
use clerk_rs::validators::jwks::MemoryCacheJwksProviderOptions;
use std::time::Duration;

let options = MemoryCacheJwksProviderOptions {
    cache_lifetime: Duration::from_secs(7200),       // default: 1 hour
    refresh_on_unknown_key: RefreshOnUnknownKey::Ratelimit(Duration::from_secs(300)), // default: 5 min
};
let provider = MemoryCacheJwksProvider::new_with_options(clerk, options);

// No-cache provider (fetches JWKS on every request)
let provider = JwksProviderNoCache::new(clerk);
```

`JwksProvider` trait can be implemented for custom key sources.

## Key Types

### ClerkJwt

```rust
pub struct ClerkJwt {
    pub azp: Option<String>,           // Authorized party
    pub exp: i32,                       // Expiration
    pub iat: i32,                       // Issued at
    pub iss: String,                    // Issuer
    pub nbf: i32,                       // Not before
    pub sid: Option<String>,            // Session ID
    pub sub: String,                    // Subject (user ID)
    pub act: Option<Actor>,             // Actor (for impersonation)
    pub org: Option<ActiveOrganization>, // Active org (flattened)
    pub other: Map<String, Value>,      // Custom template fields (flattened)
}
```

### ActiveOrganization

```rust
pub struct ActiveOrganization {
    pub id: String,           // org_id
    pub slug: String,         // org_slug
    pub role: String,         // org_role
    pub permissions: Vec<String>, // org_permissions
}

impl ActiveOrganization {
    pub fn has_permission(&self, permission: &str) -> bool;
    pub fn has_role(&self, role: &str) -> bool; // prefer has_permission
}
```

### ClerkError

```rust
pub enum ClerkError {
    Unauthorized(String),
    InternalServerError(String),
}
```

## Feature Flags

```toml
[dependencies]
clerk-rs = "0.4"                          # default: rustls-tls
clerk-rs = { version = "0.4", features = ["axum"] }
clerk-rs = { version = "0.4", features = ["actix"] }
clerk-rs = { version = "0.4", features = ["rocket"] }
clerk-rs = { version = "0.4", features = ["poem"] }
clerk-rs = { version = "0.4", default-features = false, features = ["native-tls"] }
```

## Framework Middleware

### Axum (feature = "axum")

```rust
use clerk_rs::validators::axum::ClerkLayer;
use clerk_rs::validators::jwks::MemoryCacheJwksProvider;

let app = Router::new()
    .route("/protected", get(handler))
    .layer(ClerkLayer::new(
        MemoryCacheJwksProvider::new(clerk),
        None,    // Option<Vec<String>> - excluded routes
        true,    // validate_session_cookie
    ));

// Access JWT in handler via Extension:
async fn handler(Extension(jwt): Extension<ClerkJwt>) -> impl IntoResponse { ... }
```

### Actix-web (feature = "actix")

```rust
use clerk_rs::validators::actix::ClerkMiddleware;

App::new()
    .wrap(ClerkMiddleware::new(
        MemoryCacheJwksProvider::new(clerk),
        None,    // excluded routes
        true,    // validate_session_cookie
    ))
    .route("/protected", web::get().to(handler));
```

### Rocket (feature = "rocket")

```rust
use clerk_rs::validators::rocket::{ClerkGuard, ClerkGuardConfig};

// Register config as managed state
let clerk_config = ClerkGuardConfig::new(
    MemoryCacheJwksProvider::new(clerk),
    None,    // excluded routes
    true,    // validate_session_cookie
);
rocket::build().manage(clerk_config).mount("/", routes![handler]);

// Use as request guard parameter
#[get("/")]
fn handler(jwt: ClerkGuard<MemoryCacheJwksProvider>) -> &'static str { "OK" }
```

### Poem (feature = "poem")

```rust
use clerk_rs::validators::poem::ClerkPoemMiddleware;

let middleware = ClerkPoemMiddleware::new(
    MemoryCacheJwksProvider::new(clerk),
    true,    // validate_session_cookie
    Some(vec!["/public".to_owned()]), // excluded routes
);

let app = Route::new()
    .at("/protected", get(handler))
    .with(middleware);

// Access JWT via Data extractor:
#[handler]
fn handler(jwt: Data<&ClerkJwt>) -> String { ... }
```
