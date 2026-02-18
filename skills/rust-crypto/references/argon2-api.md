# Argon2 API Reference

## Table of Contents

- [Overview](#overview)
- [Structs](#structs)
- [Enums](#enums)
- [Traits](#traits)
- [Constants](#constants)
- [Feature Flags](#feature-flags)
- [Usage Patterns](#usage-patterns)
- [Dependencies](#dependencies)

## Overview

Argon2 is a memory-hard key derivation function and the winner of the 2015 Password Hashing Competition (PHC). The crate (v0.5) provides Argon2d, Argon2i, and Argon2id variants with PHC string format support.

## Structs

### Argon2

Main context for hashing. Holds algorithm parameters.

```rust
use argon2::Argon2;

// Default: Argon2id v19, m=19456 (19 MiB), t=2, p=1
let argon2 = Argon2::default();

// Custom params
let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
```

### Params

Configuration parameters for Argon2.

```rust
use argon2::Params;

let params = Params::new(65536, 3, 4, Some(32)).unwrap();
// m_cost=65536 (64 MiB), t_cost=3 (iterations), p_cost=4 (parallelism), output_len=32
```

### ParamsBuilder

Builder pattern for constructing Params.

```rust
use argon2::ParamsBuilder;

let params = ParamsBuilder::new()
    .m_cost(65536)    // memory in KiB
    .t_cost(3)        // iterations
    .p_cost(4)        // parallelism
    .output_len(32)   // output length in bytes
    .build()
    .unwrap();
```

### SaltString (from password-hash)

Random salt generation and PHC-compatible encoding.

```rust
use argon2::password_hash::{SaltString, rand_core::OsRng};

let salt = SaltString::generate(&mut OsRng);
```

### PasswordHash (from password-hash)

Parsed PHC string format representation.

```rust
use argon2::password_hash::PasswordHash;

// Parse from stored string
let hash = PasswordHash::new("$argon2id$v=19$m=19456,t=2,p=1$...").unwrap();
```

## Enums

### Algorithm

- `Argon2d` -- maximizes resistance to GPU cracking attacks
- `Argon2i` -- optimized against side-channel attacks
- `Argon2id` -- hybrid (recommended default) -- first half Argon2i, second half Argon2d

### Version

- `V0x10` -- version 1.0
- `V0x13` -- version 1.3 (current, default)

### Error

Error conditions: AdTooLong, AlgorithmInvalid, KeyIdTooLong, MemoryTooLittle, OutputTooShort, OutputTooLong, PwdTooLong, SaltTooShort, SaltTooLong, SecretTooLong, TimeTooSmall, ThreadsTooFew, ThreadsTooMany.

## Traits

### PasswordHasher

```rust
use argon2::{Argon2, password_hash::PasswordHasher};

let hash = Argon2::default().hash_password(password, &salt)?;
// Returns PasswordHash with PHC string format
```

### PasswordVerifier

```rust
use argon2::{Argon2, password_hash::PasswordVerifier};

Argon2::default().verify_password(password, &parsed_hash)?;
// Returns Ok(()) or Err(password_hash::Error)
```

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MIN_SALT_LEN` | 8 | Minimum salt bytes |
| `RECOMMENDED_SALT_LEN` | 16 | Recommended salt bytes |
| `MAX_PWD_LEN` | 0xFFFFFFFF | Max password length |
| `MAX_SALT_LEN` | 0xFFFFFFFF | Max salt length |
| `MAX_SECRET_LEN` | 0xFFFFFFFF | Max secret (pepper) length |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `alloc` | Heap allocation support (default) |
| `password-hash` | PHC string format support via `password-hash` crate (default) |
| `std` | Standard library support |
| `zeroize` | Zero memory on drop |
| `parallel` | Multi-threaded hashing via `rayon` |

### no_std usage

```toml
[dependencies]
argon2 = { version = "0.5", default-features = false, features = ["alloc"] }
```

## Usage Patterns

### Password storage (PHC string)

```rust
use argon2::{Argon2, password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng}};

// Hash
let salt = SaltString::generate(&mut OsRng);
let hash_string = Argon2::default()
    .hash_password(b"password", &salt)?
    .to_string();
// Store hash_string in database

// Verify
let parsed = PasswordHash::new(&hash_string)?;
Argon2::default().verify_password(b"password", &parsed)?;
```

### Raw key derivation (no PHC string)

```rust
use argon2::Argon2;

let mut key = [0u8; 32];
Argon2::default().hash_password_into(b"password", b"salt-at-least-8B", &mut key)?;
```

### With secret (pepper)

```rust
use argon2::{Argon2, Algorithm, Version, Params};

let params = Params::new(19456, 2, 1, Some(32)).unwrap();
let argon2 = Argon2::new_with_secret(
    b"server-side-pepper",
    Algorithm::Argon2id,
    Version::V0x13,
    params,
).unwrap();
```

### Multi-algorithm verification (legacy migration)

```rust
use argon2::Argon2;
use password_hash::{PasswordHash, PasswordVerifier};

let hash = PasswordHash::new(&stored_hash)?;
let algs: &[&dyn PasswordVerifier] = &[&Argon2::default()];
hash.verify_password(algs, b"password")?;
```

### OWASP recommended parameters

| Algorithm | Parameters |
|-----------|-----------|
| Argon2id | m=19456 (19 MiB), t=2, p=1 (default) |
| Argon2id (high security) | m=65536 (64 MiB), t=3, p=4 |

## Dependencies

```toml
[dependencies]
argon2 = "0.5"
```

Core: `base64ct ^1`, `blake2 ^0.10`, `password-hash ^0.5` (optional)

- License: Apache-2.0 OR MIT
