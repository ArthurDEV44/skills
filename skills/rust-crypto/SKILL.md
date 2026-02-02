---
name: rust-crypto
description: >
  Rust cryptography with RustCrypto crates: AES-GCM authenticated encryption (aes-gcm) and
  HMAC message authentication (hmac). Covers key generation, encryption/decryption, in-place
  operations, MAC computation and verification, Cargo.toml dependencies, and feature flags.
  Use when writing, reviewing, or refactoring Rust cryptographic code:
  (1) Encrypting/decrypting data with AES-128-GCM or AES-256-GCM,
  (2) Generating cryptographic keys and nonces,
  (3) In-place encryption without heap allocation (no_std / embedded),
  (4) Computing or verifying HMAC tags (HMAC-SHA256, HMAC-SHA512, etc.),
  (5) Adding aes-gcm or hmac dependencies to Cargo.toml,
  (6) Choosing between Hmac and SimpleHmac for different hash functions,
  (7) Working with associated data (AAD) in AEAD,
  (8) Handling authentication tag verification and errors.
---

# Rust Crypto

## Cargo.toml

### Stable (current published releases)

```toml
[dependencies]
aes-gcm = "0.10"    # AES-GCM AEAD encryption
hmac = "0.12"        # HMAC message authentication
sha2 = "0.10"        # SHA-256/SHA-512 for use with HMAC
```

### Release candidates (latest pre-release)

```toml
[dependencies]
aes-gcm = "0.11.0-rc.2"
hmac = "0.13.0-rc.4"
sha2 = "0.11.0-rc.0"     # matches digest 0.11 used by hmac 0.13
```

## AES-GCM -- Authenticated Encryption

For full API details (types, constants, feature flags), see [references/aes-gcm-api.md](references/aes-gcm-api.md).

### Basic encrypt / decrypt (v0.10 stable)

```rust
use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Nonce, Key,
};

// Generate random key and nonce
let key = Aes256Gcm::generate_key(OsRng);
let cipher = Aes256Gcm::new(&key);
let nonce = Aes256Gcm::generate_nonce(&mut OsRng); // 96-bit

// Encrypt
let ciphertext = cipher.encrypt(&nonce, b"plaintext message".as_ref())
    .expect("encryption failure");

// Decrypt
let plaintext = cipher.decrypt(&nonce, ciphertext.as_ref())
    .expect("decryption failure");
```

### Basic encrypt / decrypt (v0.11 RC)

```rust
use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit},
    Aes256Gcm, Nonce, Key,
};

let key = Aes256Gcm::generate_key().expect("key generation");
let cipher = Aes256Gcm::new(&key);
let nonce = Aes256Gcm::generate_nonce().expect("nonce generation");

let ciphertext = cipher.encrypt(&nonce, b"plaintext".as_ref())?;
let plaintext = cipher.decrypt(&nonce, ciphertext.as_ref())?;
```

Key difference: v0.11 `generate_key()` / `generate_nonce()` take no arguments (use internal RNG), return `Result`.

### In-place encryption (no heap allocation)

```rust
use aes_gcm::{
    aead::{AeadCore, AeadInPlace, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use heapless::Vec; // or arrayvec::ArrayVec

let key = Aes256Gcm::generate_key(OsRng);
let cipher = Aes256Gcm::new(&key);
let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

let mut buffer: Vec<u8, 128> = Vec::new(); // needs 16 bytes overhead for tag
buffer.extend_from_slice(b"plaintext message").unwrap();

cipher.encrypt_in_place(&nonce, b"", &mut buffer).expect("encrypt");
cipher.decrypt_in_place(&nonce, b"", &mut buffer).expect("decrypt");
```

- Second argument to `encrypt_in_place` / `decrypt_in_place` is associated data (AAD)
- Buffer must have 16 extra bytes capacity for the authentication tag
- Use `heapless::Vec` or `arrayvec::ArrayVec` for no_std environments

### Key from existing bytes

```rust
use aes_gcm::{Aes256Gcm, Key};

// From a 32-byte array (AES-256)
let key_bytes = [0u8; 32];
let key = Key::<Aes256Gcm>::from_slice(&key_bytes);

// From a 16-byte array (AES-128)
use aes_gcm::Aes128Gcm;
let key_bytes = [0u8; 16];
let key = Key::<Aes128Gcm>::from_slice(&key_bytes);
```

### Critical rules

- **Nonces MUST be unique** per message with the same key -- reuse breaks security
- Standard nonce size is 96 bits (12 bytes)
- Authentication tag is 16 bytes, appended to ciphertext
- NCC Group audited, constant-time implementation

## HMAC -- Message Authentication

For full API details (types, Hmac vs SimpleHmac), see [references/hmac-api.md](references/hmac-api.md).

### Compute HMAC

```rust
use sha2::Sha256;
use hmac::{Hmac, Mac};

type HmacSha256 = Hmac<Sha256>;

let mut mac = HmacSha256::new_from_slice(b"my secret key")
    .expect("HMAC accepts any key size");
mac.update(b"input message");

let result = mac.finalize();
let code_bytes = result.into_bytes(); // GenericArray
```

### Verify HMAC

```rust
use sha2::Sha256;
use hmac::{Hmac, Mac};

type HmacSha256 = Hmac<Sha256>;

let mut mac = HmacSha256::new_from_slice(b"my secret key")
    .expect("HMAC accepts any key size");
mac.update(b"input message");

// Constant-time verification -- returns Result
mac.verify_slice(&expected_bytes).expect("HMAC verification failed");
```

- `verify_slice` performs constant-time comparison (no timing side-channel)
- Returns `Err(MacError)` on mismatch

### Incremental updates

```rust
let mut mac = HmacSha256::new_from_slice(b"key").unwrap();
mac.update(b"part 1");
mac.update(b"part 2");
// equivalent to mac.update(b"part 1part 2")
let result = mac.finalize();
```

### Hmac vs SimpleHmac

| | Hmac | SimpleHmac |
|---|---|---|
| Performance | Block-level API, more efficient | Digest-level, less efficient |
| Compatibility | Requires `EagerHash` (SHA-1, SHA-2, MD5...) | Works with all `Digest` impls including BLAKE2 |
| Use when | Using SHA-family hashes | Using BLAKE2 or other non-eager hashes |

```rust
// For BLAKE2:
use hmac::SimpleHmac;
use blake2::Blake2b512;

type HmacBlake2b = SimpleHmac<Blake2b512>;
```

### Common type aliases

```rust
use hmac::Hmac;
use sha2::{Sha256, Sha512};
use sha1::Sha1;

type HmacSha256 = Hmac<Sha256>;
type HmacSha512 = Hmac<Sha512>;
type HmacSha1 = Hmac<Sha1>;   // legacy only -- prefer SHA-256+
```

## Official Documentation

- [aes-gcm (stable)](https://docs.rs/aes-gcm/latest/aes_gcm/)
- [aes-gcm (0.11 RC)](https://docs.rs/aes-gcm/0.11.0-rc.2/aes_gcm/)
- [hmac (stable)](https://docs.rs/hmac/latest/hmac/)
- [hmac (0.13 RC)](https://docs.rs/hmac/0.13.0-rc.4/hmac/)
- [RustCrypto AEADs repo](https://github.com/RustCrypto/AEADs)
- [RustCrypto MACs repo](https://github.com/RustCrypto/MACs)
