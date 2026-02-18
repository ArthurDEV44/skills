---
name: rust-crypto
description: >
  Rust cryptography with RustCrypto crates: AES-GCM and ChaCha20Poly1305 authenticated encryption,
  HMAC message authentication, SHA-2 hashing, HKDF key derivation, Argon2 password hashing,
  Ed25519 digital signatures, and X25519 key exchange. Covers key generation, encryption/decryption,
  in-place operations, MAC computation, password hashing and verification, signing and verification,
  Diffie-Hellman key agreement, Cargo.toml dependencies, and feature flags.
  Use when writing, reviewing, or refactoring Rust cryptographic code:
  (1) Encrypting/decrypting data with AES-256-GCM, AES-128-GCM, or ChaCha20Poly1305,
  (2) Generating cryptographic keys and nonces,
  (3) In-place encryption without heap allocation (no_std / embedded),
  (4) Computing or verifying HMAC tags (HMAC-SHA256, HMAC-SHA512),
  (5) Hashing data with SHA-256 or SHA-512,
  (6) Deriving keys with HKDF (extract-and-expand),
  (7) Hashing and verifying passwords with Argon2id,
  (8) Signing and verifying messages with Ed25519,
  (9) Performing X25519 Diffie-Hellman key exchange,
  (10) Adding RustCrypto dependencies to Cargo.toml,
  (11) Choosing between AEAD ciphers (AES-GCM vs ChaCha20Poly1305),
  (12) Working with associated data (AAD) in AEAD.
---

# Rust Crypto

## Cargo.toml

### Stable (current published releases)

```toml
[dependencies]
aes-gcm = "0.10"              # AES-GCM AEAD encryption
chacha20poly1305 = "0.10"     # ChaCha20Poly1305 AEAD encryption
hmac = "0.12"                  # HMAC message authentication
sha2 = "0.10"                  # SHA-256/SHA-512 hashing
hkdf = "0.12"                  # HKDF key derivation (RFC 5869)
argon2 = "0.5"                 # Argon2 password hashing
ed25519-dalek = "2"            # Ed25519 digital signatures
x25519-dalek = "2"             # X25519 Diffie-Hellman key exchange
```

### Release candidates (latest pre-release)

```toml
[dependencies]
aes-gcm = "0.11.0-rc.2"
chacha20poly1305 = "0.11.0-rc.1"
hmac = "0.13.0-rc.4"
sha2 = "0.11.0-rc.0"          # matches digest 0.11 used by hmac 0.13
```

## Choosing an AEAD Cipher

| | AES-GCM | ChaCha20Poly1305 |
|---|---|---|
| Key size | 128 or 256 bits | 256 bits |
| Nonce size | 96 bits (12 bytes) | 96 bits (12 bytes) |
| Hardware accel | AES-NI + CLMUL (x86/x86_64, ARMv8) | None needed |
| Software perf | Slow without HW accel | Fast in pure software |
| Use when | Server-side, x86 with AES-NI | Mobile, embedded, cross-platform |
| Extended nonce | Not available | XChaCha20Poly1305 (192-bit nonce) |

**Rule of thumb**: Use AES-256-GCM on servers with AES-NI. Use ChaCha20Poly1305 (or XChaCha20Poly1305 for random nonces) everywhere else.

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
use aes_gcm::{Aes256Gcm, Aes128Gcm, Key};

// From a 32-byte array (AES-256)
let key_bytes = [0u8; 32];
let key = Key::<Aes256Gcm>::from_slice(&key_bytes);

// From a 16-byte array (AES-128)
let key_bytes = [0u8; 16];
let key = Key::<Aes128Gcm>::from_slice(&key_bytes);
```

### Critical rules

- **Nonces MUST be unique** per message with the same key -- reuse breaks security
- Standard nonce size is 96 bits (12 bytes)
- Authentication tag is 16 bytes, appended to ciphertext
- ~2^32 messages safe per key with random nonces (birthday bound)
- NCC Group audited, constant-time implementation

## ChaCha20Poly1305 -- Authenticated Encryption

For full API details (variants, feature flags), see [references/chacha20poly1305-api.md](references/chacha20poly1305-api.md).

### Basic encrypt / decrypt

```rust
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce,
};

let key = ChaCha20Poly1305::generate_key(&mut OsRng);
let cipher = ChaCha20Poly1305::new(&key);
let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng); // 96-bit

let ciphertext = cipher.encrypt(&nonce, b"plaintext message".as_ref())
    .expect("encryption failure");
let plaintext = cipher.decrypt(&nonce, ciphertext.as_ref())
    .expect("decryption failure");
```

### XChaCha20Poly1305 (extended 192-bit nonce)

```rust
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    XChaCha20Poly1305, XNonce,
};

let key = XChaCha20Poly1305::generate_key(&mut OsRng);
let cipher = XChaCha20Poly1305::new(&key);
let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng); // 192-bit

let ciphertext = cipher.encrypt(&nonce, b"plaintext".as_ref())?;
```

Use XChaCha20Poly1305 when generating random nonces -- the 192-bit nonce makes collisions negligible even over 2^64 messages.

### In-place encryption

```rust
use chacha20poly1305::{
    aead::{AeadCore, AeadInPlace, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce,
};
use heapless::Vec;

let key = ChaCha20Poly1305::generate_key(&mut OsRng);
let cipher = ChaCha20Poly1305::new(&key);
let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);

let mut buffer: Vec<u8, 128> = Vec::new();
buffer.extend_from_slice(b"plaintext message").unwrap();

cipher.encrypt_in_place(&nonce, b"", &mut buffer)?;
cipher.decrypt_in_place(&nonce, b"", &mut buffer)?;
```

## SHA-2 -- Cryptographic Hashing

For full API details, see [references/sha2-api.md](references/sha2-api.md).

### One-shot hashing

```rust
use sha2::{Sha256, Digest};

let hash = Sha256::digest(b"hello world");
println!("{:x}", hash); // hex-encoded
```

### Incremental hashing

```rust
use sha2::{Sha256, Digest};

let mut hasher = Sha256::new();
hasher.update(b"hello ");
hasher.update(b"world");
let result = hasher.finalize(); // equivalent to Sha256::digest(b"hello world")
```

### Available variants

- `Sha256` / `Sha224` -- 32-bit word based
- `Sha512` / `Sha384` / `Sha512_256` / `Sha512_224` -- 64-bit word based

## HMAC -- Message Authentication

For full API details (Hmac vs SimpleHmac), see [references/hmac-api.md](references/hmac-api.md).

### Compute and verify HMAC

```rust
use sha2::Sha256;
use hmac::{Hmac, Mac};

type HmacSha256 = Hmac<Sha256>;

// Compute
let mut mac = HmacSha256::new_from_slice(b"my secret key")
    .expect("HMAC accepts any key size");
mac.update(b"input message");
let result = mac.finalize();
let code_bytes = result.into_bytes(); // GenericArray

// Verify (constant-time comparison)
let mut mac = HmacSha256::new_from_slice(b"my secret key").unwrap();
mac.update(b"input message");
mac.verify_slice(&code_bytes).expect("HMAC verification failed");
```

### Hmac vs SimpleHmac

| | Hmac | SimpleHmac |
|---|---|---|
| Performance | Block-level API, more efficient | Digest-level, less efficient |
| Compatibility | Requires `EagerHash` (SHA-1, SHA-2, MD5...) | Works with all `Digest` impls including BLAKE2 |
| Use when | Using SHA-family hashes | Using BLAKE2 or other non-eager hashes |

## HKDF -- Key Derivation

For full API details, see [references/hkdf-api.md](references/hkdf-api.md).

### Extract-and-expand (RFC 5869)

```rust
use hkdf::Hkdf;
use sha2::Sha256;

let ikm = b"input keying material";
let salt = b"optional salt value";
let info = b"context and application specific info";

let hk = Hkdf::<Sha256>::new(Some(salt), ikm);
let mut okm = [0u8; 32]; // output keying material
hk.expand(info, &mut okm).expect("expand failed");
```

### Extract only (for use with multiple expand calls)

```rust
use hkdf::Hkdf;
use sha2::Sha256;

let (prk, hk) = Hkdf::<Sha256>::extract(Some(b"salt"), b"ikm");
// prk is the pseudorandom key
// hk can be used for multiple expand calls with different info

let mut key1 = [0u8; 32];
hk.expand(b"key1", &mut key1).unwrap();

let mut key2 = [0u8; 32];
hk.expand(b"key2", &mut key2).unwrap();
```

### From existing PRK

```rust
let hk = Hkdf::<Sha256>::from_prk(prk_bytes).expect("PRK too short");
let mut okm = [0u8; 32];
hk.expand(b"info", &mut okm).unwrap();
```

## Argon2 -- Password Hashing

For full API details (params, variants), see [references/argon2-api.md](references/argon2-api.md).

### Hash and verify password

```rust
use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng},
};

// Hash
let salt = SaltString::generate(&mut OsRng);
let argon2 = Argon2::default(); // Argon2id v19
let password_hash = argon2.hash_password(b"hunter2", &salt)
    .expect("hashing failed")
    .to_string(); // PHC string format: $argon2id$v=19$m=19456,t=2,p=1$...

// Verify
let parsed_hash = PasswordHash::new(&password_hash).unwrap();
Argon2::default()
    .verify_password(b"hunter2", &parsed_hash)
    .expect("invalid password");
```

### Key derivation (raw output, no PHC string)

```rust
use argon2::Argon2;

let mut output_key = [0u8; 32];
Argon2::default()
    .hash_password_into(b"password", b"somesaltvalue!!!", &mut output_key)
    .expect("hashing failed");
```

Salt must be at least 8 bytes for `hash_password_into`.

### OWASP recommended parameters

- **Argon2id** (default): minimum 19 MiB memory (`m=19456`), 2 iterations (`t=2`), 1 parallelism (`p=1`)
- **PBKDF2**: minimum 600,000 iterations with SHA-256 (FIPS-140 compliance)
- **Scrypt**: `N=2^17`, `r=8`, `p=1`

## Ed25519 -- Digital Signatures

For full API details (serialization, PKCS#8, batch), see [references/ed25519-api.md](references/ed25519-api.md).

### Sign and verify

```rust
use ed25519_dalek::{SigningKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;

// Generate keypair
let signing_key = SigningKey::generate(&mut OsRng);
let verifying_key = signing_key.verifying_key();

// Sign
let message = b"important message";
let signature: Signature = signing_key.sign(message);

// Verify
verifying_key.verify(message, &signature).expect("invalid signature");
```

### Serialize / deserialize keys

```rust
use ed25519_dalek::{SigningKey, VerifyingKey};

// To bytes
let sk_bytes: [u8; 32] = signing_key.to_bytes();
let vk_bytes: [u8; 32] = verifying_key.to_bytes();

// From bytes
let signing_key = SigningKey::from_bytes(&sk_bytes);
let verifying_key = VerifyingKey::from_bytes(&vk_bytes)?;
```

## X25519 -- Key Exchange

For full API details, see [references/x25519-api.md](references/x25519-api.md).

### Diffie-Hellman key agreement

```rust
use x25519_dalek::{EphemeralSecret, PublicKey};
use rand::rngs::OsRng;

// Alice
let alice_secret = EphemeralSecret::random_from_rng(OsRng);
let alice_public = PublicKey::from(&alice_secret);

// Bob
let bob_secret = EphemeralSecret::random_from_rng(OsRng);
let bob_public = PublicKey::from(&bob_secret);

// Both compute the same shared secret
let alice_shared = alice_secret.diffie_hellman(&bob_public);
let bob_shared = bob_secret.diffie_hellman(&alice_public);
assert_eq!(alice_shared.as_bytes(), bob_shared.as_bytes());
```

### Combine with HKDF for encryption key

```rust
use x25519_dalek::{EphemeralSecret, PublicKey};
use hkdf::Hkdf;
use sha2::Sha256;
use rand::rngs::OsRng;

let secret = EphemeralSecret::random_from_rng(OsRng);
let public = PublicKey::from(&secret);

// After exchanging public keys...
let shared = secret.diffie_hellman(&peer_public);

// ALWAYS derive a proper key -- never use raw shared secret directly
let hk = Hkdf::<Sha256>::new(None, shared.as_bytes());
let mut encryption_key = [0u8; 32];
hk.expand(b"encryption", &mut encryption_key).unwrap();
// Use encryption_key with AES-GCM or ChaCha20Poly1305
```

## Common Patterns

### Encrypt-then-MAC (when not using AEAD)

Prefer AEAD ciphers (AES-GCM, ChaCha20Poly1305) over manual encrypt-then-MAC. AEAD handles authentication internally and is harder to misuse.

### Nonce management strategies

1. **Counter-based**: Maintain a persistent counter (safe if counter never resets)
2. **Random nonces**: Use `generate_nonce()` with OsRng (safe for ~2^32 messages with 96-bit nonce)
3. **Extended nonces**: Use XChaCha20Poly1305 (192-bit) for negligible collision risk even with random nonces over 2^64 messages

### Zeroize sensitive material

```toml
[dependencies]
aes-gcm = { version = "0.10", features = ["zeroize"] }
zeroize = "1"
```

```rust
use zeroize::Zeroize;

let mut secret = vec![0u8; 32];
// ... use secret ...
secret.zeroize(); // overwrite memory on drop
```

## Official Documentation

- [aes-gcm (stable)](https://docs.rs/aes-gcm/latest/aes_gcm/)
- [aes-gcm (0.11 RC)](https://docs.rs/aes-gcm/0.11.0-rc.2/aes_gcm/)
- [chacha20poly1305](https://docs.rs/chacha20poly1305/latest/chacha20poly1305/)
- [hmac (stable)](https://docs.rs/hmac/latest/hmac/)
- [sha2](https://docs.rs/sha2/latest/sha2/)
- [hkdf](https://docs.rs/hkdf/latest/hkdf/)
- [argon2](https://docs.rs/argon2/latest/argon2/)
- [ed25519-dalek](https://docs.rs/ed25519-dalek/latest/ed25519_dalek/)
- [x25519-dalek](https://docs.rs/x25519-dalek/latest/x25519_dalek/)
- [RustCrypto Book](https://rustcrypto.org/)
- [RustCrypto AEADs repo](https://github.com/RustCrypto/AEADs)
- [RustCrypto MACs repo](https://github.com/RustCrypto/MACs)
- [RustCrypto password-hashes repo](https://github.com/RustCrypto/password-hashes)
