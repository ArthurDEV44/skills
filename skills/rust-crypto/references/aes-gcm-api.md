# AES-GCM API Reference

## Table of Contents

- [Type Aliases](#type-aliases)
- [Structs](#structs)
- [Traits](#traits)
- [Constants](#constants)
- [Feature Flags](#feature-flags)
- [Tag Sizes](#tag-sizes)
- [Security Notes](#security-notes)

## Type Aliases

- **Aes128Gcm** -- AES-GCM with 128-bit key, 96-bit nonce
- **Aes256Gcm** -- AES-GCM with 256-bit key, 96-bit nonce
- **Key** -- Key type used by `KeySizeUser` implementors
- **Nonce** -- AES-GCM nonce (standard: 96-bit / 12 bytes)
- **Tag** -- Authentication tag (16 bytes)

## Structs

### AesGcm

Generic AES-GCM implementation parameterized over AES cipher and nonce size.

```rust
// Usually accessed via type aliases:
let cipher = Aes256Gcm::new(&key);
let cipher = Aes128Gcm::new(&key);
```

### Error

Opaque error type returned by encrypt/decrypt operations. Does not leak information about failure reason (security property).

## Traits

### AeadCore

Core AEAD interface providing:
- `generate_key()` -- random key generation
- `generate_nonce()` -- random nonce generation
- Associated types for nonce size, tag size, ciphertext overhead

### Aead (v0.10) / AeadInOut (v0.11)

High-level encrypt/decrypt:
- `encrypt(nonce, plaintext)` -- returns ciphertext with appended tag
- `decrypt(nonce, ciphertext)` -- verifies tag and returns plaintext

### AeadInPlace

In-place operations (no allocation):
- `encrypt_in_place(nonce, aad, buffer)` -- encrypts buffer contents, appends tag
- `decrypt_in_place(nonce, aad, buffer)` -- verifies tag, decrypts buffer contents

### KeyInit

Key initialization:
- `new(key)` -- from `Key` type
- `new_from_slice(bytes)` -- from byte slice (returns `Result`)

### KeySizeUser

Exposes associated `KeySize` type (16 for AES-128, 32 for AES-256).

### TagSize

Valid tag sizes: U12, U13, U14, U15, U16 (bytes). Default is U16 (128-bit tag).

## Constants

- **A_MAX** -- Maximum associated data length
- **C_MAX** -- Maximum ciphertext length
- **P_MAX** -- Maximum plaintext length

## Feature Flags

| Feature | Description |
|---------|-------------|
| `aes` | Include AES implementation (enabled by default) |
| `alloc` | Heap allocation support; disable for no_std embedded |
| `arrayvec` | `aead::Buffer` impl for `arrayvec::ArrayVec` |
| `bytes` | `aead::Buffer` impl for `bytes::BytesMut` |
| `heapless` | `aead::Buffer` impl for `heapless::Vec` |
| `zeroize` | Zero key material on drop |

### no_std usage

```toml
[dependencies]
aes-gcm = { version = "0.10", default-features = false, features = ["aes", "heapless"] }
```

## Security Notes

- NCC Group audited (funded by MobileCoin), no significant findings
- Constant-time via AES-NI + CLMUL on x86/x86_64, portable constant-time on other platforms
- **Not suitable** for processors with variable-time multiplication
- Nonces MUST be unique per key -- reuse completely breaks confidentiality and authenticity
- 96-bit nonces: safe for ~2^32 messages per key with random nonces (birthday bound)
- License: Apache-2.0 OR MIT
