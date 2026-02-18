# ChaCha20Poly1305 API Reference

## Table of Contents

- [Type Aliases](#type-aliases)
- [Structs](#structs)
- [Traits](#traits)
- [Variants](#variants)
- [Feature Flags](#feature-flags)
- [Security Notes](#security-notes)

## Type Aliases

### Main Variants

| Type | Key Size | Nonce Size | Description |
|------|----------|------------|-------------|
| `ChaCha20Poly1305` | 256-bit | 96-bit (12 bytes) | Standard 20-round (RFC 8439) |
| `XChaCha20Poly1305` | 256-bit | 192-bit (24 bytes) | Extended nonce variant |
| `ChaCha8Poly1305` | 256-bit | 96-bit | Reduced 8-round (feature-gated) |
| `ChaCha12Poly1305` | 256-bit | 96-bit | Reduced 12-round (feature-gated) |
| `XChaCha8Poly1305` | 256-bit | 192-bit | Extended nonce, 8-round |
| `XChaCha12Poly1305` | 256-bit | 192-bit | Extended nonce, 12-round |

### Supporting Types

- **Key** -- 256-bit (32-byte) key
- **Nonce** -- 96-bit (12-byte) nonce
- **XNonce** -- 192-bit (24-byte) nonce (for XChaCha variants)
- **Tag** -- Poly1305 authentication tag (16 bytes)

## Structs

### ChaChaPoly1305

Generic ChaCha+Poly1305 AEAD construction parameterized over stream cipher variant and nonce size.

```rust
// Usually accessed via type aliases:
let cipher = ChaCha20Poly1305::new(&key);
let cipher = XChaCha20Poly1305::new(&key);
```

### Error

Opaque error type for encrypt/decrypt failures. Does not leak failure details.

## Traits

Same trait hierarchy as AES-GCM (shared `aead` crate):

### AeadCore

- `generate_key(&mut OsRng)` -- random 256-bit key
- `generate_nonce(&mut OsRng)` -- random nonce (96-bit or 192-bit depending on variant)

### Aead

- `encrypt(nonce, plaintext)` -- returns ciphertext + 16-byte tag
- `decrypt(nonce, ciphertext)` -- verifies tag, returns plaintext

### AeadInPlace

- `encrypt_in_place(nonce, aad, buffer)` -- in-place encryption with AAD
- `decrypt_in_place(nonce, aad, buffer)` -- in-place decryption with AAD

### KeyInit

- `new(key)` -- from `Key` type
- `new_from_slice(bytes)` -- from 32-byte slice

## Variants

### ChaCha20Poly1305 (standard)

- RFC 8439 compliant
- 96-bit nonce -- safe for ~2^32 messages with random nonces
- Mandatory in TLS 1.3

### XChaCha20Poly1305 (extended nonce)

- 192-bit nonce -- negligible collision risk even with 2^64+ random nonces
- Preferred when using random nonces
- Built on HChaCha20 + ChaCha20Poly1305

### ChaCha8/12Poly1305 (reduced round)

- Feature-gated: `features = ["reduced-round"]`
- Lower security margin but faster
- Use only when performance is critical and threat model accepts reduced rounds

## Feature Flags

| Feature | Description |
|---------|-------------|
| `alloc` | Heap allocation support (default) |
| `heapless` | `aead::Buffer` impl for `heapless::Vec` |
| `reduced-round` | Enable ChaCha8/12 and XChaCha8/12 variants |
| `zeroize` | Zero key material on drop |

### no_std usage

```toml
[dependencies]
chacha20poly1305 = { version = "0.10", default-features = false, features = ["heapless"] }
```

## Dependencies

- `aead ^0.5`, `chacha20 ^0.9`, `cipher ^0.4`, `poly1305 ^0.8`, `zeroize ^1.5`

## Security Notes

- NCC Group audited, no significant findings
- Constant-time implementation (no timing side-channels)
- Fast in pure software without hardware acceleration
- Nonces MUST be unique per key -- reuse breaks security
- For random nonces: prefer XChaCha20Poly1305 (192-bit nonce avoids birthday bound)
- License: Apache-2.0 OR MIT
