# HMAC API Reference

## Table of Contents

- [Types](#types)
- [Traits](#traits)
- [Hmac vs SimpleHmac](#hmac-vs-simplehmac)
- [Reset Variants](#reset-variants)
- [Dependencies](#dependencies)

## Types

### Hmac (type alias)

Buffered wrapper around `HmacCore`. Works with hash functions that implement `EagerHash` (block-level API). Most SHA-family hashes qualify.

```rust
use hmac::Hmac;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;
```

### HmacCore

Generic core HMAC instance operating over blocks. Not typically used directly -- access through the `Hmac` type alias.

### SimpleHmac

Alternative HMAC implementation using the `Digest` trait instead of block-level API. Less memory-efficient but compatible with all hash functions including BLAKE2.

```rust
use hmac::SimpleHmac;
use blake2::Blake2b512;

type HmacBlake2 = SimpleHmac<Blake2b512>;
```

## Traits

### Mac

Convenience wrapper trait covering MAC functionality:
- `new(key)` -- create from `Key` type
- `new_from_slice(bytes)` -- create from byte slice (any length)
- `update(data)` -- feed data incrementally
- `finalize()` -- produce the MAC tag (consumes self)
- `verify_slice(tag)` -- constant-time tag verification
- `verify(tag)` -- verify against `CtOutput`

### KeyInit

Key-based initialization:
- `new(key)` -- from fixed-size key
- `new_from_slice(bytes)` -- from variable-length byte slice

HMAC accepts keys of any length. Keys shorter than the hash block size are zero-padded. Keys longer than the block size are first hashed.

### EagerHash

Implemented by hash functions that expose block-level core (SHA-1, SHA-2, MD5, etc.). Required by `Hmac` but not by `SimpleHmac`.

## Hmac vs SimpleHmac

| | Hmac | SimpleHmac |
|---|---|---|
| Internal API | Block-level (`EagerHash`) | Digest-level (`Digest`) |
| Memory | More efficient (processes blocks directly) | Less efficient (extra buffering) |
| SHA-1, SHA-2, MD5 | Yes | Yes |
| BLAKE2 | No (not `EagerHash`) | Yes |
| Streebog | No | Yes |

**Rule of thumb**: Use `Hmac` unless your hash function doesn't compile with it, then switch to `SimpleHmac`.

## Reset Variants

- **HmacReset** / **SimpleHmacReset** -- versions that support `reset()` to reuse the instance with the same key for multiple messages

```rust
use hmac::{HmacReset, Mac};
use sha2::Sha256;

type HmacSha256 = HmacReset<Sha256>;

// v0.13 RC
let mut mac = HmacSha256::new_from_slice(b"key").unwrap();
mac.update(b"message 1");
let tag1 = mac.finalize_reset(); // resets internal state

mac.update(b"message 2");
let tag2 = mac.finalize();
```

## Dependencies

### Stable (hmac 0.12)

```toml
[dependencies]
hmac = "0.12"
sha2 = "0.10"     # or sha1 = "0.10", md-5 = "0.10"
```

Requires `digest ^0.10`.

### RC (hmac 0.13)

```toml
[dependencies]
hmac = "0.13.0-rc.4"
sha2 = "0.11.0-rc.0"
```

Requires `digest ^0.11`.

- License: Apache-2.0 OR MIT
