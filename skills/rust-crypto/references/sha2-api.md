# SHA-2 API Reference

## Table of Contents

- [Type Aliases](#type-aliases)
- [Structs](#structs)
- [Traits](#traits)
- [Functions](#functions)
- [Dependencies](#dependencies)

## Type Aliases

| Type | Output Size | Word Size | Description |
|------|-------------|-----------|-------------|
| `Sha224` | 28 bytes | 32-bit | Truncated SHA-256 |
| `Sha256` | 32 bytes | 32-bit | Most common variant |
| `Sha384` | 48 bytes | 64-bit | Truncated SHA-512 |
| `Sha512` | 64 bytes | 64-bit | Largest standard variant |
| `Sha512_224` | 28 bytes | 64-bit | SHA-512 with 224-bit output |
| `Sha512_256` | 32 bytes | 64-bit | SHA-512 with 256-bit output |

## Structs

### Sha256VarCore / Sha512VarCore

Block-level SHA-256/SHA-512 hasher with variable output size. Not typically used directly -- access through the type aliases above.

## Traits

### Digest

Convenience wrapper for cryptographic hash functions with fixed output:
- `new()` -- create hasher instance
- `update(data)` -- feed data incrementally
- `finalize(self)` -- produce hash (consumes self)
- `finalize_reset(&mut self)` -- produce hash and reset for reuse
- `digest(data)` -- one-shot: create, update, finalize

```rust
use sha2::{Sha256, Digest};

// One-shot
let hash = Sha256::digest(b"data");

// Incremental
let mut hasher = Sha256::new();
hasher.update(b"part1");
hasher.update(b"part2");
let hash = hasher.finalize();
```

### Output

`finalize()` returns `GenericArray<u8, Self::OutputSize>` which implements:
- `AsRef<[u8]>` -- access raw bytes
- `fmt::LowerHex` -- hex-encode via `format!("{:x}", hash)`

## Functions

- `compress256(state, blocks)` -- raw SHA-256 compression (low-level)
- `compress512(state, blocks)` -- raw SHA-512 compression (low-level)

## Dependencies

### Stable (sha2 0.10)

```toml
[dependencies]
sha2 = "0.10"
```

Requires `digest ^0.10`.

### RC (sha2 0.11)

```toml
[dependencies]
sha2 = "0.11.0-rc.0"
```

Requires `digest ^0.11`.

### Optional

- `sha2-asm ^0.6.1` -- assembly-optimized compression functions (feature: `asm`)
- `cpufeatures ^0.2` -- runtime CPU feature detection

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Standard library support (default) |
| `oid` | OID (Object Identifier) support for `const_oid` |
| `asm` | Use `sha2-asm` for assembly-optimized compression |
| `compress` | Expose raw compression functions |
| `force-soft` | Force software implementation (disable hardware accel) |
| `loongarch64_asm` | LoongArch64 assembly support |

- License: Apache-2.0 OR MIT
