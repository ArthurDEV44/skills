# HKDF API Reference

## Table of Contents

- [Overview](#overview)
- [Structs](#structs)
- [Type Aliases](#type-aliases)
- [Error Types](#error-types)
- [Usage Patterns](#usage-patterns)
- [Dependencies](#dependencies)

## Overview

HKDF (HMAC-based Extract-and-Expand Key Derivation Function) as specified in RFC 5869. Derives one or more cryptographically strong keys from input keying material (IKM).

Two-phase design:
1. **Extract**: Condense IKM into a fixed-length pseudorandom key (PRK) using HMAC
2. **Expand**: Derive output keying material (OKM) from PRK using HMAC with context info

## Structs

### Hkdf

Primary struct for key derivation. Parameterized over hash function.

```rust
use hkdf::Hkdf;
use sha2::Sha256;

// Combined extract-and-expand
let hk = Hkdf::<Sha256>::new(Some(salt), ikm);
let mut okm = [0u8; 32];
hk.expand(info, &mut okm).unwrap();
```

Methods:
- `new(salt, ikm)` -- extract PRK from salt + IKM, store for expand
- `extract(salt, ikm)` -- returns `(GenericArray<PRK>, Hkdf)` tuple
- `from_prk(prk)` -- create from existing PRK (returns `Result`, validates PRK length)
- `expand(info, okm)` -- derive output keying material (returns `Result`)
- `expand_multi_info(infos, okm)` -- expand with multiple info slices concatenated

### HkdfExtract

Streaming extract context for incremental IKM feeding.

```rust
use hkdf::HkdfExtract;
use sha2::Sha256;

let mut extract = HkdfExtract::<Sha256>::new(Some(salt));
extract.input_ikm(ikm_part1);
extract.input_ikm(ikm_part2);
let (prk, hk) = extract.finalize();
```

## Type Aliases

- **SimpleHkdf** -- HKDF variant using `SimpleHmac` (works with BLAKE2 and other non-eager hashes)
- **SimpleHkdfExtract** -- Streaming extract using `SimpleHmac`

```rust
use hkdf::SimpleHkdf;
use blake2::Blake2b512;

let hk = SimpleHkdf::<Blake2b512>::new(Some(salt), ikm);
```

## Error Types

- **InvalidLength** -- output length exceeds 255 * hash_len (RFC 5869 limit)
- **InvalidPrkLength** -- PRK provided to `from_prk` is shorter than hash output

## Usage Patterns

### Derive encryption key from password + salt

```rust
use hkdf::Hkdf;
use sha2::Sha256;

let hk = Hkdf::<Sha256>::new(Some(b"app-salt"), b"user-password");
let mut aes_key = [0u8; 32];
hk.expand(b"aes-256-gcm-key", &mut aes_key).unwrap();
```

### Derive multiple keys from shared secret (e.g., after DH)

```rust
let hk = Hkdf::<Sha256>::new(None, shared_secret);

let mut enc_key = [0u8; 32];
hk.expand(b"encryption", &mut enc_key).unwrap();

let mut mac_key = [0u8; 32];
hk.expand(b"authentication", &mut mac_key).unwrap();
```

### Maximum output length

OKM length must be `<= 255 * HashLen`. For SHA-256 (32 bytes): max 8160 bytes.

## Dependencies

### Stable (hkdf 0.12)

```toml
[dependencies]
hkdf = "0.12"
sha2 = "0.10"     # or any digest 0.10 compatible hash
```

Requires `hmac ^0.12`.

- License: Apache-2.0 OR MIT
