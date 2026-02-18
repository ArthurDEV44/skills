# Standard Library Traits

## Display & Debug

```rust
use std::fmt;

// Debug -- machine-readable, usually derived
#[derive(Debug)]
struct Point { x: f64, y: f64 }

// Display -- human-readable, must be implemented manually
impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
// println!("{}", point);   // Display
// println!("{:?}", point); // Debug
// println!("{:#?}", point); // Pretty Debug
```

Implementing `Display` automatically provides `ToString` via blanket impl.

## From & Into

Infallible conversions. Implement `From`; `Into` is provided automatically.

```rust
struct Email(String);

impl From<String> for Email {
    fn from(s: String) -> Self {
        Email(s)
    }
}

// Both work after implementing From:
let email = Email::from("user@example.com".to_string());
let email: Email = "user@example.com".to_string().into();
```

**TryFrom / TryInto** for fallible conversions:

```rust
impl TryFrom<String> for Email {
    type Error = EmailError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s.contains('@') { Ok(Email(s)) }
        else { Err(EmailError::InvalidFormat) }
    }
}
```

Use `From` in function signatures for flexible APIs:

```rust
fn send_email(to: impl Into<Email>) {
    let email = to.into();
    // ...
}
```

## Default

Provides a sensible default value:

```rust
#[derive(Default)]
struct Config {
    verbose: bool,       // false
    retries: u32,        // 0
    name: String,        // ""
}

let config = Config::default();
let config = Config { verbose: true, ..Default::default() };
```

Custom implementation:

```rust
impl Default for Config {
    fn default() -> Self {
        Self { verbose: false, retries: 3, name: "default".into() }
    }
}
```

## PartialEq, Eq, PartialOrd, Ord

```rust
#[derive(PartialEq)]        // == and !=
#[derive(PartialEq, Eq)]    // + reflexivity guarantee (a == a always true)
#[derive(PartialOrd)]       // <, >, <=, >= (returns Option<Ordering>)
#[derive(PartialOrd, Ord)]  // + total ordering (returns Ordering)
```

**When to use which:**
- `PartialEq` -- most types; NaN != NaN means `f64` is only `PartialEq`
- `Eq` -- add when equality is reflexive (all values equal themselves)
- `PartialOrd` -- when some values are incomparable (e.g., NaN)
- `Ord` -- when every pair of values has a defined order; required for `BTreeMap` keys and `.sort()`

Custom comparison:

```rust
impl PartialEq for CaseInsensitive {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_lowercase() == other.0.to_lowercase()
    }
}
```

## Hash

```rust
#[derive(Hash, PartialEq, Eq)]  // Required for HashMap/HashSet keys
struct Id(u64);
```

Rule: if `a == b`, then `hash(a) == hash(b)`. Always derive `Hash` alongside `PartialEq` + `Eq`.

## AsRef & AsMut

Cheap reference-to-reference conversions:

```rust
fn read_file(path: impl AsRef<Path>) {
    let path = path.as_ref();
    // Works with &str, String, PathBuf, &Path...
}

read_file("data.txt");
read_file(String::from("data.txt"));
read_file(Path::new("data.txt"));
```

## Borrow & ToOwned

`Borrow` is like `AsRef` but with the contract that borrowed and owned forms hash and compare the same:

```rust
use std::borrow::Borrow;

fn find<Q>(map: &HashMap<String, i32>, key: &Q) -> Option<&i32>
where
    String: Borrow<Q>,
    Q: Hash + Eq + ?Sized,
{
    map.get(key)
}
// Works with both &str and &String
```

`ToOwned` is the inverse: `&str` -> `String`, `&[T]` -> `Vec<T>`, `&Path` -> `PathBuf`.

## Send & Sync

Marker traits for thread safety (auto-implemented by the compiler):

| Trait | Meaning | Example types |
|-------|---------|---------------|
| `Send` | Safe to **move** to another thread | Most types; NOT `Rc<T>` |
| `Sync` | Safe to **share references** across threads | Most types; NOT `Cell<T>`, `RefCell<T>` |

```rust
// T: Send -- can be moved to a thread
fn spawn_work<T: Send + 'static>(val: T) {
    std::thread::spawn(move || { /* use val */ });
}

// T: Sync -- &T can be shared
fn share<T: Sync>(val: &T) {
    // safe to send &T to multiple threads
}
```

Key relationships:
- `T: Sync` iff `&T: Send`
- `Arc<T>` is `Send + Sync` when `T: Send + Sync`
- `Mutex<T>` is `Sync` when `T: Send` (makes non-Sync types shareable)

## Sized

`Sized` means the type has a known size at compile time. Most types are `Sized`. All generic parameters have an implicit `T: Sized` bound.

```rust
// Opt out with ?Sized to accept unsized types (trait objects, slices)
fn print_it<T: Display + ?Sized>(val: &T) {
    println!("{val}");
}
print_it("hello");                // T = str (unsized)
print_it(&42);                    // T = i32 (sized)
```

Unsized types (`str`, `[T]`, `dyn Trait`) can only be used behind a pointer (`&`, `Box`, `Arc`).

## Deref & DerefMut

Enable smart pointer dereferencing and deref coercion:

```rust
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &T { &self.0 }
}

// Deref coercion: &MyBox<String> -> &String -> &str
fn greet(name: &str) { println!("Hello, {name}!"); }
greet(&MyBox("world".to_string()));  // works via deref coercion
```

## Sources

- [std::fmt](https://doc.rust-lang.org/std/fmt/index.html)
- [std::convert](https://doc.rust-lang.org/std/convert/index.html)
- [std::default::Default](https://doc.rust-lang.org/std/default/trait.Default.html)
- [std::cmp](https://doc.rust-lang.org/std/cmp/index.html)
- [std::hash::Hash](https://doc.rust-lang.org/std/hash/trait.Hash.html)
- [Send and Sync (Nomicon)](https://doc.rust-lang.org/nomicon/send-and-sync.html)
- [Sized (Rust Reference)](https://doc.rust-lang.org/reference/special-types-and-traits.html#sized)
- [Deref coercions (The Rust Book)](https://doc.rust-lang.org/book/ch15-02-deref.html)
