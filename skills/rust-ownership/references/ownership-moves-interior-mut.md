# Ownership, Move Semantics, and Interior Mutability

## Move Semantics

Assignment, passing to functions, and returning from functions **move** ownership by default. The source becomes invalid after the move.

```rust
let s = String::from("hello");
let t = s;           // s is moved to t
// s is now invalid -- compile error if used

fn takes_ownership(s: String) { /* s dropped at end */ }
takes_ownership(t);  // t is moved into the function
// t is now invalid
```

Move is a **memcpy** of the stack representation (pointer, length, capacity for String). No heap allocation or deep copy occurs. The source is simply marked as uninitialized.

### Return moves

```rust
fn create() -> String {
    let s = String::from("hello");
    s  // ownership transferred to caller (NRVO may elide the copy)
}

let owned = create(); // owned takes ownership
```

## Copy vs Clone

### `Copy` -- implicit bitwise duplication

Types implementing `Copy` are duplicated on assignment instead of moved. Must also implement `Clone`. Only for types where a bitwise copy is correct (no heap pointers, no `Drop` impl).

```rust
// Copy types: all primitives, &T, tuples/arrays of Copy types
let x: i32 = 5;
let y = x;     // bitwise copy, x is still valid
println!("{x} {y}");

// A struct is Copy only if ALL fields are Copy
#[derive(Copy, Clone)]
struct Point { x: f64, y: f64 }

// CANNOT be Copy: String, Vec<T>, Box<T>, or any type with Drop
```

**Rule**: A type cannot implement both `Copy` and `Drop`. If a type needs a destructor, it must be moved, not copied.

### `Clone` -- explicit deep duplication

`Clone` provides an explicit `.clone()` method for deep copies:

```rust
let s1 = String::from("hello");
let s2 = s1.clone();  // heap data duplicated
println!("{s1} {s2}"); // both valid
```

### Decision tree

1. Is it a primitive or `&T`? -> Already `Copy`
2. All fields `Copy` and no `Drop` needed? -> Derive `Copy + Clone`
3. Deep copy makes sense? -> Derive or implement `Clone`
4. Expensive to clone? -> Consider `Arc<T>` for shared ownership

## Partial Moves

Destructuring can move some fields while borrowing others:

```rust
struct Person { name: String, age: u32 }

let person = Person { name: String::from("Alice"), age: 30 };
let Person { name, ref age } = person;
// name: moved out   -- person.name is invalid
// age: borrowed      -- *age == 30
// person: partially moved -- cannot use as a whole
println!("{name} is {age}");
```

## Owned vs Borrowed Types

| Owned | Borrowed | Use case |
|-------|----------|----------|
| `String` | `&str` | Text data |
| `Vec<T>` | `&[T]` | Sequences |
| `PathBuf` | `&Path` | File paths |
| `OsString` | `&OsStr` | OS strings |
| `CString` | `&CStr` | C interop |
| `Box<T>` | `&T` | Heap-allocated values |

**Guideline**: Accept `&str` / `&[T]` in function parameters (maximally flexible). Return `String` / `Vec<T>` when the function creates new data.

```rust
// Good: accepts both String and &str via deref coercion
fn greet(name: &str) -> String {
    format!("Hello, {name}!")
}

greet("world");                    // &str directly
greet(&String::from("world"));    // String auto-derefs to &str
```

## Reborrowing

When passing `&mut T` to a function, the compiler creates a temporary shorter-lived `&mut` (a **reborrow**) rather than moving the original:

```rust
fn push_one(v: &mut Vec<i32>) { v.push(1); }

let mut v = vec![];
let r: &mut Vec<i32> = &mut v;
push_one(r);  // reborrow: &mut *r -- r is NOT consumed
push_one(r);  // works again
r.push(2);    // r still valid
```

This happens implicitly for `&mut` references. Shared references (`&T`) are `Copy`, so they don't need reborrowing.

### Reborrow through deref

```rust
let mut s = String::from("hello");
let r: &mut String = &mut s;
// &mut String -> &mut str via DerefMut
let slice: &mut str = &mut *r;  // reborrow + deref
```

## Two-Phase Borrowing

The compiler allows a limited form of overlapping borrows to support common method-call patterns:

```rust
let mut v = vec![1, 2, 3];
v.push(v.len());  // OK since Rust 2021: two-phase borrow
// Phase 1: &mut v reserved (not yet activated)
// Phase 2: &v used for v.len()
// Phase 3: &mut v activated for push
```

Two-phase borrows only apply to method calls and indexing, not arbitrary code.

## Interior Mutability

Allows mutation through `&T` by shifting borrow rules from compile time to runtime (or using `unsafe`).

### `Cell<T>` -- zero-cost for `Copy` types

Provides `.get()` and `.set()` with no runtime overhead. `T` must be `Copy`.

```rust
use std::cell::Cell;

let counter = Cell::new(0u32);
counter.set(counter.get() + 1);  // mutate through &Cell<u32>
```

**Not thread-safe** (`!Sync`). Use for single-threaded mutation of `Copy` values.

### `RefCell<T>` -- runtime borrow checking

Dynamic borrow checking: `borrow()` -> `Ref<T>`, `borrow_mut()` -> `RefMut<T>`. Panics at runtime if rules violated.

```rust
use std::cell::RefCell;

let data = RefCell::new(vec![1, 2, 3]);
data.borrow_mut().push(4);             // exclusive borrow at runtime
println!("{:?}", data.borrow());       // shared borrow at runtime

// PANIC: overlapping borrow_mut() and borrow()
// let r = data.borrow();
// let w = data.borrow_mut();  // panics: already borrowed
```

**Pattern**: `Rc<RefCell<T>>` for shared mutable ownership in single-threaded code:

```rust
use std::rc::Rc;
use std::cell::RefCell;

let shared = Rc::new(RefCell::new(vec![1, 2]));
let clone = Rc::clone(&shared);
shared.borrow_mut().push(3);
println!("{:?}", clone.borrow()); // [1, 2, 3]
```

### `UnsafeCell<T>` -- the primitive

All interior mutability types are built on `UnsafeCell<T>`. It provides `.get() -> *mut T`. Only use directly when building custom synchronization primitives.

```rust
use std::cell::UnsafeCell;

let uc = UnsafeCell::new(5);
unsafe { *uc.get() = 10; }  // raw pointer mutation
```

### `OnceCell<T>` / `OnceLock<T>`

Write-once interior mutability. `OnceCell` for single-threaded, `OnceLock` for multi-threaded:

```rust
use std::sync::OnceLock;

static CONFIG: OnceLock<String> = OnceLock::new();
CONFIG.get_or_init(|| "production".to_string());
println!("{}", CONFIG.get().unwrap());
```

## `mem::replace` and `mem::take` -- Moving Out of `&mut`

You can't move a value out of a `&mut` reference directly (it would leave the source uninitialized). Use `mem::replace` or `mem::take`:

```rust
use std::mem;

let mut s = String::from("hello");
let r = &mut s;
let owned = mem::take(r);           // r is now "" (Default), owned is "hello"
let old = mem::replace(r, String::from("world")); // r is "world", old is ""
```

## Sources

- [Ownership (Rust Book ch4.1)](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html)
- [References and Borrowing (Rust Book ch4.2)](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html)
- [Interior mutability (Rust Reference)](https://doc.rust-lang.org/reference/interior-mutability.html)
- [std::cell module](https://doc.rust-lang.org/std/cell/index.html)
- [std::mem module](https://doc.rust-lang.org/std/mem/index.html)
