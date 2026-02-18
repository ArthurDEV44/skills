# Trait Objects & Dynamic Dispatch

## What is a Trait Object?

A trait object (`dyn Trait`) is a fat pointer: a data pointer + a vtable pointer. The vtable contains function pointers for each trait method, enabling runtime polymorphism.

```rust
// &dyn Trait = 2 words: (*data, *vtable)
// Box<dyn Trait> = 2 words: (*heap_data, *vtable)
```

## Trait Object Syntax

```rust
dyn Trait
dyn Trait + Send
dyn Trait + Send + Sync
dyn Trait + 'static
dyn Trait + Send + 'static
```

Only **one** non-auto trait is allowed. You can add auto traits (`Send`, `Sync`, `Unpin`) and a lifetime bound.

## Object Safety Rules

A trait is **object-safe** (can be used as `dyn Trait`) when ALL of these hold:

1. **No `Self: Sized` supertrait** -- `trait Foo: Sized {}` is NOT object-safe
2. **All methods are object-safe:**
   - Receiver is `&self`, `&mut self`, `self: Box<Self>`, `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` where P is one of the above
   - No generic type parameters on the method
   - Does not return `Self` (the concrete type is erased)
   - Does not use `impl Trait` in argument or return position
3. **Associated functions without a receiver** must have `where Self: Sized` bound (this opts them out of the vtable)

### Making Non-Object-Safe Methods Opt Out

```rust
trait MyTrait {
    fn normal(&self);                          // Object-safe

    fn returns_self(&self) -> Self             // NOT object-safe...
    where Self: Sized;                         // ...but this excludes it from dyn

    fn generic<T>(&self, val: T)               // NOT object-safe...
    where Self: Sized;                         // ...excluded from dyn
}

// Now `dyn MyTrait` works, but `returns_self` and `generic` are unavailable on trait objects
```

### Common Object Safety Violations

| Pattern | Problem | Fix |
|---------|---------|-----|
| `fn clone(&self) -> Self` | Returns `Self` | Add `where Self: Sized` or return `Box<dyn Trait>` |
| `fn foo<T>(&self, t: T)` | Generic method | Add `where Self: Sized` or use `&dyn Any` |
| `trait Foo: Sized` | Sized supertrait | Remove `Sized` bound |
| `fn new() -> Self` | Associated fn returning Self | Add `where Self: Sized` |

## Trait Object Lifetimes

Trait objects have an implicit lifetime. Default lifetime elision rules:

```rust
// Box<dyn Trait> is Box<dyn Trait + 'static> (owned, defaults to 'static)
// &'a dyn Trait is &'a (dyn Trait + 'a) (borrowed, inherits reference lifetime)

// Be explicit when needed:
fn process(handler: Box<dyn Handler + 'static>) { /* ... */ }
fn borrow_handler<'a>(handler: &'a dyn Handler) { /* ... */ }
```

## Using Trait Objects

### Heterogeneous Collections

```rust
let shapes: Vec<Box<dyn Shape>> = vec![
    Box::new(Circle { radius: 5.0 }),
    Box::new(Rectangle { w: 3.0, h: 4.0 }),
];
for shape in &shapes {
    println!("area: {}", shape.area());
}
```

### As Function Parameters

```rust
fn draw(shape: &dyn Drawable) { shape.draw(); }          // By reference
fn process(handler: Box<dyn Handler>) { handler.run(); }  // Owned
```

### Returning Trait Objects

```rust
fn make_processor(fast: bool) -> Box<dyn Processor> {
    if fast { Box::new(FastProcessor) }
    else    { Box::new(SlowProcessor) }
}
```

## Performance: Static vs Dynamic Dispatch

| | Static (`impl Trait` / generics) | Dynamic (`dyn Trait`) |
|---|---|---|
| **Dispatch** | Monomorphized at compile time | Vtable lookup at runtime |
| **Code size** | Larger (one copy per concrete type) | Smaller (single code path) |
| **Speed** | Faster (inlining possible) | Slight overhead (~1ns per call) |
| **Flexibility** | One concrete type per call site | Any type at runtime |
| **Use when** | Performance-critical, type known | Heterogeneous collections, plugins |

## Downcasting with `Any`

```rust
use std::any::Any;

fn handle(val: &dyn Any) {
    if let Some(s) = val.downcast_ref::<String>() {
        println!("Got string: {s}");
    } else if let Some(n) = val.downcast_ref::<i32>() {
        println!("Got int: {n}");
    }
}
```

## Sources

- [Trait Objects (Rust Reference)](https://doc.rust-lang.org/reference/types/trait-object.html)
- [Object Safety (Rust Reference)](https://doc.rust-lang.org/reference/items/traits.html#object-safety)
- [Dynamic dispatch (Rust by Example)](https://doc.rust-lang.org/rust-by-example/trait/dyn.html)
- [Using Trait Objects (The Rust Book)](https://doc.rust-lang.org/book/ch18-02-trait-objects.html)
