# Advanced Trait Patterns

## Blanket Implementations

Implement a trait for all types satisfying a bound:

```rust
// Standard library example:
impl<T: Display> ToString for T {
    fn to_string(&self) -> String {
        format!("{self}")
    }
}
```

Writing your own:

```rust
trait Loggable {
    fn log(&self);
}

// Blanket: anything Debug + Display gets Loggable for free
impl<T: Debug + Display> Loggable for T {
    fn log(&self) {
        println!("[LOG] {self} ({self:?})");
    }
}
```

**Caution:** Blanket impls are powerful but limit downstream implementors. If `impl<T: Foo> Bar for T` exists, no one can manually implement `Bar` for a type that also implements `Foo`.

## Coherence & Orphan Rule

**Orphan rule:** You can only implement trait `T` for type `S` if at least one of `T` or `S` is defined in your crate.

| Trait | Type | Allowed? |
|-------|------|----------|
| Your trait | Your type | Yes |
| Your trait | External type | Yes |
| External trait | Your type | Yes |
| External trait | External type | **No** |

### Newtype Pattern (Workaround)

Wrap the external type in a newtype to claim ownership:

```rust
struct Wrapper(Vec<String>);

impl fmt::Display for Wrapper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}]", self.0.join(", "))
    }
}
```

Implement `Deref` to transparently access inner methods:

```rust
impl std::ops::Deref for Wrapper {
    type Target = Vec<String>;
    fn deref(&self) -> &Vec<String> { &self.0 }
}
// wrapper.len(), wrapper.iter(), etc. all work
```

## Sealed Traits

Prevent external crates from implementing your trait:

```rust
mod private {
    pub trait Sealed {}
}

pub trait MyTrait: private::Sealed {
    fn do_thing(&self);
}

// Only types in this crate can implement Sealed, and therefore MyTrait
pub struct Foo;
impl private::Sealed for Foo {}
impl MyTrait for Foo {
    fn do_thing(&self) { /* ... */ }
}
```

Use when you want to add methods to the trait later without breaking downstream code.

## Extension Traits

Add methods to types you don't own via a new trait:

```rust
trait StrExt {
    fn is_blank(&self) -> bool;
}

impl StrExt for str {
    fn is_blank(&self) -> bool {
        self.trim().is_empty()
    }
}

// Now available on all &str:
"  ".is_blank(); // true
```

Convention: name extension traits `<Type>Ext` (e.g., `IteratorExt`, `ResultExt`).

## Marker Traits

Traits with no methods, used to tag types with properties:

```rust
// Standard library markers (auto traits):
// Send -- safe to transfer across threads
// Sync -- safe to share references across threads
// Unpin -- safe to move out of Pin
// Sized -- known size at compile time

// Custom marker:
trait Validated {}

fn save<T: Validated + Serialize>(data: &T) {
    // Only validated data can be saved
}
```

### Negative Implementations

Opt out of auto traits:

```rust
struct NotSend(*mut u8);
// NotSend is automatically !Send because it contains a raw pointer

// Explicitly opt out (nightly only):
// impl !Send for MyType {}
```

## Generic Associated Types (GATs)

Associated types with their own generic parameters (stabilized in Rust 1.65):

```rust
trait LendingIterator {
    type Item<'a> where Self: 'a;
    fn next(&mut self) -> Option<Self::Item<'_>>;
}

impl LendingIterator for WindowsMut {
    type Item<'a> = &'a mut [u8] where Self: 'a;
    fn next(&mut self) -> Option<&mut [u8]> {
        // Return a mutable slice borrowing from self
        /* ... */
    }
}
```

GATs enable patterns where the associated type borrows from `self`, which regular associated types cannot express.

## Type-State Pattern with Traits

Use zero-sized marker types and trait bounds to encode state in the type system:

```rust
struct Locked;
struct Unlocked;

struct Door<State> {
    _state: std::marker::PhantomData<State>,
}

trait Openable {}
impl Openable for Unlocked {}

impl<S> Door<S> {
    fn inspect(&self) { println!("Inspecting door"); }
}

impl Door<Locked> {
    fn unlock(self) -> Door<Unlocked> {
        Door { _state: std::marker::PhantomData }
    }
}

impl Door<Unlocked> {
    fn open(&self) { println!("Opening door"); }
    fn lock(self) -> Door<Locked> {
        Door { _state: std::marker::PhantomData }
    }
}
// door.open() only compiles if door is Door<Unlocked>
```

## Trait Design Guidelines

1. **Prefer associated types over generics** when there's one natural implementation per type
2. **Keep traits focused** -- one responsibility per trait, compose with supertraits
3. **Provide defaults** for methods that have sensible default implementations
4. **Seal your trait** if you plan to add methods in future versions
5. **Use extension traits** to add convenience methods to external types
6. **Document object safety** -- explicitly state whether your trait is meant for `dyn` usage
7. **Prefer `impl Trait` in arguments** over `dyn Trait` unless heterogeneous dispatch is needed
8. **Consider blanket impls carefully** -- they constrain all future implementors

## Sources

- [Advanced Traits (The Rust Book)](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html)
- [Coherence (Rust Reference)](https://doc.rust-lang.org/reference/items/implementations.html#trait-implementation-coherence)
- [Generic Associated Types (RFC 1598)](https://rust-lang.github.io/rfcs/1598-generic_associated_types.html)
- [API Guidelines - Future Proofing](https://rust-lang.github.io/api-guidelines/future-proofing.html)
- [Sealed traits pattern](https://rust-lang.github.io/api-guidelines/future-proofing.html#sealed-traits-protect-against-downstream-implementations-c-sealed)
