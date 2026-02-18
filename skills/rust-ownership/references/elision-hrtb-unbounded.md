# Lifetime Elision, HRTBs, and Unbounded Lifetimes

## Lifetime Elision Rules

Lifetime positions: anywhere a lifetime can appear (`&'a T`, `&'a mut T`, `T<'a>`).

**Input** = function argument types. **Output** = return types.

### Rules (applied in order)

1. Each elided lifetime in input position becomes a distinct parameter
2. If exactly one input lifetime (elided or not), assign it to all elided outputs
3. If one input is `&self` / `&mut self`, assign self's lifetime to all elided outputs
4. Otherwise, eliding an output lifetime is an error

### Examples

```rust
fn print(s: &str);                                   // -> fn print<'a>(s: &'a str)
fn debug(lvl: usize, s: &str);                       // -> fn debug<'a>(lvl: usize, s: &'a str)
fn substr(s: &str, until: usize) -> &str;            // -> fn substr<'a>(s: &'a str, ...) -> &'a str
fn get_mut(&mut self) -> &mut T;                     // -> fn get_mut<'a>(&'a mut self) -> &'a mut T
fn args(&mut self, args: &[T]) -> &mut Command;      // -> fn args<'a, 'b>(&'a mut self, args: &'b [T]) -> &'a mut Command
fn new(buf: &mut [u8]) -> BufWriter<'_>;             // -> fn new<'a>(buf: &'a mut [u8]) -> BufWriter<'a>

// ILLEGAL: ambiguous output lifetime
fn get_str() -> &str;                                // ERROR: no input lifetime
fn frob(s: &str, t: &str) -> &str;                   // ERROR: multiple input lifetimes
```

### Elision in `impl` blocks

```rust
impl<'a> MyStruct<'a> {
    fn method(&self) -> &str { &self.data }
    // Expands to: fn method<'b>(&'b self) -> &'b str
    // Note: output tied to &self ('b), NOT to 'a
}
```

### Elision in type aliases and trait objects

```rust
type FunPtr = fn(&str) -> &str;                      // -> for<'a> fn(&'a str) -> &'a str
type FunTrait = dyn Fn(&str) -> &str;                // -> dyn for<'a> Fn(&'a str) -> &'a str
```

### Elision in `impl Trait`

```rust
fn first_word(s: &str) -> impl Iterator<Item = char> + '_ {
    // '_ captures the input lifetime -- equivalent to:
    // fn first_word<'a>(s: &'a str) -> impl Iterator<Item = char> + 'a
    s.chars()
}
```

### Elision in `async fn`

```rust
async fn fetch(url: &str) -> Result<String, Error> { /* ... */ }
// The returned Future captures url's lifetime:
// fn fetch<'a>(url: &'a str) -> impl Future<Output = Result<String, Error>> + 'a
```

## Higher-Rank Trait Bounds (HRTB)

### The problem

When a closure takes and returns references, the lifetime can't be named in the `where` clause because it depends on how the closure is called:

```rust
struct Closure<F> {
    data: (u8, u16),
    func: F,
}

impl<F> Closure<F>
    where F: Fn(&(u8, u16)) -> &u8,  // what lifetime goes here?
{
    fn call(&self) -> &u8 {
        (self.func)(&self.data)
    }
}
```

### The solution: `for<'a>`

```rust
where for<'a> F: Fn(&'a (u8, u16)) -> &'a u8,
// or equivalently:
where F: for<'a> Fn(&'a (u8, u16)) -> &'a u8,
```

`for<'a>` means "for all choices of `'a`" -- F must work with any lifetime.

In practice, Rust's elision sugar on `Fn` traits handles this automatically. HRTBs are rarely written explicitly outside of advanced generic code.

### Common HRTB patterns

```rust
// Accepting a closure that processes references
fn apply<F>(f: F) where F: for<'a> Fn(&'a str) -> &'a str {
    let s = String::from("hello");
    println!("{}", f(&s));
}

// Trait bound: type must be deserializable from any lifetime
fn process<T>() where T: for<'de> Deserialize<'de> { /* ... */ }

// Higher-ranked trait bound on associated types
fn use_parser<P>() where for<'a> P: Parser<'a>, for<'a> <P as Parser<'a>>::Output: Debug { /* ... */ }
```

### When HRTBs are required vs inferred

| Context | HRTB needed? |
|---------|-------------|
| `Fn(&T) -> &U` in where clause | Inferred via elision (sugar for `for<'a>`) |
| Explicit lifetime in `Fn` bound | Write `for<'a>` manually |
| `Deserialize<'de>` bounds | Almost always `for<'de>` |
| Trait with lifetime parameter | Write `for<'a>` when trait is generic over the lifetime |

## Unbounded Lifetimes

Unsafe code can produce references with **unbounded** lifetimes (no constraint on how long they live):

```rust
fn get_str<'a>(s: *const String) -> &'a str {
    unsafe { &*s }  // unbounded: 'a can be anything
}
```

An unbounded lifetime is more powerful than `'static` -- it molds to whatever context demands. This is almost always wrong and can cause use-after-free:

```rust
fn main() {
    let soon_dropped = String::from("hello");
    let dangling = get_str(&soon_dropped);
    drop(soon_dropped);
    println!("{dangling}"); // use-after-free: undefined behavior!
}
```

### Fix: bound the lifetime

- Use elision at function boundaries (output lifetime gets bounded by input)
- Return from a function with a bound lifetime
- Within a function, store the reference in a location with a known lifetime

```rust
// SAFE: output lifetime bounded by input
fn get_str(s: &String) -> &str {
    s.as_str()  // lifetime tied to input via elision
}
```

### Recognizing unbounded lifetimes in unsafe code

Any time you dereference a raw pointer and produce a reference, the lifetime is unbounded:

```rust
unsafe fn from_raw<'a>(ptr: *const T) -> &'a T { &*ptr }         // unbounded
unsafe fn from_raw_mut<'a>(ptr: *mut T) -> &'a mut T { &mut *ptr } // unbounded

// Safe wrapper: bind lifetime to the source
fn from_ref<'a>(source: &'a Wrapper) -> &'a T {
    unsafe { &*source.ptr }  // bounded by source's lifetime
}
```

## Sources

- [Lifetime elision (Nomicon)](https://doc.rust-lang.org/nomicon/lifetime-elision.html)
- [Lifetime elision (Rust Reference)](https://doc.rust-lang.org/reference/lifetime-elision.html)
- [Higher-Rank Trait Bounds (Nomicon)](https://doc.rust-lang.org/nomicon/hrtb.html)
- [Unbounded lifetimes (Nomicon)](https://doc.rust-lang.org/nomicon/unbounded-lifetimes.html)
