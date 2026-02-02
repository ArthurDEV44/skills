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
fn get_str() -> &str;                                // ERROR
fn frob(s: &str, t: &str) -> &str;                   // ERROR
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

## Unbounded Lifetimes

Unsafe code can produce references with **unbounded** lifetimes (no constraint on how long they live):

```rust
fn get_str<'a>(s: *const String) -> &'a str {
    unsafe { &*s }  // unbounded: 'a can be anything
}
```

An unbounded lifetime is more powerful than `'static` -- it molds to whatever context demands. This is almost always wrong and can cause use-after-free.

### Fix: bound the lifetime

- Use elision at function boundaries (output lifetime gets bounded by input)
- Return from a function with a bound lifetime
- Within a function, store the reference in a location with a known lifetime

```rust
// SAFE: output lifetime bounded by input
fn get_str(s: &String) -> &str {
    s.as_str()  // lifetime tied to input
}
```

## Sources

- [Lifetime elision](https://doc.rust-lang.org/nomicon/lifetime-elision.html)
- [Higher-Rank Trait Bounds](https://doc.rust-lang.org/nomicon/hrtb.html)
- [Unbounded lifetimes](https://doc.rust-lang.org/nomicon/unbounded-lifetimes.html)
