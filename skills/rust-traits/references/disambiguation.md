# Trait Disambiguation in Rust

## Problem

A type can implement multiple traits with identically named methods. Calling the method directly is ambiguous.

## Methods with `&self` -- Use `Trait::method(&instance)`

```rust
trait Pilot {
    fn fly(&self);
}
trait Wizard {
    fn fly(&self);
}

struct Human;

impl Pilot for Human {
    fn fly(&self) { println!("This is your captain speaking."); }
}
impl Wizard for Human {
    fn fly(&self) { println!("Up!"); }
}
impl Human {
    fn fly(&self) { println!("*waving arms furiously*"); }
}

fn main() {
    let person = Human;
    person.fly();          // Human's own method: "*waving arms furiously*"
    Pilot::fly(&person);   // Pilot trait: "This is your captain speaking."
    Wizard::fly(&person);  // Wizard trait: "Up!"
}
```

When calling `person.fly()`, Rust prefers the direct `impl Human` method. Prefix with the trait name to select a specific implementation.

## Associated Functions (no `self`) -- Fully Qualified Syntax

```rust
trait Animal {
    fn baby_name() -> String;
}

struct Dog;

impl Dog {
    fn baby_name() -> String { String::from("Spot") }
}
impl Animal for Dog {
    fn baby_name() -> String { String::from("puppy") }
}

fn main() {
    println!("{}", Dog::baby_name());              // "Spot" (inherent impl)
    println!("{}", <Dog as Animal>::baby_name());   // "puppy" (trait impl)
}
```

## Fully Qualified Syntax (General Form)

```
<Type as Trait>::function(receiver_if_method, next_arg, ...)
```

- For methods: `<Form as UsernameWidget>::get(&form)`
- For associated functions: `<Dog as Animal>::baby_name()`

Use this when:
1. Multiple traits have methods/functions with the same name
2. Rust cannot infer which implementation to call
3. You need to be explicit about which trait's method to invoke

## Return Type Disambiguation

When two traits have `get` methods with different return types:

```rust
trait UsernameWidget {
    fn get(&self) -> String;
}
trait AgeWidget {
    fn get(&self) -> u8;
}

struct Form { username: String, age: u8 }

impl UsernameWidget for Form {
    fn get(&self) -> String { self.username.clone() }
}
impl AgeWidget for Form {
    fn get(&self) -> u8 { self.age }
}

let form = Form { username: "rustacean".into(), age: 28 };
let username = <Form as UsernameWidget>::get(&form); // String
let age = <Form as AgeWidget>::get(&form);           // u8
```

## Sources

- [Disambiguating (Rust by Example)](https://doc.rust-lang.org/rust-by-example/trait/disambiguating.html)
- [Fully qualified syntax (The Rust Book)](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name)
