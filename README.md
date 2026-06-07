# Flintnsteel

[![License](https://img.shields.io/crates/l/flintnsteel)](LICENSE)
[![Version](https://img.shields.io/crates/v/flintnsteel)](https://crates.io/crates/flintnsteel)

An arena allocator, lightweight and fast. Read mode in [the main documentation](https://docs.rs/flintnsteel).

# Example

```rust
use flintnsteel::Arena;

fn main() {
    let arena = Arena::new();
    assert_eq!(*arena.alloc(0), 0);
}
```
