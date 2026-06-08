# Flintnsteel

[![License](https://img.shields.io/crates/l/flintnsteel)](LICENSE)
[![Version](https://img.shields.io/crates/v/flintnsteel)](https://crates.io/crates/flintnsteel)

An arena allocator, lightweight and fast. Extendable with reference counting,
automatic drops and multithreaded allocator recycling.

Read more in [the main documentation](https://docs.rs/flintnsteel).

# Example

```rust
use flintnsteel::Arena;

fn main() {
    let arena = Arena::new();
    
    let string = arena.alloc_str_copied("What's up my allocation bros");
    assert_eq!(string.len(), 28);
    
    let num = arena.alloc(0);
    *num += 1;
    assert_eq!(*num, 1);
    
    let rc = arena.alloc_slice_copied_rc(&[1, 2, 3, 4, 5]);
    drop(arena);
    assert_eq!(rc, &[1, 2, 3, 4, 5]);
}
```
