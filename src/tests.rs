#![allow(clippy::bool_assert_comparison)]

use core::f32::consts::PI;
use core::alloc::Layout;

use crate::pool::Stack;
use crate::Arena;

#[test]
fn boxed() {
    static mut DROPPED: bool = false;

    struct Droppable;

    impl Drop for Droppable {
        fn drop(&mut self) {
            unsafe { DROPPED = true; }
        }
    }

    let arena = Arena::new();
    let boxed = arena.alloc_boxed(Droppable);
    drop(boxed);

    assert!(unsafe { DROPPED });
}

#[test]
fn boxed_slice() {
    let arena = Arena::new();
    let boxed = arena.alloc_slice_with_boxed(|i| i + 1, 512);
    assert_eq!(boxed.len(), 512);
}

#[test]
fn arena_single_int() {
    let arena = Arena::new();
    let val = 42i32;
    let ptr = arena.alloc(val);
    assert_eq!(*ptr, 42);
}

#[test]
fn arena_single_float() {
    let arena = Arena::new();
    let val = PI;
    let ptr = arena.alloc(val);
    assert_eq!(*ptr, PI);
}

#[test]
fn arena_single_struct() {
    #[derive(PartialEq, Debug)]
    struct Point {
        x: i32,
        y: i32
    }
    let arena = Arena::new();
    let val = Point { x: 10, y: 20 };
    let ptr = arena.alloc(val);
    assert_eq!(*ptr, Point { x: 10, y: 20 });
}

#[test]
fn arena_slice_copied_empty() {
    let arena = Arena::new();
    let slice: &[i32] = &[];
    let ptr = arena.alloc_slice_copied(slice);
    assert_eq!(ptr.len(), 0);
}

#[test]
fn arena_slice_copied_single() {
    let arena = Arena::new();
    let slice = [42i32];
    let ptr = arena.alloc_slice_copied(&slice);
    assert_eq!(ptr.len(), 1);
    assert_eq!(ptr[0], 42);
}

#[test]
fn arena_slice_copied_multiple() {
    let arena = Arena::new();
    let slice = [1, 2, 3, 4, 5];
    let ptr = arena.alloc_slice_copied(&slice);
    assert_eq!(ptr.len(), 5);
    for (i, &val) in ptr.iter().enumerate() {
        assert_eq!(val, i + 1);
    }
}

#[test]
fn arena_slice_copied_different_types() {
    let arena = Arena::new();
    let slice = [true, false, true];
    let ptr = arena.alloc_slice_copied(&slice);
    assert_eq!(ptr.len(), 3);
    assert_eq!(ptr[0], true);
    assert_eq!(ptr[1], false);
    assert_eq!(ptr[2], true);
}

#[test]
fn arena_slice_cloned_empty() {
    let arena = Arena::new();
    let slice: &[String] = &[];
    let ptr = arena.alloc_slice_cloned(slice);
    assert_eq!(ptr.len(), 0);
}

#[test]
fn arena_slice_cloned_single() {
    let arena = Arena::new();
    let slice = ["hello"];
    let ptr = arena.alloc_slice_cloned(&slice);
    assert_eq!(ptr.len(), 1);
    assert_eq!(ptr[0], "hello");
}

#[test]
fn arena_slice_cloned_multiple() {
    let arena = Arena::new();
    let slice = ["a", "b", "c"];
    let ptr = arena.alloc_slice_cloned(&slice);
    assert_eq!(ptr.len(), 3);
    assert_eq!(ptr[0], "a");
    assert_eq!(ptr[1], "b");
    assert_eq!(ptr[2], "c");
}

#[test]
fn arena_str_copied_empty() {
    let arena = Arena::new();
    let s = "";
    let ptr = arena.alloc_str_copied(s);
    assert_eq!(ptr.len(), 0);
    assert_eq!(ptr, "");
}

#[test]
fn arena_str_copied_single_char() {
    let arena = Arena::new();
    let s = "a";
    let ptr = arena.alloc_str_copied(s);
    assert_eq!(ptr.len(), 1);
    assert_eq!(ptr, "a");
}

#[test]
fn arena_str_copied_unicode() {
    let arena = Arena::new();
    let s = "Hello, 世界!";
    let ptr = arena.alloc_str_copied(s);
    assert_eq!(ptr.len(), s.len());
    assert_eq!(ptr, s);
}

#[test]
fn arena_str_copied_non_ascii() {
    let arena = Arena::new();
    let s = "café";
    let ptr = arena.alloc_str_copied(s);
    assert_eq!(ptr, s);
}

#[test]
fn arena_layout_basic() {
    let arena = Arena::new();
    let layout = Layout::new::<i32>();
    let ptr = arena.alloc_layout(layout);
    unsafe {
        ptr.write(123);
        assert_eq!(*ptr.as_ptr(), 123);
    }
}

#[test]
fn arena_layout_align_8() {
    let arena = Arena::new();
    let layout = Layout::from_size_align(8, 8).unwrap();
    let ptr = arena.alloc_layout(layout);
    let addr = ptr.as_ptr() as usize;
    assert_eq!(addr % 8, 0);
}

#[test]
fn arena_layout_align_16() {
    let arena = Arena::new();
    let layout = Layout::from_size_align(16, 16).unwrap();
    let ptr = arena.alloc_layout(layout);
    let addr = ptr.as_ptr() as usize;
    assert_eq!(addr % 16, 0);
}

#[test]
fn arena_layout_multiple() {
    let arena = Arena::new();
    let layout1 = Layout::new::<i32>();
    let layout2 = Layout::new::<f64>();
    let ptr1 = arena.alloc_layout(layout1);
    let ptr2 = arena.alloc_layout(layout2).cast();
    unsafe {
        ptr1.write(42);
        ptr2.write(PI);
        assert_eq!(*ptr1.as_ptr(), 42);
        assert_eq!(*ptr2.as_ptr(), PI);
    }
}

#[test]
fn arena_layout_size_0() {
    let arena = Arena::new();
    let layout = Layout::from_size_align(0, 1).unwrap();
    let ptr = arena.try_alloc_layout(layout);
    
    assert!(ptr.is_some())
}

#[test]
fn arena_multiple_values() {
    let arena = Arena::new();
    let a = arena.alloc(1i32);
    let b = arena.alloc(2i32);
    let c = arena.alloc(3i32);
    assert_eq!(*a, 1);
    assert_eq!(*b, 2);
    assert_eq!(*c, 3);
}

#[test]
fn arena_str_then_int() {
    let arena = Arena::new();
    let s = arena.alloc_str_copied("test");
    let i = arena.alloc(99i32);
    assert_eq!(s, "test");
    assert_eq!(*i, 99);
}

#[test]
fn arena_slice_then_str() {
    let arena = Arena::new();
    let slice = arena.alloc_slice_copied(&[1, 2, 3]);
    let str = arena.alloc_str_copied("hello");
    assert_eq!(slice[0], 1);
    assert_eq!(slice[1], 2);
    assert_eq!(slice[2], 3);
    assert_eq!(str, "hello");
}

#[test]
fn arena_with_capacity() {
    let arena = Arena::with_capacity(100);
    let a = arena.alloc(1i32);
    let b = arena.alloc_slice_copied(&[2, 3, 4]);
    let c = arena.alloc_str_copied("short");
    assert_eq!(*a, 1);
    assert_eq!(b.len(), 3);
    assert_eq!(c, "short");
}

#[test]
fn arena_alignment_padding() {
    let arena = Arena::new();
    let a = arena.alloc(1i8);
    let b = arena.alloc(2i32);
    let addr_a = a as *const i8 as usize;
    let addr_b = b as *const i32 as usize;
    assert_eq!(addr_b % 4, 0);
    assert!(addr_b < addr_a);
}

#[test]
fn arena_slice_cloned_with_struct() {
    #[derive(Clone, PartialEq, Debug)]
    struct S {
        x: i32
    }
    let arena = Arena::new();
    let slice = [S { x: 1 }, S { x: 2 }];
    let ptr = arena.alloc_slice_cloned(&slice);
    assert_eq!(ptr[0].x, 1);
    assert_eq!(ptr[1].x, 2);
}

#[test]
fn arena_layout_large_size() {
    let arena = Arena::new();
    let layout = Layout::from_size_align(1024, 8).unwrap();
    let ptr = arena.alloc_layout(layout);
    unsafe {
        ptr.as_ptr().add(1023).write(42u8);
        assert_eq!(*ptr.as_ptr().add(1023), 42);
    }
}

#[test]
fn arena_str_copied_preserves_content() {
    let arena = Arena::new();
    let s = "This is a long string with multiple words and punctuation!";
    for _ in 0..100 {
        let ptr = arena.alloc_str_copied(s);
        assert_eq!(ptr, s);
    }
}

#[test]
fn arena_and_drop_arena() {
    let arena = Arena::new();
    drop(arena);
}

#[test]
fn arena_str_copied_large() {
    let arena = Arena::new();
    let s = "a".repeat(1000);
    let ptr = arena.alloc_str_copied(&s);
    assert_eq!(ptr.len(), 1000);
    assert_eq!(*ptr, s);
}

#[test]
fn arena_layout_mixed_sizes() {
    let arena = Arena::new();
    let l1 = Layout::new::<u8>();
    let l2 = Layout::new::<u16>();
    let l3 = Layout::new::<u32>();
    let l4 = Layout::new::<u64>();
    let p1 = arena.alloc_layout(l1);
    let p2 = arena.alloc_layout(l2).cast();
    let p3 = arena.alloc_layout(l3).cast();
    let p4 = arena.alloc_layout(l4).cast();
    unsafe {
        p1.write(1u8);
        p2.write(2u16);
        p3.write(3u32);
        p4.write(4u64);
        assert_eq!(*p1.as_ptr(), 1);
        assert_eq!(*p2.as_ptr(), 2);
        assert_eq!(*p3.as_ptr(), 3);
        assert_eq!(*p4.as_ptr(), 4);
    }
}

#[test]
fn arena_multiple_types_in_sequence() {
    let arena = Arena::new();
    let a = arena.alloc(1i8);
    let b = arena.alloc(2u16);
    let c = arena.alloc(3u32);
    let d = arena.alloc(4u64);
    let e = arena.alloc_str_copied("test");
    let f = arena.alloc_slice_copied(&[5u8, 6u8]);
    assert_eq!(*a, 1);
    assert_eq!(*b, 2);
    assert_eq!(*c, 3);
    assert_eq!(*d, 4);
    assert_eq!(e, "test");
    assert_eq!(f[0], 5);
    assert_eq!(f[1], 6);
}

#[test]
fn arena_dealloc_unaligned() {
    // This test doesn't have an assertion because it's intended for Miri
    let arena = Arena::with_capacity(32);
    let layout = Layout::from_size_align(31, 8).unwrap();
    let ptr = arena.alloc_layout(layout);
    unsafe {
        arena.dealloc(ptr, layout);
    }
    arena.alloc_layout(layout);
}

#[test]
fn stack_new() {
    let stack = Stack::<u8>::new();
    assert_eq!(stack.cap(), 0);
}

#[test]
fn stack_with_capacity() {
    let stack = Stack::<u8>::with_capacity(3);
    assert_eq!(stack.cap(), 3);
}

#[test]
fn stack_push() {
    let mut stack = Stack::<u8>::new();
    stack.push(1);
    stack.push(2);
    
    assert_eq!(stack.len(), 2);
}

#[test]
fn stack_pop() {
    let mut stack = Stack::<u8>::new();
    stack.push(255);
    stack.push(0);

    assert_eq!(stack.pop(), Some(0));
    assert_eq!(stack.pop(), Some(255));
}

#[test]
fn stack_reserve() {
    let mut stack = Stack::<u8>::new();
    assert_eq!(stack.cap(), 0);
    
    stack.reserve(3);
    assert_eq!(stack.cap(), 3);
}