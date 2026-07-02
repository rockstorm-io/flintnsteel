//! An arena allocator, light and fast.
//!
//! There are already many great arena allocators, the goal of this one wasn't just to
//! make a faster one, but to remove the overhead of the features you don't use. In other words,
//! this is an attempt of making one single crate for a case when you need to allocate a bulk of strings,
//! and for a case when you need reference-counted allocations, multithreaded sharing and drop types.
//! And remember, the second case shouldn't make the first one slower.
//!
//! This crate provides:
//! - [`Arena`] allocator.
//! - [`ArenaPool`] for multithreaded sharing (requires `std` crate).
//! - [`Box`], [`Rc`] as wrappers of an allocation.
//! - [`Vec`] as a growable array backed by the arena and [`vec_in!`] macro as
//!   an alternative to `vec!` (requires `vec` feature).
//!
//! A string type is planned to be added in the future.
//!
//! # Arena
//!
//! The [`Arena`] struct is the implementation of arena allocator. By itself,
//! it is minimal. Internally it has a linked list of heap-allocated chunks,
//! each containing refcount, link to the previous chunk and allocation
//! metadata.
//!
//! Its API comes in three variants, called allocation flavors. Regular allocation
//! simply returns a `&mut T`, types which implement [`Drop`] trait must be dropped
//! manually. On the other hand, boxed allocations return [`Box`], which will drop the
//! value when it gets dropped. Reference counted allocations return [`Rc`] which is
//! identical to having a `&T` that can outlive the arena.
//!
//! Allocation flavors are the main way of accomplishing the compromise, where
//! more high-level features doesn't penetrate simpler ones. See [`Arena`]
//! documentation for more.
//!
//! ## Example
//!
//! ```
//! use flintnsteel::Arena;
//!
//! let arena = Arena::new();
//!
//! let string = arena.alloc_str_copied("Why do we have Vec but don't have a String?");
//! assert_eq!(string, "Why do we have Vec but don't have a String?");
//!
//! let rc = arena.alloc_rc(0);
//! assert_eq!(*rc, 0);
//!
//! drop(arena);
//! assert_eq!(*rc, 0);
//! ```
//!
//! # `no_std`
//!
//! Since the allocator uses `GlobalAlloc` as a base allocator, the `alloc` crate required,
//! but `std` crate can be disabled using `no_std` feature.
//!
//! Disabling `std` will also turn off [`ArenaPool`] and [`ArenaPoolGuard`], because they
//! depend on [`Mutex`]
//!
//! [`Mutex`]: std::sync::Mutex
//! [`Vec`]: vec::Vec
//! [`Box`]: boxed::Box

#![cfg_attr(feature = "no_std", no_std)]
#![cfg_attr(feature = "allocator_api", feature(allocator_api))]

// TODO: add benchmarks

use crate::boxed::Box;
use crate::rc::Rc;

use core::ptr::{NonNull, copy_nonoverlapping, drop_in_place, slice_from_raw_parts_mut};
use core::str::from_utf8_unchecked_mut;
use core::slice::from_raw_parts_mut;
use core::cell::Cell;
use core::cmp::max;

extern crate alloc as _alloc;

use _alloc::alloc::{alloc, dealloc, Layout};

pub mod boxed;

#[cfg(feature = "vec")]
pub mod vec;

pub mod rc;

#[cfg(not(feature = "no_std"))]
mod pool;

#[cfg(test)]
mod tests;

#[cfg(not(feature = "no_std"))]
pub use pool::*;

/// Aligns `n` up to `align`.
///
/// # Example
///
/// ```rust,compile_fail
/// use crate::align_to_checked;
///
/// assert_eq!(
///     align_to_checked(10, 8),
///     Some(16)
/// );
/// ```
#[inline]
const fn align_to_checked(n: usize, align: usize) -> Option<usize> {
    match n.checked_add(align - 1) {
        Some(n) => Some(n & !(align - 1)),
        _ => None,
    }
}

/// Aligns `n` up to `align`, but without an overflow check.
///
/// # Safety
///
/// - rounding `n` to `align` must not overflow usize.
///
/// # Example
///
/// ```rust,compile_fail
/// use crate::align_to_unchecked;
///
/// assert_eq!(
///     unsafe { align_to_unchecked(10, 8) },
///     16
/// );
/// ```
#[inline]
const unsafe fn align_to_unchecked(n: usize, align: usize) -> usize {
    unsafe { align_to_checked(n, align).unwrap_unchecked() }
}

/// Aligns `ptr` down to `align`, keeping provenance.
///
/// # Example
///
/// ```rust,compile_fail
/// use crate::align_down_with_provenance;
///
/// assert_eq!(
///     align_down_with_provenance(9 as *mut u8, 4),
///     8 as *mut u8
/// );
/// ```
#[inline]
fn align_down_with_provenance(ptr: *mut u8, align: usize) -> *mut u8 {
    ptr.wrapping_sub(ptr.addr() & (align - 1))
}

/// Aligns `ptr` up to `align`, keeping provenance
#[inline]
unsafe fn align_up_non_null_with_provenance_unchecked(ptr: NonNull<u8>, align: usize) -> NonNull<u8> {
    unsafe {
        let addr = ptr.addr().get();
        let rounded = align_to_unchecked(addr, align);
        let delta = rounded - addr;
        ptr.add(delta)
    }
}

/// Checks if `addr` is aligned to `align`.
///
/// # Safety
///
/// - `align` must be non-zero and a power of two
#[cfg(any(feature = "vec", feature = "allocator_api", feature = "allocator-api2"))]
#[inline]
unsafe fn is_aligned_to_unchecked(addr: usize, align: usize) -> bool {
    addr & (align - 1) == addr
}

/// Alignment of arena's chunks
const CHUNK_ALIGNMENT: usize = 16;

/// Typical overhead of underlying allocators
const MALLOC_OVERHEAD: usize = 16;

/// Typical page size
const PAGE_SIZE: usize = 4 * 1024;

/// Size of [`ChunkFooter`] in bytes
const FOOTER_SIZE: usize = size_of::<ChunkFooter>();

const _: () = {
    assert!(MALLOC_OVERHEAD == CHUNK_ALIGNMENT);
    assert!(CHUNK_ALIGNMENT.is_power_of_two());
    assert!(align_of::<ChunkFooter>() == CHUNK_ALIGNMENT);
    assert!(PAGE_SIZE < isize::MAX as usize);
};

/// Overhead of a chunk allocation, in bytes
const OVERHEAD: usize = align_to_checked(MALLOC_OVERHEAD + FOOTER_SIZE, CHUNK_ALIGNMENT).unwrap();

/// Minimum size of a first chunk
const FIRST_CHUNK_SIZE: usize = 1 << 9;

/// Minimum size of usable space of a first chunk
const FIRST_CHUNK_SIZE_WITHOUT_OVERHEAD: usize = FIRST_CHUNK_SIZE - OVERHEAD;

/// Data required to deallocate a chunk of memory owned by an [`Arena`]
#[repr(C, align(16))]
#[derive(Debug)]
struct ChunkFooter {
    /// Reference counter of this chunk.
    ///
    /// Chunk can not be dropped until the count is zero
    refcount: Cell<usize>,

    /// Pointer to a footer of the previous chunk.
    ///
    /// `None` if the chunk doesn't exist
    previous_chunk_ptr: Cell<Option<NonNull<ChunkFooter>>>,

    /// Pointer to the allocation of this chunk
    ptr: NonNull<u8>,

    /// [`Layout`] of the allocation of this chunk
    layout: Layout,
}

impl ChunkFooter {
    /// Creates a new [`ChunkFooter`] from parts describing its chunk
    #[inline]
    fn new(ptr: NonNull<u8>, layout: Layout, previous_chunk_ptr: Option<NonNull<ChunkFooter>>) -> Self {
        Self {
            previous_chunk_ptr: Cell::new(previous_chunk_ptr),
            refcount: Cell::new(0),
            layout,
            ptr
        }
    }

    /// Increments refcount of a chunk described by its [`ChunkFooter`].
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid [`ChunkFooter`]
    pub(crate) unsafe fn increment_refcount(ptr: NonNull<Self>) {
        let footer = unsafe {
            // Safety: caller guarantees `ptr` to be valid `ChunkFooter`
            ptr.as_ref()
        };
        footer.refcount.update(|c| c.wrapping_add(1));
    }
    
    /// Deallocates a chunk described by [`ChunkFooter`] or decrements the reference
    /// counter if the chunk is still in use.
    /// 
    /// Returns pointer to the previous chunk.
    /// 
    /// # Safety
    /// 
    /// - `ptr` must point to a valid [`ChunkFooter`].
    /// 
    /// - if the chunk gets deallocated, data must not be accessed
    #[inline]
    pub(crate) unsafe fn drop_from_ptr(ptr: NonNull<Self>) -> Option<NonNull<ChunkFooter>> {
        let footer = unsafe {
            // Safety: caller guarantees `ptr` to be valid `ChunkFooter`
            ptr.as_ref()
        };
        let previous_chunk_ptr = footer.previous_chunk_ptr.get();
        
        let refcount = footer.refcount.get();
        if refcount == 0 {
            unsafe {
                // Safety: a valid `ChunkFooter` is guaranteed to follow `dealloc` safety
                // requirements. No other pointer to the allocation *should* exist because
                // refcount is zero
                dealloc(footer.ptr.as_ptr(), footer.layout)
            }
        } else {
            footer.refcount.set(refcount - 1);
        }
        
        previous_chunk_ptr
    }
}

// Some parts of `RawArena` implementation were derived from `oxc_allocator` crate
// https://github.com/oxc-project/oxc/blob/main/crates/oxc_allocator

/// Implementation of an arena allocator, publicly exposed by [`Arena`].
///
/// This struct provides a few simple methods similar to [`Allocator`] trait,
/// you can allocate layout, deallocate, grow of shrink. The main allocator,
/// i.e. [`Arena`], wraps this allocator to expose richer and more flexible API.
///
/// # Refcount
///
/// Each chunk has its own refcount, which blocks deallocation of the chunk
/// unless the count is zero. Evidently, after arena resets its chunks,
/// responsibility of deallocating referenced chunks goes entirely on
/// the incrementor.
///
/// # Allocation guarantees
///
/// Any allocation is valid for lifetime of the arena or unless `reset` is called.
/// Reference counted allocations are valid for their origin chunk until
/// the refcount is zero
#[repr(C)]
#[derive(Debug, Clone)]
struct RawArena {
    /// Bump allocation pointer, pointing to the last allocation of the current chunk.
    ///
    /// Equals to [`NonNull::dangling`] if this [`Arena`] has no chunks. Value is
    /// always within `start_ptr..=current_chunk_ptr`
    cursor_ptr: Cell<NonNull<u8>>,

    /// Pointer to a footer of current (last in the list) chunk.
    ///
    /// `None` if this [`Arena`] has no chunks
    current_chunk_ptr: Cell<Option<NonNull<ChunkFooter>>>,

    /// Pointer to allocatable region of current chunk.
    ///
    /// Equals to [`NonNull::dangling`] if this [`Arena`] has no chunks
    start_ptr: Cell<NonNull<u8>>,
}

impl RawArena {
    /// Creates a new [`RawArena`] without performing any allocations
    #[inline]
    fn new() -> Self {
        Self {
            cursor_ptr: Cell::new(NonNull::dangling()),
            current_chunk_ptr: Cell::new(None),
            start_ptr: Cell::new(NonNull::dangling()),
        }
    }

    /// Creates a new [`RawArena`] with pre-allocated `capacity`.
    ///
    /// Returns `None` if the global allocator failed to allocate enough memory
    fn try_with_capacity(capacity: usize) -> Option<Self> {
        let (chunk_ptr, start_ptr) = unsafe {
            new_chunk(capacity, 1, None)?
        };

        Some(Self {
            cursor_ptr: Cell::new(chunk_ptr.cast::<u8>()),
            current_chunk_ptr: Cell::new(Some(chunk_ptr)),
            start_ptr: Cell::new(start_ptr),
        })
    }

    // Allocation methods

    /// Attempts to allocate `layout` in the current chunk, otherwise
    /// creates a new one large enough to fit requested size.
    ///
    /// Returns `None` if the global allocation failed to allocate enough
    /// capacity or `layout` has invalid size or alignment
    #[inline]
    fn try_alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        match self.try_alloc_in(layout) {
            Some(ptr) => Some(ptr),
            _ => self.try_alloc_with_new(layout),
        }
    }

    /// Attempts to allocate `layout` in the current chunk.
    ///
    /// Returns `None` if the chunk doesn't have enough space to fit
    /// aligned layout
    #[inline]
    fn try_alloc_in(&self, layout: Layout) -> Option<NonNull<u8>> {
        let cursor_ptr = self.cursor_ptr.get().as_ptr();
        let start_ptr = self.start_ptr.get().as_ptr();

        let align = layout.align();

        let ptr = cursor_ptr.wrapping_sub(layout.size());
        let ptr_aligned = align_down_with_provenance(ptr, align);
        debug_assert!(ptr_aligned.addr().is_multiple_of(align));

        // Check the pointer to be within current chunk
        if ptr_aligned.addr().wrapping_sub(start_ptr.addr()) > isize::MAX as usize {
            return None;
        }

        let non_null = unsafe {
            // Safety: if `ptr_aligned` was null, it would have entered the branch above
            NonNull::new_unchecked(ptr_aligned)
        };
        self.cursor_ptr.set(non_null);

        Some(non_null)
    }

    /// Attempts to allocate `layout` in a new chunk.
    ///
    /// Returns `None` in case of: invalid layout alignment, failed allocation,
    /// allocation larger than `isize::MAX`
    #[inline(never)]
    #[cold]
    fn try_alloc_with_new(&self, layout: Layout) -> Option<NonNull<u8>> {
        let current_chunk_ptr = self.current_chunk_ptr.get();

        let minimum_size = layout.size().max(FIRST_CHUNK_SIZE_WITHOUT_OVERHEAD);
        let size = if let Some(ptr) = current_chunk_ptr {
            let previous_chunk = unsafe {
                // Safety: `ptr` is guaranteed to be valid if `current_chunk_ptr` is `Some`
                ptr.as_ref()
            };

            let previous_chunk_size = previous_chunk.layout.size() - FOOTER_SIZE;
            max(previous_chunk_size * 2, minimum_size)
        } else {
            minimum_size
        };

        let (chunk_ptr, start_ptr) = unsafe {
            // Safety: `Layout::align()` is guaranteed to return a `usize` which is a power of two
            // and `current_chunk_ptr` is a valid pointer to chunk
            new_chunk(size, layout.align(), current_chunk_ptr)?
        };

        self.current_chunk_ptr.set(Some(chunk_ptr));
        self.cursor_ptr.set(chunk_ptr.cast::<u8>());
        self.start_ptr.set(start_ptr);

        self.try_alloc_in(layout)
    }

    /// Deallocates an allocation inside this [`RawArena`] associated with its `ptr` and `layout`.
    ///
    /// # Safety
    ///
    /// Both `ptr` and `layout` must correspond to a valid allocation created by this arena
    pub unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        let cursor_ptr = self.cursor_ptr.get();
        if ptr == cursor_ptr {
            let new_cursor_ptr = unsafe {
                // Safety: a valid pointer and a layout to an allocation can not create overflow conditions
                align_up_non_null_with_provenance_unchecked(cursor_ptr.add(layout.size()), layout.align())
            };
            self.cursor_ptr.set(new_cursor_ptr);
        }
    }

    /// Attempts to grow an allocation from `old_layout` to `new_layout`.
    ///
    /// See more documentation and safety requirements in [`Allocator::grow`] documentation
    ///
    /// [`Allocator::grow`]: core::alloc::Allocator::grow
    #[cfg(any(feature = "vec", feature = "allocator_api", feature = "allocator-api2"))]
    unsafe fn grow(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Option<NonNull<u8>> {
        let cursor_ptr = self.cursor_ptr.get();

        let old_size = old_layout.size();
        let new_size = new_layout.size();

        if old_layout.align() >= new_layout.align() && cursor_ptr == ptr {
            let layout = unsafe {
                // Safety: `Layout::align()` returns a value which is both a power of two and non-zero,
                // and `new_size - old_size` will always be less or equal to `new_size`, therefore
                // also less or equal `isize::MAX`
                Layout::from_size_align_unchecked(new_size - old_size, old_layout.align())
            };

            if let Some(new_ptr) = self.try_alloc_in(layout) {
                unsafe {
                    new_ptr.copy_from(ptr, old_size);
                }
                return Some(new_ptr);
            }
        }

        let new_ptr = self.try_alloc(new_layout)?;
        unsafe {
            // Safety: allocations do not overlap
            new_ptr.copy_from_nonoverlapping(ptr, old_size);
        }
        Some(new_ptr)
    }

    /// Attempts to shrink an allocation from `old_layout` to `new_layout`.
    ///
    /// See more documentation and safety requirements in [`Allocator::shrink`] documentation
    ///
    /// [`Allocator::shrink`]: core::alloc::Allocator::shrink
    #[cfg(any(feature = "vec", feature = "allocator_api", feature = "allocator-api2"))]
    unsafe fn shrink(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Option<NonNull<u8>> {
        if old_layout.align() < new_layout.align() {
            if unsafe { is_aligned_to_unchecked(ptr.addr().get(), new_layout.align()) } {
                return Some(ptr);
            }

            let new_ptr = self.try_alloc(new_layout)?;
            unsafe {
                // Safety: allocations do not overlap
                new_ptr.copy_from_nonoverlapping(ptr, new_layout.size());
            }
            return Some(new_ptr);
        }

        let cursor_ptr = self.cursor_ptr.get();

        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let delta = old_size - new_size;

        if cursor_ptr == ptr && delta >= old_size.div_ceil(2) {
            let new_ptr = unsafe { self.cursor_ptr.get().add(delta) };

            unsafe {
                // Safety: ranges do not overlap because we checked `delta` to be
                // at least 2 times smaller than `old_size`
                new_ptr.copy_from_nonoverlapping(ptr, new_size);
            }
            return Some(new_ptr);
        }
        Some(ptr)
    }

    // Helper methods

    /// Returns pointer to the current chunk, if it exists.
    ///
    /// `None` returned if this arena owns no chunks
    fn get_chunk_ptr(&self) -> Option<NonNull<ChunkFooter>> {
        self.current_chunk_ptr.get()
    }

    /// Resets all allocations of this [`RawArena`], deallocating all chunks
    /// except the largest one.
    ///
    /// # Safety
    ///
    /// The method itself is not unsafe, but previously allocated pointers are.
    /// In case of [`Arena`] lifetime is enforced by borrow checker, but since we're
    /// dealing with primitive pointer allocation here, dereferencing pointer
    /// to the old allocations is UB
    fn reset(&mut self) {
        let chunk = self.current_chunk_ptr.get();
        let Some(current_chunk_ptr) = chunk else {
            return
        };

        self.cursor_ptr.set(current_chunk_ptr.cast::<u8>());
        unsafe {
            // Safety: `current_chunk_ptr` is a valid `ChunkFooter` because it can be accessed
            // only within `RawArena` implementation
            let footer = current_chunk_ptr.as_ref();

            // Safety: caller guarantees to prevent access to deallocated memory
            dealloc_chunk_list(footer.previous_chunk_ptr.replace(None));
        }
    }
}

impl Drop for RawArena {
    fn drop(&mut self) {
        unsafe {
            // Safety: deallocated elements could not be accessed because the arena is dropped
            dealloc_chunk_list(self.current_chunk_ptr.get())
        }
    }
}

/// Creates a new chunk capable of storing at least `size` bytes aligned to `align`.
///
/// Returns pointers to the chunk's footer and the start of usable region.
///
/// # Safety
///
/// - `align` must be a power of two.
///
/// - `current_chunk_ptr` must point to a valid chunk
unsafe fn new_chunk(size: usize, align: usize, current_chunk_ptr: Option<NonNull<ChunkFooter>>) -> Option<(NonNull<ChunkFooter>, NonNull<u8>)> {
    debug_assert!(align.is_power_of_two());

    let align = max(align, CHUNK_ALIGNMENT);
    if align > isize::MAX as usize {
        return None;
    }

    let size_aligned = unsafe {
        // Safety:
        // - `align` is required to be a valid alignment.
        // - caller guarantees `size` to be less or equal to `isize::MAX`, so rounding
        //   to `align` which checked to be also less or equal to `isize::MAX` can not
        //   overflow
        align_to_unchecked(size, align)
    };

    // For smaller allocations, use next closest power of two, otherwise
    // round up to page size
    let chunk_size_with_overhead = if size_aligned > PAGE_SIZE {
        if size_aligned > isize::MAX as usize - 1 - OVERHEAD {
            return None;
        }

        // Safety: `PAGE_SIZE` can not overflow `usize` and `size_aligned` checked above
        unsafe { align_to_unchecked(size_aligned + OVERHEAD, PAGE_SIZE) }
    } else {
        (size_aligned + OVERHEAD).next_power_of_two()
    };
    let chunk_size = chunk_size_with_overhead - OVERHEAD;

    debug_assert!(size_aligned.is_multiple_of(CHUNK_ALIGNMENT));
    debug_assert!(chunk_size.is_multiple_of(CHUNK_ALIGNMENT));
    debug_assert!(chunk_size_with_overhead.is_multiple_of(CHUNK_ALIGNMENT));

    let layout = Layout::from_size_align(chunk_size + FOOTER_SIZE, align).ok()?;
    let ptr = unsafe {
        // Safety: `chunk_size_with_footer` is never zero
        NonNull::new(alloc(layout))?
    };

    let chunk_footer_ptr = unsafe {
        // Safety: `ptr` points to an allocation of `chunk_size + FOOTER_SIZE` bytes
        ptr.add(chunk_size).cast::<ChunkFooter>()
    };

    unsafe {
        // Safety: `ptr` is aligned to `CHUNK_ALIGNMENT`, which is equal to
        // alignment of `ChunkFooter`
        chunk_footer_ptr.write(ChunkFooter::new(ptr, layout, current_chunk_ptr));
    }

    Some((chunk_footer_ptr, ptr))
}

/// Deallocates all chunks in a list starting from `last_chunk_ptr`.
///
/// # Safety
///
/// - `Some(last_chunk_ptr)` must be a valid `ChunkFooter`.
///
/// - deallocated chunks (including their footers) must not be accessed after this call
#[inline]
unsafe fn dealloc_chunk_list(mut last_chunk_ptr: Option<NonNull<ChunkFooter>>) {
    while let Some(ptr) = last_chunk_ptr {
        let footer = unsafe { ptr.as_ref() };
        last_chunk_ptr = footer.previous_chunk_ptr.replace(None);

        unsafe {
            // Safety: `ptr` is a valid `ChunkFooter` and no access to `ptr`
            // fields performed after this call
            ChunkFooter::drop_from_ptr(ptr);
        }
    }
}

/// A drop guard for initializing an array of `T`, dropping `..len` range
/// when it gets dropped.
///
/// Used in [`Arena::alloc_slice_cloned_boxed`] and [`Arena::alloc_slice_with_boxed`] methods.
struct ArrayDropGuard<T> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T> ArrayDropGuard<T> {
    /// Creates a new [`ArrayDropGuard`].
    ///
    /// # Safety
    ///
    /// The guard must be dropped after the loop finishes initialization
    #[inline]
    unsafe fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            len: 0
        }
    }

    /// Increments the count of initialized elements of this [`ArrayDropGuard`].
    ///
    /// # Safety
    ///
    /// The last value must be initialized
    #[inline]
    unsafe fn increment(&mut self) {
        self.len += 1;
    }
}

impl<T> Drop for ArrayDropGuard<T> {
    #[cold]
    fn drop(&mut self) {
        unsafe {
            // Safety: caller guarantees that the guard would have been dropped if the loop
            // never panicked, so we assume that the panic appeared, and drop the elements
            drop_in_place(slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
        }
    }
}

/// An exit point of a program (or a thread) when an allocation failed
#[inline]
pub(crate) fn unwrap_alloc<T>(alloc: Option<T>) -> T {
    match alloc {
        Some(alloc) => alloc,
        None => panic_alloc()
    }
}

/// Cold path of [`unwrap_alloc`], invoking panic
#[inline(never)]
#[cold]
pub(crate) fn panic_alloc() -> ! {
    panic!("allocation failed")
}

/// An arena allocator, backed by the global allocator.
///
/// # Example
///
/// ```
/// use flintnsteel::Arena;
///
/// let mut arena = Arena::new();
///
/// assert_eq!(*arena.alloc(1), 1);
/// assert_eq!(arena.alloc_str_copied("NoDrops™"), "NoDrops™");
///
/// arena.reset();
///
/// let flipflop = arena.alloc(false);
/// assert!(!*flipflop);
///
/// *flipflop = true;
/// assert!(*flipflop);
/// ```
///
/// # Design
///
/// This arena is a downwards bump allocation, based on a list of heap-allocated chunks, each
/// linked with the previous one. Each chunk contains a small footer storing its metadata.
/// Overall design is comparably minimal, and can be extended using allocation flavors when needed.
///
/// Arena allocators provide fast allocations, but the data becomes static when a new allocation
/// created. Deallocating specific allocation is possible using [`Arena::dealloc`], which is
/// unsafe.
///
/// # Allocation flavors
///
/// The crate provides alternative types from [`alloc`] module, these are [`Box`], [`Rc`] and [`Vec`].
/// But, unlike the [`Vec`], boxes and reference counters are unique: they are integrated
/// within the arena's interface.
///
/// Allocation flavors are basically answer the following question: `How do you want to interact
/// with your allocation for a duration of its lifetime?`. Simple allocations like [`Arena::alloc`]
/// return mutable reference to a type, reference counted allocations return autonomous [`Rc`]
/// which you can pass freely, boxed allocations provide [`Drop`] implementation, but carry
/// a reference to the arena.
///
/// # Drop
///
/// Allocating types which rely on [`Drop`] implementation will result in a memory leak
/// because the arena does not track each allocation.
///
/// It's recommended either dropping allocations manually, or using allocation methods
/// which return [`Box`]
///
/// Example:
///
/// ```
/// use flintnsteel::Arena;
/// use std::sync::Mutex;
///
/// struct Bread {
///     mutex_bread: Mutex<bool>,
///     temperature: u32,
/// }
///
/// let arena = Arena::new();
/// let boxed = arena.alloc_boxed(Bread {
///     mutex_bread: Mutex::new(false), // This needs to be dropped
///     temperature: 50, // And that's not
/// });
///
/// // Box ensures that `mutex_bread` has been dropped
/// ```
///
/// # Allocation methods
///
/// The primary allocation methods come in two variants: panicking and non-panicking.
///
/// When `allocator_api` or `allocator-api2` features enabled, [`Arena`] will get
/// `Allocator` trait implementation, exposing `shrink` and `grow` methods alongside
/// the standarlone allocation interface
#[repr(transparent)]
#[derive(Debug)]
pub struct Arena {
    alloc: RawArena
}

// Public methods

impl Arena {
    /// Creates a new [`Arena`] without performing any allocations.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// assert_eq!(arena.size(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            alloc: RawArena::new()
        }
    }

    /// Creates a new [`Arena`] with pre-allocated `capacity`.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::with_capacity(8);
    /// assert!(arena.size() > 8);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        unwrap_alloc(Self::try_with_capacity(capacity))
    }

    /// Attempts to create a new [`Arena`] with pre-allocated `capacity`.
    ///
    /// See [`Arena::with_capacity`] documentation for an example
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Option<Self> {
        Some(Self { alloc: RawArena::try_with_capacity(capacity)? })
    }
    
    // Helper methods

    /// Calculates total size of allocated memory of this [`Arena`].
    ///
    /// This method traverses over the list of chunks, making it considerably
    /// expensive.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// arena.alloc(1);
    ///
    /// // The actual size will always be greater than usable space due to
    /// // chunk metadata overhead
    /// assert!(arena.size() > 1);
    /// ```
    pub fn size(&self) -> usize {
        let mut size = 0;
        let mut current_ptr = self.alloc.get_chunk_ptr();

        while let Some(ptr) = current_ptr {
            let footer = unsafe {
                // Safety: we guarantee `Arena` to hold a valid pointer in case of `Some`
                ptr.as_ref()
            };
            size += footer.layout.size();

            current_ptr = footer.previous_chunk_ptr.get();
        }

        size
    }
    
    /// Checks if `ptr` points to the last allocation in this [`Arena`].
    /// 
    /// # Example
    /// 
    /// ```
    /// use core::ptr::from_mut;
    ///
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    ///
    /// let first = arena.alloc(1);
    /// let second = arena.alloc(2);
    ///
    /// assert_eq!(arena.is_last_allocation(first as *mut i32 as *const u8), false);
    /// assert_eq!(arena.is_last_allocation(second as *mut i32 as *const u8), true);
    /// ```
    #[inline]
    pub fn is_last_allocation(&self, ptr: *const u8) -> bool {
        core::ptr::eq(self.alloc.cursor_ptr.get().as_ptr(), ptr)
    }

    // Allocation methods

    /// Allocates `value` inside this [`Arena`] and returns a mutable reference to it.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let value = arena.alloc(1);
    ///
    /// assert_eq!(*value, 1);
    ///
    /// *value = 3;
    ///
    /// assert_eq!(*value, 3);
    /// ```
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.alloc_with(|| value)
    }

    /// Attempts to allocate `value` inside this arena and returns a
    /// mutable reference to it. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc`] documentation for an example
    #[inline]
    pub fn try_alloc<T>(&self, value: T) -> Option<&mut T> {
        self.try_alloc_with(|| value)
    }

    /// Allocates a copy of `slice` inside this [`Arena`] and returns an exclusive reference to it.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// assert_eq!(arena.alloc_slice_cloned(&[1, 2, 3, 4, 5]), &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn alloc_slice_copied<T>(&self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        unwrap_alloc(self.try_alloc_slice_copied(slice))
    }

    /// Attempts to allocate a copy of `slice` inside this [`Arena`] and returns an
    /// exclusive reference to it. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_slice_copied`] documentation for an example.
    #[allow(clippy::mut_from_ref)]
    pub fn try_alloc_slice_copied<T>(&self, slice: &[T]) -> Option<&mut [T]>
    where
        T: Copy,
    {
        let layout = Layout::for_value(slice);
        let ptr = self.try_alloc_layout(layout)?.cast::<T>();

        unsafe {
            // Safety: `ptr` points to an allocation with the same layout as `[T]`
            copy_nonoverlapping(slice.as_ptr(), ptr.as_ptr(), slice.len());
        }

        Some(unsafe {
            // Safety: allocation is exclusive and the data is initialized
            from_raw_parts_mut(ptr.as_ptr(), slice.len())
        })
    }

    /// Allocates an array of cloned values of `slice` and returns an exclusive reference to it.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory or [`Clone`] implementation
    /// of `T` have panicked.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// #[derive(Debug, Clone, PartialEq)]
    /// struct Wrapper(usize);
    ///
    /// let arena = Arena::new();
    /// let slice = [Wrapper(0), Wrapper(1), Wrapper(2)];
    ///
    /// assert_eq!(arena.alloc_slice_cloned(&slice), &slice);
    /// ```
    pub fn alloc_slice_cloned<T>(&self, slice: &[T]) -> &mut [T]
    where
        T: Clone,
    {
        unwrap_alloc(self.try_alloc_slice_cloned(slice))
    }

    /// Attempts to allocate an array of cloned values of `slice` and returns
    /// an exclusive reference to it. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_slice_cloned`] documentation for an example.
    ///
    /// # Panics
    ///
    /// If [`Clone`] implementation of `T` have panicked
    #[allow(clippy::mut_from_ref)]
    pub fn try_alloc_slice_cloned<T>(&self, slice: &[T]) -> Option<&mut [T]>
    where
        T: Clone,
    {
        let layout = Layout::for_value(slice);
        let ptr = self.try_alloc_layout(layout)?.cast::<T>();

        for (idx, value) in slice.iter().cloned().enumerate() {
            unsafe {
                // Safety: `ptr` points to an allocation capable of holding `slice`, therefore
                // both addition of index and write of a value derived from `slice` are valid
                ptr.add(idx).write(value)
            }
        }

        Some(unsafe {
            // Safety: allocation is exclusive and the data is initialized
            from_raw_parts_mut(ptr.as_ptr(), slice.len())
        })
    }

    /// Allocates a copy of `str` and returns an exclusive reference to it.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// assert_eq!(arena.alloc_str_copied("Yep's"), "Yep's");
    /// ```
    #[inline]
    pub fn alloc_str_copied(&self, str: &str) -> &mut str {
        unwrap_alloc(self.try_alloc_str_copied(str))
    }

    /// Attempts to allocate a copy of `str` and returns an exclusive reference
    /// to it. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_str_copied`] documentation for an example
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn try_alloc_str_copied(&self, str: &str) -> Option<&mut str> {
        let bytes = self.try_alloc_slice_copied(str.as_bytes())?;
        Some(unsafe {
            // Safety: `str` is valid UTF-8, therefore its copy is also valid UTF-8
            from_utf8_unchecked_mut(bytes)
        })
    }

    /// Allocates `T` received from `f` closure inside this [`Arena`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// fn big_calculation() -> usize {
    ///     1 // Assume the name is accurate
    /// }
    ///
    /// let arena = Arena::new();
    /// let _ = arena.alloc_with(|| big_calculation());
    /// ```
    #[inline]
    pub fn alloc_with<F, T>(&self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        unwrap_alloc(self.try_alloc_with(f))
    }

    /// Attempts to allocate `T` received from `f` closure inside this [`Arena`].
    /// Returns `None` if allocation failed.
    ///
    /// See example in [`Arena::alloc_with`] documentation
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn try_alloc_with<F, T>(&self, f: F) -> Option<&mut T>
    where
        F: FnOnce() -> T
    {
        #[inline(always)]
        unsafe fn inner_writer<T, F>(ptr: NonNull<T>, f: F)
        where
            F: FnOnce() -> T,
        {
            // This function is translated as:
            // - Allocate space for a T on the stack.
            // - Call `f()` with the return value being put onto this stack space.
            // - memcpy from the stack to the heap.
            //
            // Ideally we want LLVM to always realize that doing a stack allocation is unnecessary and optimize
            // the code so it writes directly into the heap instead. It seems we get it to realize this most
            // consistently if we put this critical line into its own function instead of inlining it into the
            // surrounding code.
            unsafe { ptr.write(f()) };
        }

        let layout = Layout::new::<T>();
        let mut ptr = self.try_alloc_layout(layout)?.cast::<T>();
        unsafe {
            // Safety: `Arena::alloc_layout` guarantees to return a pointer which is valid
            // for all writes of `T`
            inner_writer(ptr, f)
        };

        Some(unsafe {
            // Safety: `ptr` initialized above
            ptr.as_mut()
        })
    }

    /// Allocates an array of `T` with `len` length and initializes its values
    /// with `f` closure.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let slice = arena.alloc_slice_with(|index| index * 2, 8);
    /// assert_eq!(
    ///     slice,
    ///     &[0, 2, 4, 6, 8, 10, 12, 14]
    /// );
    /// ```
    #[inline]
    pub fn alloc_slice_with<F, T>(&self, f: F, len: usize) -> &mut [T]
    where
        F: FnMut(usize) -> T,
    {
        unwrap_alloc(self.try_alloc_slice_with(f, len))
    }

    /// Attempts to allocate an array of `T` with `len` length and initializes its values
    /// with `f` closure. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_slice_with`] documentation for an example
    #[allow(clippy::mut_from_ref)]
    pub fn try_alloc_slice_with<F, T>(&self, mut f: F, len: usize) -> Option<&mut [T]>
    where
        F: FnMut(usize) -> T,
    {
        let layout = Layout::array::<T>(len).ok()?;
        let ptr = self.try_alloc_layout(layout)?.cast::<T>();

        for idx in 0..len {
            unsafe {
                // Safety: `idx` is incremented up to length of the array sitting in the allocation
                ptr.add(idx).write(f(idx))
            }
        }

        Some(unsafe {
            // Safety: the data initialized above
            from_raw_parts_mut(ptr.as_ptr(), len)
        })
    }

    /// Allocates `layout` inside this [`Arena`] and returns pointer to the allocation.
    ///
    /// See [`Arena::try_alloc_layout`] for guarantees.
    ///
    /// # Panics
    ///
    /// If global allocator failed to allocate enough space.
    ///
    /// # Example
    ///
    /// ```
    /// use core::alloc::Layout;
    /// use flintnsteel::Arena;
    ///
    /// #[derive(Debug, Default, PartialEq)]
    /// struct Loaf {
    ///     rarity: usize,
    ///     properties: [bool; 5]
    /// }
    ///
    /// let arena = Arena::new();
    /// let layout = Layout::new::<Loaf>();
    ///
    /// let ptr = arena.alloc_layout(layout).cast::<Loaf>();
    /// unsafe { ptr.write(Loaf::default()); }
    ///
    /// assert_eq!(unsafe { ptr.read() }, Loaf::default());
    #[inline]
    pub fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {
        unwrap_alloc(self.try_alloc_layout(layout))
    }

    /// Attempts to allocate `layout` inside this [`Arena`].
    ///
    /// See example of panicking version in [`Arena::try_alloc_layout`] documentation.
    ///
    /// # Guarantees
    ///
    /// Returned pointer is:
    ///
    /// - valid for all writes until [`Arena::reset()`] is called or
    ///   the arena is dropped.
    ///
    /// - aligned according to `layout` alignment.
    ///
    /// - uninitialized, i.e. not valid for reads unless it has been written to
    #[inline]
    pub fn try_alloc_layout(&self, layout: Layout) -> Option<NonNull<u8>> {
        self.alloc.try_alloc(layout)
    }

    /// Allocates a reference counted `value` inside this [`Arena`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// assert_eq!(*arena.alloc_rc(0), 0);
    /// ```
    #[inline]
    pub fn alloc_rc<T>(&self, value: T) -> Rc<T> {
        self.alloc_with_rc(|| value)
    }

    /// Attempts to allocate reference counted `value` inside this [`Arena`].
    /// Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_rc`] documentation for an example
    #[inline]
    pub fn try_alloc_rc<T>(&self, value: T) -> Option<Rc<T>> {
        self.try_alloc_with_rc(|| value)
    }

    /// Allocates a reference counted copy of `slice` inside this [`Arena`].
    /// 
    /// # Panics
    /// 
    /// If the global allocator failed to allocate enough memory
    #[inline]
    pub fn alloc_slice_copied_rc<T>(&self, slice: &[T]) -> Rc<[T]>
    where
        T: Copy,
    {
        unwrap_alloc(self.try_alloc_slice_copied_rc(slice))
    }

    /// Attempts to allocate a reference counted copy of `slice` inside this [`Arena`].
    /// Returns `None` if allocation failed
    #[inline]
    pub fn try_alloc_slice_copied_rc<T>(&self, slice: &[T]) -> Option<Rc<[T]>>
    where
        T: Copy,
    {
        let ptr = NonNull::from_mut(self.try_alloc_slice_copied(slice)?);

        Some(unsafe {
            // Safety: we've just allocated `ptr`
            Rc::new_unchecked(ptr, self.alloc.get_chunk_ptr())
        })
    }

    /// Allocates a reference counted clone of `slice` inside this [`Arena`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory or [`Clone`]
    /// implementation of `T` have panicked
    #[inline]
    pub fn alloc_slice_cloned_rc<T>(&self, slice: &[T]) -> Rc<[T]>
    where
        T: Clone,
    {
        unwrap_alloc(self.try_alloc_slice_cloned_rc(slice))
    }

    /// Attempts to allocate a reference counted clone of `slice` inside this [`Arena`].
    /// Returns `None` if allocation failed.
    /// 
    /// # Panics
    /// 
    /// If [`Clone`] implementation of `T` have panicked
    #[inline]
    pub fn try_alloc_slice_cloned_rc<T>(&self, slice: &[T]) -> Option<Rc<[T]>>
    where
        T: Clone,
    {
        let ptr = NonNull::from_mut(self.try_alloc_slice_cloned(slice)?);

        Some(unsafe {
            // Safety: we've just allocated `ptr`
            Rc::new_unchecked(ptr, self.alloc.get_chunk_ptr())
        })
    }
    
    /// Allocates a reference counted copy of `str` inside this [`Arena`].
    /// 
    /// # Panics
    /// 
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let rc = arena.alloc_str_copied_rc("Cat science");
    /// assert_eq!(&*rc, "Cat science");
    /// ```
    #[inline]
    pub fn alloc_str_copied_rc(&self, str: &str) -> Rc<str> {
        unwrap_alloc(self.try_alloc_str_copied_rc(str))
    }

    /// Attempts to allocate a reference counted copy of `str` inside this [`Arena`].
    /// Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_str_copied_rc`] documentation for an example
    #[inline]
    pub fn try_alloc_str_copied_rc(&self, str: &str) -> Option<Rc<str>> {
        let ptr = NonNull::from_mut(self.try_alloc_str_copied(str)?);

        Some(unsafe {
            // Safety: we've just allocated `ptr`
            Rc::new_unchecked(ptr, self.alloc.get_chunk_ptr())
        })
    }

    /// Allocates reference counted `T` received from `f` closure inside
    /// this [`Arena`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// assert_eq!(*arena.alloc_with_rc(|| 0), 0);
    /// ```
    #[inline]
    pub fn alloc_with_rc<T, F>(&self, f: F) -> Rc<T>
    where
        F: FnOnce() -> T,
    {
        unwrap_alloc(self.try_alloc_with_rc(f))
    }

    /// Attempts to allocate reference counted `T` received from `f` closure
    /// inside this [`Arena`]. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_with_rc`] documentation for an example
    #[inline]
    pub fn try_alloc_with_rc<T, F>(&self, f: F) -> Option<Rc<T>>
    where
        F: FnOnce() -> T
    {
        let ptr = NonNull::from_mut(self.try_alloc_with(f)?);

        Some(unsafe {
            // Safety: we've just allocated `ptr`
            Rc::new_unchecked(ptr, self.alloc.get_chunk_ptr())
        })
    }

    /// Allocates a reference counted array of `T` with `len` length and initializes
    /// it with values received from `f` closure.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory or `f` have panicked.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let rc = arena.alloc_slice_with_rc(|i| i / 2, 5);
    /// assert_eq!(&*rc, &[0, 0, 1, 1, 2])
    /// ```
    #[inline]
    pub fn alloc_slice_with_rc<T, F>(&self, f: F, len: usize) -> Rc<[T]>
    where
        F: FnMut(usize) -> T,
    {
        unwrap_alloc(self.try_alloc_slice_with_rc(f, len))
    }

    /// Attempts to allocate a reference counted array of `T` with `len` length and initialized
    /// it with values received from `f` closure. Returns `None` if allocation failed.
    /// 
    /// See [`Arena::alloc_slice_with_rc`] documentation for an example
    #[inline]
    pub fn try_alloc_slice_with_rc<T, F>(&self, f: F, len: usize) -> Option<Rc<[T]>>
    where
        F: FnMut(usize) -> T,
    {
        let ptr = NonNull::from_mut(self.try_alloc_slice_with(f, len)?);

        Some(unsafe {
            // Safety: we've just allocated `ptr`
            Rc::new_unchecked(ptr, self.alloc.get_chunk_ptr())
        })
    }

    /// Allocates `value` inside this [`Arena`] and puts it in a fresh [`Box`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let boxed = arena.alloc_boxed(0);
    /// assert_eq!(*boxed, 0);
    /// ```
    #[inline]
    pub fn alloc_boxed<T>(&self, value: T) -> Box<'_, T> {
        self.alloc_with_boxed(|| value)
    }

    /// Attempts to allocate `value` inside this [`Arena`] and puts it in
    /// a fresh [`Box`]. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_boxed`] documentation for an example
    #[inline]
    pub fn try_alloc_boxed<T>(&self, value: T) -> Option<Box<'_, T>> {
        self.try_alloc_with_boxed(|| value)
    }

    /// Allocates a copy of `slice` inside this [`Arena`] and puts it in a fresh [`Box`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory
    #[inline]
    pub fn alloc_slice_copied_boxed<T>(&self, slice: &[T]) -> Box<'_, [T]>
    where
        T: Copy,
    {
        unwrap_alloc(self.try_alloc_slice_copied_boxed(slice))
    }

    /// Attempts to allocate a copy of `slice` inside this [`Arena`] and puts it
    /// in a fresh [`Box`]. Returns `None` if allocator failed
    #[inline]
    pub fn try_alloc_slice_copied_boxed<T>(&self, slice: &[T]) -> Option<Box<'_, [T]>>
    where
        T: Copy,
    {
        let ptr = NonNull::from_mut(self.try_alloc_slice_copied(slice)?);

        Some(unsafe {
            // Safety: allocation has the same lifetime as self
            Box::from_raw(ptr)
        })
    }

    /// Allocates an array of cloned values of `slice` inside this [`Arena`] and puts
    /// it in a fresh [`Box`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough capacity or [`Clone`]
    /// implementation of `T` have panicked
    #[inline]
    pub fn alloc_slice_cloned_boxed<T>(&self, slice: &[T]) -> Box<'_, [T]>
    where
        T: Clone,
    {
        unwrap_alloc(self.try_alloc_slice_cloned_boxed(slice))
    }

    /// Attempts to allocate an array of cloned values of `slice` and puts it in
    /// a fresh [`Box`]. Returns `None` if allocation failed.
    ///
    /// # Panics
    ///
    /// If [`Clone`] implementation of `T` have panicked
    #[inline]
    pub fn try_alloc_slice_cloned_boxed<T>(&self, slice: &[T]) -> Option<Box<'_, [T]>>
    where
        T: Clone,
    {
        let layout = Layout::for_value(slice);
        let ptr = self.try_alloc_layout(layout)?.cast::<T>();

        let mut guard = unsafe { ArrayDropGuard::new(ptr) };
        for (idx, value) in slice.iter().cloned().enumerate() {
            unsafe {
                // Safety: `ptr` points to an allocation of `slice.len()` elements
                ptr.add(idx).write(value);
                guard.increment();
            }
        }
        drop(guard);

        Some(unsafe {
            // Safety: all values was initialized above
            Box::from_raw(NonNull::slice_from_raw_parts(ptr, slice.len()))
        })
    }

    /// Allocates a copy of `str` inside this [`Arena`] and puts it in a fresh [`Box`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let boxed = arena.alloc_str_copied_boxed("Drops allowed!");
    /// assert_eq!(&*boxed, "Drops allowed!");
    /// ```
    #[inline]
    pub fn alloc_str_copied_boxed(&self, str: &str) -> Box<'_, str> {
        unwrap_alloc(self.try_alloc_str_copied_boxed(str))
    }

    /// Attempts to allocate a copy of `str` inside this [`Arena`] and puts it in
    /// a fresh [`Box`]. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_str_copied_boxed`] documentation for an example
    #[inline]
    pub fn try_alloc_str_copied_boxed(&self, str: &str) -> Option<Box<'_, str>> {
        let ptr = NonNull::from_mut(self.try_alloc_str_copied(str)?);

        Some(unsafe {
            // Safety: allocation has the same lifetime as self
            Box::from_raw(ptr)
        })
    }

    /// Allocates `T` received from `f` closure and puts it in a fresh [`Box`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let boxed = arena.alloc_with_boxed(|| 0);
    /// assert_eq!(*boxed, 0);
    /// ```
    #[inline]
    pub fn alloc_with_boxed<T, F>(&self, f: F) -> Box<'_, T>
    where
        F: FnOnce() -> T,
    {
        unwrap_alloc(self.try_alloc_with_boxed(f))
    }

    /// Attempts to allocate `T` received from `f` closure and puts in
    /// in a fresh [`Box`]. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_with_boxed`] documentation for an example
    #[inline]
    pub fn try_alloc_with_boxed<T, F>(&self, f: F) -> Option<Box<'_, T>>
    where
        F: FnOnce() -> T,
    {
        let ptr = NonNull::from_mut(self.try_alloc_with(f)?);

        Some(unsafe {
            // Safety: allocation has the same lifetime as self
            Box::from_raw(ptr)
        })
    }

    /// Creates a new array of `T` with `len` length and initialized its values with `f` closure,
    /// then puts it in a fresh [`Box`].
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough memory of `f` have panicked.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let boxed = arena.alloc_slice_with_boxed(|i| i << 2, 5);
    /// assert_eq!(&*boxed, &[0, 4, 8, 12, 16]);
    /// ```
    #[inline]
    pub fn alloc_slice_with_boxed<T, F>(&self, f: F, len: usize) -> Box<'_, [T]>
    where
        F: FnMut(usize) -> T,
    {
        unwrap_alloc(self.try_alloc_slice_with_boxed(f, len))
    }

    /// Attempts to allocate an array of `T` with `len` length inside this [`Arena`] and initializes
    /// its values with `f` closure. Returns `None` if allocation failed.
    ///
    /// See [`Arena::alloc_slice_with_boxed`] documentation for an example.
    ///
    /// # Panics
    ///
    /// If `f` have panicked
    #[inline]
    pub fn try_alloc_slice_with_boxed<T, F>(&self, mut f: F, len: usize) -> Option<Box<'_, [T]>>
    where
        F: FnMut(usize) -> T,
    {
        let layout = Layout::array::<T>(len).ok()?;
        let ptr = self.try_alloc_layout(layout)?.cast::<T>();

        let mut guard = unsafe { ArrayDropGuard::new(ptr) };
        for i in 0..len {
            unsafe {
                // Safety: we increment `i` up to `len`
                ptr.add(i).write(f(i));
                guard.increment();
            }
        }
        drop(guard);

        Some(unsafe {
            // Safety: all values was initialized above
            Box::from_raw(NonNull::slice_from_raw_parts(ptr, len))
        })
    }

    /// Resets this arena by deallocating all chunk except the biggest one.
    ///
    /// All allocations must be treated as invalid after this method called. The mutable reference
    /// will enforce lifetime for non-layout allocations, but in the opposite case you must consider
    /// this note.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let mut arena = Arena::new();
    ///
    /// // Arena resized to fit at least 8 bytes
    /// let _ = arena.alloc_slice_copied(&[1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8]);
    /// arena.reset();
    /// // No resize needed, previous data removed
    /// let _ = arena.alloc(1u64);
    /// ```
    pub fn reset(&mut self) {
        self.alloc.reset();
    }

    /// Deallocates an allocation of this [`Arena`] associated with its `ptr` and `layout`.
    ///
    /// Occupied space will be freed only if the allocation is last in this arena.
    ///
    /// # Safety
    ///
    /// Both `ptr` and `layout` must correspond to a valid allocation acquired from this arena
    /// and the data must not be accessed after this call. Violation of this requirements
    /// will either produce UB or corrupt the arena.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    /// use core::alloc::Layout;
    /// use core::ptr::NonNull;
    ///
    /// let arena = Arena::new();
    ///
    /// let number = arena.alloc(1);
    /// let string = arena.alloc_str_copied("Nümberbass");
    ///
    /// assert_eq!(*number, 1);
    /// assert_eq!(string, "Nümberbass");
    ///
    /// let layout = Layout::for_value(string);
    /// let ptr = NonNull::from_mut(string).cast::<u8>();
    ///
    /// unsafe {
    ///     // Safety: `ptr` and `layout` correspond to a valid allocation
    ///     arena.dealloc(ptr, layout);
    /// }
    ///
    /// assert_eq!(*number, 1);
    /// ```
    pub unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {
            self.alloc.dealloc(ptr, layout);
        }
    }
}

// Private methods

impl Arena {
    /// See [`Allocator::grow`] documentation for safety requirements
    ///
    /// [`Allocator::grow`]: allocator_api2::alloc::Allocator::grow
    #[inline]
    #[cfg(feature = "vec")]
    pub(crate) unsafe fn grow(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Option<NonNull<u8>> {
        unsafe {
            self.alloc.grow(ptr, old_layout, new_layout)
        }
    }

    /// See [`Allocator::shrink`] documentation for safety requirements
    ///
    /// [`Allocator::shrink`]: allocator_api::alloc::Allocator::shrink
    #[inline]
    #[cfg(feature = "vec")]
    pub(crate) unsafe fn shrink(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Option<NonNull<u8>> {
        unsafe {
            self.alloc.shrink(ptr, old_layout, new_layout)
        }
    }
}

impl Default for Arena {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(feature = "allocator_api", feature = "allocator-api2"))]
macro_rules! impl_allocator_trait {
    ($trait:ty, $error:ty) => {
        #[doc(hidden)]
        const _: () = {
            use $error as _AllocError;

            unsafe impl $trait for &Arena {
                #[inline]
                fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, _AllocError> {
                    self.try_alloc_layout(layout)
                        .map(|ptr| NonNull::slice_from_raw_parts(ptr, layout.size()))
                        .ok_or(_AllocError)
                }

                unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
                    unsafe {
                        self.dealloc(ptr, layout);
                    }
                }

                unsafe fn shrink(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, _AllocError> {
                    unsafe {
                        self.alloc.shrink(ptr, old_layout, new_layout)
                            .map(|ptr| NonNull::slice_from_raw_parts(ptr, new_layout.size()))
                            .ok_or(_AllocError)
                    }
                }

                unsafe fn grow(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, _AllocError> {
                    unsafe {
                        self.alloc.grow(ptr, old_layout, new_layout)
                            .map(|ptr| NonNull::slice_from_raw_parts(ptr, new_layout.size()))
                            .ok_or(_AllocError)
                    }
                }
            }
        };
    };
}

#[cfg(feature = "allocator_api")]
impl_allocator_trait!(core::alloc::Allocator, core::alloc::AllocError);

#[cfg(feature = "allocator-api2")]
impl_allocator_trait!(allocator_api2::alloc::Allocator, allocator_api2::alloc::AllocError);