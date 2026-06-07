use crate::Arena;

use core::ptr::{copy_nonoverlapping, NonNull};
use core::ops::{Deref, DerefMut};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::cmp::max;

extern crate alloc as _alloc;

use _alloc::alloc::{alloc, dealloc, Layout};

use std::sync::Mutex;

/// A heap-allocated stack structure, optimized for `push` and `pop` methods.
///
/// # Example
///
/// ```rust,compile_fail
/// use crate::pool::Stack;
///
/// let mut stack = Stack::new();
///
/// stack.push(1);
/// assert_eq!(1, stack.pop());
/// ```
#[derive(Debug)]
pub(crate) struct Stack<T> {
    ptr: NonNull<T>,
    cursor: NonNull<T>,
    end: NonNull<T>,

    // Hint the compiler to disable auto-implementing traits, i.e. `Arena`
    // is not `Sync` so `Stack<Arena>` should inherit this behavior
    _phantom: PhantomData<T>,
}

// Public methods

impl<T> Stack<T> {
    /// Constant assertion for ZST.
    ///
    /// Must be referenced in all constructors to be evaluated at compile time
    const ASSERT: () = {
        assert!(size_of::<T>() != 0, "ZST types are not supported");
    };

    /// Creates a new [`Stack`] without performing any allocations.
    ///
    /// # Example
    ///
    /// ```rust,compile_fail
    /// use crate::pool::Stack;
    ///
    /// let stack = Stack::new();
    /// assert_eq!(stack.cap(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        const { Self::ASSERT };
        Self {
            ptr: NonNull::dangling(),
            cursor: NonNull::dangling(),
            end: NonNull::dangling(),
            _phantom: Default::default(),
        }
    }

    /// Creates a new [`Stack`] with pre-allocated capacity.
    ///
    /// # Example
    ///
    /// ```rust,compile_fail
    /// use crate::pool::Stack;
    ///
    /// let stack = Stack::with_capacity(5);
    /// assert_eq!(stack.cap(), 5);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        const { Self::ASSERT };
        Self::try_with_capacity(capacity).expect("allocation failed")
    }

    /// Reserves capacity for at least `n` more elements.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough capacity or `n` is greater
    /// than `isize::MAX`, resulting in a layout error.
    ///
    /// # Example
    ///
    /// ```rust,compile_fail
    /// use crate::pool::Stack;
    ///
    /// let mut stack = Stack::new();
    /// assert_eq!(stack.cap(), 0);
    ///
    /// stack.reserve(3);
    /// assert!(stack.cap() >= 3);
    /// ```
    #[allow(dead_code)]
    pub fn reserve(&mut self, n: usize) {
        self.try_reserve(n).expect("failed to reserve capacity")
    }

    /// Pushes `value` to the end of this [`Stack`].
    ///
    /// See example in [`Stack`] documentation.
    ///
    /// # Panics
    ///
    /// If the global allocator failed to allocate enough capacity
    #[inline]
    pub fn push(&mut self, value: T) {
        if self.cursor == self.end {
            // We don't want `try_reserve` to be inlined competely in a cold branch
            // so additional function used to wrap inlined `try_reserve`
            // optimized to add `element`
            #[inline(never)]
            #[cold]
            fn reserve<T>(stack: &mut Stack<T>) {
                stack.try_reserve(1).expect("allocation failed");
            }

            reserve(self);
        }

        unsafe {
            self.cursor.write(value);
            self.cursor = self.cursor.add(1);
        }
    }

    /// Pops a value at the end of this [`Stack`].
    ///
    /// Returns `None` if this stack does not own any elements.
    ///
    /// # Example
    ///
    /// ```rust,compile_fail
    /// use crate::pool::Stack;
    ///
    /// let mut stack = Stack::new();
    ///
    /// stack.push(0);
    /// stack.push(1);
    /// stack.push(2);
    ///
    /// assert_eq!(stack.pop(), Some(2));
    /// assert_eq!(stack.pop(), Some(1));
    /// assert_eq!(stack.pop(), Some(0));
    /// assert_eq!(stack.pop, None);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len() == 0 {
            return None;
        }
        
        Some(unsafe { self.pop_unchecked() })
    }
    
    /// Pops a value at the end of this [`Stack`] without performing a bounds check.
    /// 
    /// Use [`Stack::pop`] if you're looking for a safe alternative.
    /// 
    /// # Safety
    /// 
    /// - the stack must own at least one element
    #[inline]
    pub unsafe fn pop_unchecked(&mut self) -> T {
        self.cursor = unsafe {
            // Safety: caller guarantees stack to have at least one element
            self.cursor.sub(1)
        };
        
        unsafe {
            // Safety: if stack has at least one element, `self.cursor` always
            // points to an initialized slot
            self.cursor.read()
        }
    }

    /// Returns an amount of elements in this [`Stack`].
    ///
    /// # Example
    ///
    /// ```rust,compile_fail
    /// use crate::pool::Stack;
    ///
    /// let mut stack = Stack::new();
    ///
    /// stack.push(0);
    /// assert_eq!(stack.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        debug_assert_eq!(NonNull::<T>::dangling(), NonNull::<T>::dangling());
        unsafe {
            // Safety: in case of uninitialized allocation, `NonNull::dangling()` always equals
            // to any other `NonNull::dangling` if they point to the same type. For initialized
            // allocations, we guarantee distance between `self.cursor` and `self.ptr` to be
            // multiple of `size_of::<T>()`
            self.cursor.offset_from_unsigned(self.ptr)
        }
    }

    /// Returns capacity of this [`Stack`]
    #[inline]
    pub fn cap(&self) -> usize {
        debug_assert_eq!(NonNull::<T>::dangling(), NonNull::<T>::dangling());
        unsafe {
            // Safety: see comment in `Stack::len()` method
            self.end.offset_from_unsigned(self.ptr)
        }
    }
}

// Private methods

impl<T> Stack<T> {
    /// Attempts to create a new [`Stack`] with pre-allocated `capacity`.
    ///
    /// Returns `None` if the global allocator failed to allocate enough space or `capacity`
    /// is equal to zero or greater than `isize::MAX`.
    #[inline]
    fn try_with_capacity(capacity: usize) -> Option<Self> {
        if capacity == 0 {
            return Some(Self::new());
        }

        // Check must be performed because `dealloc` method relies on the fact that
        // if `self.cap != 0` layout is valid and the allocation is successful
        let layout = Layout::array::<T>(capacity).ok()?;
        let ptr = unsafe {
            // Safety: `capacity` checked to be non-zero, therefore an array
            // of non-ZST `T` can not be zero
            alloc(layout)
        };

        let non_null = NonNull::new(ptr.cast::<T>())?;
        let end = unsafe { non_null.add(capacity) };

        Some(Self {
            ptr: non_null,
            cursor: non_null,
            end,
            _phantom: Default::default(),
        })
    }

    /// Attempts to reserve capacity to fit at least `n` more elements.
    ///
    /// Returns `None` if either the global allocator failed to allocate enough
    /// capacity or `size_of::<T>() * new_cap` is greater than `isize::MAX`.
    ///
    /// The new capacity is amortized, i.e. it might be greater than `n`
    #[inline]
    fn try_reserve(&mut self, n: usize) -> Option<()> {
        let new_cap = max(self.cap() * 2, n);
        if n == 0 {
            return Some(());
        }

        let layout = Layout::array::<T>(new_cap).ok()?;
        let ptr = unsafe {
            // Safety: `capacity` checked to be non-zero, therefore an array
            // of non-ZST `T` can not be zero
            alloc(layout)
        };
        let non_null = NonNull::new(ptr.cast::<T>())?;

        let bytes_len = unsafe {
            // Safety: distance between `self.cursor` and `self.ptr` is multiple to `size_of::<T>()`
            // as these are aligned non-nulls, both point to a valid allocation
            self.cursor.byte_offset_from(self.ptr) as usize
        };

        unsafe {
            // Safety: allocations can not overlap because we haven't freed the current one yet
            copy_nonoverlapping(self.ptr.as_ptr().cast::<u8>(), ptr, bytes_len);
            // Safety: we will assign new pointer and capacity values after, therefore preventing
            // any access to the old allocation
            self.dealloc();
        }

        self.ptr = non_null;

        // Safety: `non_null` points to an allocation with size of `new_cap`
        // and `bytes_len` is less or equal to `new_cap`
        unsafe {
            self.cursor = non_null.add(bytes_len);
            self.end = non_null.add(new_cap);
        }

        Some(())
    }


    /// Deallocates an allocation owned by this [`Stack`].
    ///
    /// `cap`, `ptr` and `cursor` fields will remain untouched. Data is not copied or dropped.
    ///
    /// # Safety
    ///
    /// - after this method have been called, data located inside deallocated memory must not
    ///   be accessed.
    /// - this method must not be called again until this [`Stack`] gets a new allocation
    #[inline]
    unsafe fn dealloc(&mut self) {
        if self.cap() == 0 {
            return;
        }

        // Unsafe code here relies on:
        // - `ptr` is a valid pointer to an allocation and the caller guarantees to prevent
        //   double-free conditions.
        // - a layout derived from `cap * size_of::<T>()` is valid for the allocation and multiplication
        //   can not overflow.
        // These assumptions are guaranteed by other methods if `self.cap() != 0`.
        let layout = unsafe {
            Layout::from_size_align_unchecked(self.end.byte_offset_from_unsigned(self.ptr), align_of::<T>())
        };

        unsafe {
            // Safety: caller guarantees to not call this method on deallocated pointer
            dealloc(self.ptr.as_ptr().cast(), layout);
        }
    }
}

impl<T> Drop for Stack<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Safety: exclusive reference acquired from `Drop::drop`, the check above ensures the stack
            // to not be empty, heap allocation isn't dropped yet
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len()));
        }
        
        unsafe {
            // Safety: this stack dropped after the function finishes, therefore no other
            // dealloc can be called after this one
            self.dealloc();
        }
    }
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Both implementations are safe because `Stack` does not have any interior
// mutability and operates similarly to `Vec`, which has the same `Send` and `Sync`
// implementations

unsafe impl<T: Send> Send for Stack<T> {}

unsafe impl<T: Sync> Sync for Stack<T> {}

/// A wrapper around claimed [`Arena`] from an [`ArenaPool`] instance.
///
/// Implementations of [`Deref`] and [`DerefMut`] traits allow to use this wrapper
/// as any other arena.
///
/// # Drop
///
/// This struct implements [`Drop`] trait, automatically reclaiming internal arena
/// into the origin pool. `'a` lifetime indicates lifetime of the origin pool.
///
/// # Example
///
/// ```
/// use flintnsteel::ArenaPool;
///
/// let mut pool = ArenaPool::new(5);
/// let guard = pool.get().unwrap();
///
/// // Now we can allocate things in the arena
/// let _ = guard.alloc_str_copied("Imagine waiting for Mutex lock to acquire");
///
/// // Reclaim arena back to the pool
/// drop(guard);
/// ```
#[derive(Debug)]
pub struct ArenaPoolGuard<'a> {
    arena: ManuallyDrop<Arena>,
    origin: &'a ArenaPool,
}

impl<'a> Deref for ArenaPoolGuard<'a> {
    type Target = Arena;

    fn deref(&self) -> &Self::Target {
        &self.arena
    }
}

impl<'a> DerefMut for ArenaPoolGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.arena
    }
}

impl<'a> Drop for ArenaPoolGuard<'a> {
    #[inline]
    fn drop(&mut self) {
        let arena = unsafe {
            ManuallyDrop::take(&mut self.arena)
        };
        self.origin.insert(arena);
    }
}

/// A wrapper around `!Send` stack implementing [`Send`] unsafely.
///
/// [`Arena`] is not [`Send`] because it can allocate [`Rc`], but in our case each
/// thread gets its own arena, which is reset after the thread gets dropped.
#[derive(Debug, Default)]
struct ArenaStack {
    stack: Stack<Arena>,
}

// Safety: each thread gets its own arena, so no `Rc` can exist when a thread
// acquires an arena. For references and boxed, borrow lifetime prevents from sending arena.
// Note: this is strongly dependent on `ArenaPool` implementation and public interface
unsafe impl Send for ArenaStack {}

/// A pool of arenas for thread-safe reuse.
///
/// This struct provides a centralized way to reuse memory allocations across thread
/// boundaries internally using [`Mutex`] and a stack of arenas, initialized to serve
/// a fixed amount of threads in a single allocation.
///
/// # Example
///
/// ```
/// use flintnsteel::{Arena, ArenaPool};
/// use std::sync::Arc;
/// use std::thread;
///
/// let pool = Arc::new(ArenaPool::new(1));
///
/// let cloned = pool.clone();
/// let handle = thread::spawn(move || {
///     let arena = cloned.get().unwrap();
///     let rc = arena.alloc_rc(0);
///
///     assert_eq!(*rc, 0);
///     assert_eq!(*rc.clone(), 0);
///
///     // Guard dropped here
/// });
///
/// // Wait for the thread to finish and return its arena back to the pool
/// handle.join().unwrap();
///
/// // Acquire the same arena because `pool` was initialized with only one
/// let arena = pool.get().unwrap();
/// let str = arena.alloc_str_copied("Hello!");
/// assert_eq!(str, "Hello!");
/// ```
#[derive(Debug, Default)]
pub struct ArenaPool {
    pool: Mutex<ArenaStack>
}

impl ArenaPool {
    /// Constructs a new [`ArenaPool`] which holds `size` amount of arenas.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::ArenaPool;
    ///
    /// let mut pool = ArenaPool::new(2);
    ///
    /// let first = pool.get();
    /// assert!(first.is_some());
    ///
    /// let second = pool.get();
    /// assert!(second.is_some());
    ///
    /// assert!(pool.get().is_none());
    /// ```
    #[must_use]
    pub fn new(size: usize) -> Self {
        let mut stack = Stack::with_capacity(size);
        for _ in 0..size {
            stack.push(Arena::new());
        }

        Self {
            pool: Mutex::new(ArenaStack { stack })
        }
    }

    /// Returns an available arena of this [`ArenaPool`].
    ///
    /// `None` returned if this pool doesn't own any arenas
    #[inline]
    pub fn get(&self) -> Option<ArenaPoolGuard<'_>> {
        let arena = self.pool.lock().unwrap().stack.pop()?;
        let guard = ArenaPoolGuard {
            arena: ManuallyDrop::new(arena),
            origin: self
        };

        Some(guard)
    }

    /// Returns an arena from this [`ArenaPool`] if it's not empty, otherwise
    /// creates a new one.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::ArenaPool;
    ///
    /// let mut pool = ArenaPool::new(0);
    /// assert!(pool.get().is_none());
    ///
    /// let guard = pool.get_or_insert();
    /// drop(guard);
    ///
    /// assert!(pool.get().is_some());
    /// ```
    #[must_use]
    #[inline]
    pub fn get_or_insert(&self) -> ArenaPoolGuard<'_> {
        if let Some(arena) = self.get() {
            arena
        } else {
            ArenaPoolGuard {
                arena: ManuallyDrop::new(Arena::new()),
                origin: self
            }
        }
    }

    /// Inserts `arena` into this [`ArenaPool`], making it accessible via [`ArenaPool::get`].
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::{Arena, ArenaPool};
    ///
    /// let mut pool = ArenaPool::new(0);
    ///
    /// assert!(pool.get().is_none());
    /// pool.insert(Arena::new());
    /// assert!(pool.get().is_some());
    /// ```
    #[inline]
    pub fn insert(&self, mut arena: Arena) {
        arena.reset();

        let mut lock = self.pool.lock().ok().unwrap();
        lock.stack.push(arena);
    }
}