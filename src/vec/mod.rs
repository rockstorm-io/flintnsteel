use core::ptr::{NonNull, copy_nonoverlapping, drop_in_place, slice_from_raw_parts_mut};
use core::slice::{Iter, IterMut, from_raw_parts, from_raw_parts_mut, SliceIndex};
use core::ops::{Bound, Deref, DerefMut, RangeBounds, Index, IndexMut};
use core::fmt::{Formatter, Display, Debug};
use core::mem::{MaybeUninit, forget};
use core::hint::assert_unchecked;
use core::cmp::{Ordering, max};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::iter::IntoIterator;
use core::alloc::Layout;
use core::ptr::copy;

use crate::vec::into_iter::IntoIter;
use crate::vec::splice::Splice;
use crate::vec::drain::Drain;
use crate::{Arena, panic_alloc};
use crate::boxed::{Box, CloneIn};

pub mod into_iter;
pub mod splice;
pub mod drain;

// We don't have a separate test module because doctests fulfill this requirement.
// The `Vec` was rewritten based on stdlib 1.95.0 API and implementation

/// Creates a [`Vec`] containing the arguments.
///
/// `vec_in!` allows to define vectors with the same syntax as array
/// expressions. There are two forms of the macro:
///
/// Create a [`Vec`] containing a given list of elements:
///
/// ```
/// use flintnsteel::{Arena, vec_in};
///
/// let arena = Arena::new();
/// let vec = vec_in!(&arena; 1, 2, 3);
///
/// assert_eq!(vec[0], 1);
/// assert_eq!(vec[1], 2);
/// assert_eq!(vec[2], 3);
/// ```
///
/// Create a [`Vec`] from a given element and size:
///
/// ```
/// use flintnsteel::{Arena, vec_in};
///
/// let arena = Arena::new();
/// let vec = vec_in!(&arena; 1; 3);
/// assert_eq!(&*vec, &[1, 1, 1]);
/// ```
///
/// Note that unline array expressions this syntax supports all elements which
/// implement [`Clone`], and the number of elements doesn't have to be
/// constant
#[macro_export]
macro_rules! vec_in {
    ($arena:expr) => {{ $crate::vec::Vec::new_in($arena) }};
    ($arena:expr; $elem:expr; $n:expr) => {{
        let n = $n;
        let mut vec = $crate::vec::Vec::with_capacity_in(n, $arena);
        vec.extend_with(n, $elem);
        vec
    }};
    ($arena:expr; $($x:expr),*) => {{
        let mut vec = $crate::vec::Vec::new_in($arena);
        $(
            vec.push($x);
        )*
        vec
    }};
}

/// Prints a message indicating out of bounds index and panics.
///
/// Used in [`range`] function
#[cold]
#[inline(never)]
fn slice_assert_fail(idx: usize, len: usize) -> ! {
    panic!("range index {} out of range of slice {} length", idx, len);
}

/// Start and end indexes of a range.
///
/// See [`range`] function documentation
pub(crate) struct Range {
    pub(crate) start: usize,
    pub(crate) end: usize
}

/// Converts `range` into start and end indexes.
///
/// This function is an alternative to [`slice::range`] function which is exposed
/// behind a nightly feature.
///
/// # Panics
///
/// If a range index is out of bounds of a slice of `len` length.
///
/// # Example
///
/// ```compile_fail
/// use crate::vec::range;
///
/// assert_eq!(range(0..3, 3), Range { start: 0, end: 2 });
/// assert_eq!(range(1..=3, 4), Range { start: 1, end: 3 });
/// ```
#[inline]
pub(crate) fn range<R: RangeBounds<usize>>(range: R, len: usize) -> Range {
    let start = match range.start_bound() {
        Bound::Excluded(start) if *start >= len => slice_assert_fail(*start, len),
        Bound::Excluded(start) => *start + 1,
        Bound::Included(start) if *start > len => slice_assert_fail(*start + 1, len),
        Bound::Included(start) => *start,
        Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        Bound::Included(end) if *end >= len => slice_assert_fail(*end, len),
        Bound::Included(end) => *end + 1,
        Bound::Excluded(end) if *end > len => slice_assert_fail(*end, len),
        Bound::Excluded(end) => *end,
        Bound::Unbounded => len,
    };

    Range { start, end }
}

/// Constant properties of a sized type.
///
/// This trait is a stable polyfill of [`core::mem::SizedTypeProperties`]
pub(crate) trait SizedTypeProperties<T>: Sized {
    /// Size of this type, in bytes
    const SIZE: usize = size_of::<T>();

    /// Alignment of this type, in bytes
    const ALIGN: usize = align_of::<T>();

    /// `true` if this type is zero-sized, i.e. require no space to exist, `false` if
    /// the size is greater than zero
    const IS_ZST: bool = Self::SIZE == 0;

    /// Maximum length of `[T]` which won't overflow `isize::MAX`
    const MAX_SLICE_LEN: usize = match Self::SIZE {
        0 => usize::MAX,
        n => (isize::MAX as usize) / n,
    };
}

impl<T> SizedTypeProperties<T> for T {}

/// An allocation error returned by fallible [`Vec`] methods working
/// with the arena allocator
#[derive(Debug, Default, Copy, Clone)]
pub struct AllocError;

impl Display for AllocError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "allocation failed")
    }
}

/// A contiguous growable array, living inside an [`Arena`] allocator.
///
/// # Drop
///
/// The [`Vec`] struct will drop its elements when it gets dropped, but it never frees
/// its allocation.
///
/// # Example
///
/// ```
/// use flintnsteel::vec::Vec;
/// use flintnsteel::Arena;
///
/// let arena = Arena::new();
/// let mut vec = Vec::new_in(&arena);
///
/// vec.push(1);
/// vec.push(2);
///
/// assert_eq!(vec.pop(), Some(2));
/// assert_eq!(vec[0], 1);
/// ```
///
/// For more examples see [`std::vec::Vec`].
///
/// # Reallocation
///
/// When using any method of this struct which might grow the allocation, you must
/// consider its cost. Growing a [`Vec`] already owning some memory span will either create
/// a fresh new allocation inside the arena and **don't** free the previous one, or extending
/// the current one in-place if it's the last inside the arena. And that's expensive because
/// the memory is leaked until [`Arena::reset`] is called.
///
/// The easiest solution to that is to use [`Vec::with_capacity_in`] with sufficient
/// capacity for expected memory usage.
///
/// # Dynamic slice
///
/// Bump arena allocators are naturally hostile to frequent growth because the data is
/// packed tightly. This struct is primarily a dynamic alternative to `&mut [T]` you would
/// get from [`Arena::alloc_slice_with`] and similar method
///
/// Creating a static slice from a [`Vec`] can be achieved using [`Vec::into_slice`]
/// or [`Vec::into_boxed_slice`] methods (the second one is for the cases when you still
/// want to `Drop` your items).
///
/// Similarly to `&mut [T]`, a [`Vec`] bound to the lifetime of the arena allocator
/// it's lives in. This reference to the arena is stored inside the [`Vec`] and
/// can be acquired using [`Vec::arena`] method.
///
/// # Comparison with `std::vec::Vec`
///
/// When `allocator_api` feature is enabled or if you use `allocator-api2` crate,
/// the [`std::vec::Vec`] will allow user to define an allocator. [`Arena`] allocator
/// does implement [`Allocator`] trait required for this extension, but the features
/// are still limited.
///
/// This struct is a better alternative to the above scenario because it doesn't
/// require `allocator-api2` dependency for stable environment and integrates
/// with [`Arena`] by default.
///
/// Additionally, [`FromIterator`] trait can not be used for an arena-based vector,
/// so the [`Vec`] provides its own alternative called [`FromIteratorIn`]
pub struct Vec<'a, T> {
    arena: &'a Arena,
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

impl<'a, T> Vec<'a, T> {
    /// Constructs a new, empty [`Vec`] inside the `arena` allocator.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// #![allow(unused_mut)]
    ///
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec: Vec<'_, i32> = Vec::new_in(&arena);
    /// ```
    #[inline]
    pub fn new_in(arena: &'a Arena) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            arena
        }
    }

    /// Constructs a new, empty [`Vec`] inside `arena` allocator with
    /// at least the specified `capacity`.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the vector will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = Vec::with_capacity_in(10, &arena);
    ///
    /// // The vector contains no items, even though it has capacity for more
    /// assert_eq!(vec.len(), 0);
    /// assert!(vec.capacity() >= 10);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     vec.push(i);
    /// }
    /// assert_eq!(vec.len(), 10);
    /// assert!(vec.capacity() >= 10);
    ///
    /// // ...but this may make the vector reallocate
    /// vec.push(11);
    /// assert_eq!(vec.len(), 11);
    /// assert!(vec.capacity() >= 11);
    ///
    /// // A vector of a zero-sized type will always over-allocate, since no
    /// // allocation is necessary
    /// let vec_units = Vec::<()>::with_capacity_in(10, &arena);
    /// assert_eq!(vec_units.capacity(), usize::MAX);
    #[inline]
    pub fn with_capacity_in(capacity: usize, arena: &'a Arena) -> Self {
        Self::try_with_capacity_in(capacity, arena).unwrap()
    }

    /// Attempts to construct a new, empty [`Vec`] inside `arena` allocator with
    /// at least the specified `capacity`.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the vector will not allocate.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the arena allocator failed to allocate enough capacity
    /// or `capacity` exceeds `isize::MAX`
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, arena: &'a Arena) -> Result<Self, AllocError> {
        if capacity == 0 {
            Ok(Self::new_in(arena))
        } else {
            Self::try_alloc_in(capacity, arena).ok_or(AllocError)
        }
    }

    /// Creates a new [`Vec`] directly from a pointer, a length, a capacity,
    /// and an arena allocator.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// - `ptr` must point to a valid allocation inside `arena`.
    /// - `T` needs to have the same alignment as what `ptr` was allocated with.
    /// - The size of `T` times the `capacity` needs to be the same size as the pointer
    ///   was allocated with.
    /// - `length` needs to be less than or equal to `capacity`.
    /// - The first `length` values must be properly initialized values of type `T`.
    /// - `capacity` needs to fit the layout size that the pointer was allocated with.
    /// - The allocated size in bytes must be no larger than `isize::MAX`.
    ///
    /// These requirements are always upheld by any `ptr` that has been allocated
    /// via [`Vec`]. Other allocation sources are allowed if the invariants are
    /// upheld.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `Vec<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function
    #[inline]
    pub const unsafe fn from_raw_parts_in(ptr: NonNull<T>, len: usize, cap: usize, arena: &'a Arena) -> Self {
        Self {
            arena,
            ptr,
            len,
            cap
        }
    }

    /// Decomposes this [`Vec`] into its raw components: `(NonNull, len, cap, &Arena)`.
    ///
    /// Returns the `NonNull` pointer to the underlying data, the length of the vector
    /// (in elements), the allocated capacity of the data (in elements), and the arena
    /// allocator. These are the same arguments in the same order as
    /// the arguments to [`Vec::from_raw_parts_in`].
    ///
    /// After calling this method, the caller is responsible for managing
    /// the allocation and elements previously owned by the vector
    #[inline]
    pub const fn into_raw_parts_with_arena(self) -> (NonNull<T>, usize, usize, &'a Arena) {
        let ptr = self.ptr;
        let len = self.len;
        let cap = self.cap;
        let arena = self.arena;

        forget(self);

        (ptr, len, cap, arena)
    }

    /// Converts this [`Vec`] into [`Box`].
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let vec = vec_in![&arena; 1, 2, 3];
    ///
    /// let slice = vec.into_boxed_slice();
    /// ```
    #[inline]
    pub fn into_boxed_slice(self) -> Box<'a, [T]> {
        unsafe {
            let ptr = slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len);
            forget(self);

            // Safety: `ptr` can not be null because `self.ptr` is `NonNull`
            let non_null = NonNull::new_unchecked(ptr);

            // Safety: we guarantee `self.ptr` to be aligned, valid for all reads and
            // writes of `self.len * size_of::<T>()` bytes
            Box::from_raw(non_null)
        }
    }

    /// Converts this [`Vec`] into a slice containing its contents.
    ///
    /// Lifetime of returned slice bound to the arena allocator.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = Vec::new_in(&arena);
    ///
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// let slice = vec.into_slice();
    /// assert_eq!(slice, &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn into_slice(mut self) -> &'a mut [T] {
        let ptr = self.as_mut_ptr();
        let len = self.len();

        forget(self);

        unsafe {
            // Safety: we guarantee `..len` elements to be initialized, pointer will not be dangling
            // if `len` is not zero. The elements will not be dropped because we called `forget(self)`
            from_raw_parts_mut(ptr, len)
        }
    }

    /// Drops the contents of this [`Vec`] and deallocates its allocation.
    ///
    /// By default, [`Drop`] implementation will not deallocate contents of a [`Vec`]
    /// for a performance improvement. But if you know that the allocation this [`Vec`]
    /// owns is the last, this method can be used to reclaim the memory back to the arena.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let vec = Vec::<u8>::with_capacity_in(8, &arena);
    ///
    /// vec.dealloc();
    ///
    /// // This will be stored in the same place where the vector was
    /// let _ = arena.alloc(0u64);
    /// ```
    ///
    /// An example where calling [`Arena::dealloc`] would have no effect:
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let vec = Vec::<i32>::with_capacity_in(4, &arena);
    ///
    /// let _ = arena.alloc(0);
    ///
    /// // The allocation isn't the last because we allocated an i32 above,
    /// // so calling `.dealloc()` will just drop the elements as `Drop` does
    /// vec.dealloc();
    /// ```
    #[inline]
    pub fn dealloc(self) {
        let layout = self.calculate_layout();
        let ptr = self.ptr.cast::<u8>();
        let arena = self.arena;

        drop(self);

        unsafe {
            // Safety: `Vec` guarantees to have a valid pointer for non-zero length.
            // If the pointer is valid, UB will not be triggered because the length
            // is zero, therefore the cursor won't be incremented
            arena.dealloc(ptr, layout);
        }
    }

    /// Returns an immutable reference to the arena where this [`Vec`] lives.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let vec = Vec::<i32>::new_in(&arena);
    ///
    /// vec.arena().alloc_str_copied("Now we can allocate things!");
    /// ```
    #[inline]
    pub fn arena(&self) -> &'a Arena {
        self.arena
    }

    /// Appends `value` to the back of this [`Vec`].
    ///
    /// # Panics
    ///
    /// If the arena failed to allocate enough memory or the capacity exceeds `isize::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2];
    /// vec.push(3);
    /// assert_eq!(&*vec, &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity() {
            #[cold]
            fn push_grow_one<T>(vec: &mut Vec<'_, T>) {
                vec.try_grow_one().unwrap()
            }
            push_grow_one(self)
        }

        unsafe {
            // Safety: we've reserved capacity
            self.push_unchecked(value);
        }
    }

    /// Appends `value` to the back of this [`Vec`] without performing capacity check.
    ///
    /// # Safety
    ///
    /// Remaining capacity must not be zero, i.e. `self.len() < self.capacity()`
    #[inline]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(self.len < self.capacity());

        unsafe {
            // Safety: caller guarantees capacity to be sufficient
            self.ptr.add(self.len()).write(value);
        }
        self.len += 1;
    }

    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(&*vec, &[1, 2]);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        self.len -= 1;
        Some(unsafe {
            // Safety: the length was decremented
            assert_unchecked(self.len() < self.capacity());

            // Safety: we have at least one element
            self.ptr.add(self.len()).read()
        })
    }

    /// Removes and returns the last element from a vector if the predicate
    /// returns `true`, or [`None`] if the predicate returns false or the vector
    /// is empty (the predicate will not be called in that case).
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3, 4];
    /// let pred = |x: &mut i32| *x % 2 == 0;
    ///
    /// assert_eq!(vec.pop_if(pred), Some(4));
    /// assert_eq!(&*vec, &[1, 2, 3]);
    /// assert_eq!(vec.pop_if(pred), None);
    /// ```
    #[inline]
    pub fn pop_if<F>(&mut self, pred: F) -> Option<T>
    where
        F: FnOnce(&mut T) -> bool,
    {
        let last = self.last_mut()?;
        if pred(last) { self.pop() } else { None }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted.
    ///
    /// The vector may reserve more space to speculatively avoid frequent
    /// reallocations. After calling `reserve`, capacity will be greater
    /// than or equal to `self.len() + additional`.
    ///
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// If the arena allocator failed to allocate enough capacity
    /// or the new capacity exceeds `isize::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1];
    /// vec.reserve(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        if self.try_reserve(additional).is_err() {
            panic_alloc();
        }
    }

    /// Attempts to reserve capacity for at least `additional` more elements.
    ///
    /// The vector may reserve more space to speculatively avoid
    /// frequent reallocations. After calling `try_reserve`, capacity will be
    /// greater than or equal to `self.len() + additional` if it returns
    /// `Ok(())`.
    ///
    /// Does nothing if capacity is already sufficient. This method preserves the
    /// contents even if an error occurs.
    ///
    /// # Errors
    ///
    /// If the arena allocator failed to allocate enough capacity
    /// or the new capacity exceeds `isize::MAX`
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocError> {
        if self.needs_to_grow(additional) {
            #[cold]
            fn reserve_try_grow<T>(vec: &mut Vec<'_, T>, new_cap: usize) -> Result<(), AllocError> {
                let cap_amortized = vec.cap.saturating_mul(2);
                let new_cap = max(cap_amortized, new_cap);

                unsafe {
                    vec.try_grow_unchecked(new_cap)
                }
            }
            reserve_try_grow(self, self.len() + additional)?;
        }
        Ok(())
    }

    /// Reserves the minimum capacity for at least `additional` more elements.
    ///
    /// Unlike `reserve`, this will not deliberately over-allocate to speculatively
    /// avoid frequent allocations. After calling `reserve_exact`, capacity will be
    /// greater than or equal to`self.len() + additional`.
    ///
    /// Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// If the arena allocator failed to allocate enough capacity
    /// or the new capacity exceeds `isize::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1];
    /// vec.reserve_exact(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) {
        if self.try_reserve_exact(additional).is_err() {
            panic_alloc();
        }
    }

    /// Attempts to reserve the minimum capacity for at least `additional` elements.
    ///
    /// Unlike `try_reserve`, this will not deliberately over-allocate to speculatively
    /// avoid frequent allocations. After calling `try_reserve_exact`, capacity will be greater
    /// than or equal to `self.len() + additional` if it returns `Ok(())`.
    ///
    /// Does nothing if the capacity is already sufficient.
    ///
    /// # Errors
    ///
    /// If the arena allocator failed to allocate enough capacity
    /// or the new capacity exceeds `isize::MAX`
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), AllocError> {
        if self.needs_to_grow(additional) {
            #[cold]
            fn reserve_try_grow_exact<T>(vec: &mut Vec<'_, T>, new_cap: usize) -> Result<(), AllocError> {
                unsafe {
                    vec.try_grow_unchecked(new_cap)
                }
            }
            reserve_try_grow_exact(self, self.len() + additional)?;
        }
        Ok(())
    }

    /// Shrinks the capacity of this [`Vec`] to the `capacity` lower bound.
    ///
    /// The new capacity will remain at least large as the current length and specified value.
    ///
    /// # Panics
    ///
    /// If the arena allocator failed to shrink the allocation.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = Vec::with_capacity_in(10, &arena);
    ///
    /// vec.extend([1, 2, 3]);
    /// assert!(vec.capacity() >= 10);
    /// vec.shrink_to(4);
    /// assert!(vec.capacity() >= 4);
    /// vec.shrink_to(0);
    /// assert!(vec.capacity() >= 3);
    /// ```
    #[inline]
    pub fn shrink_to(&mut self, capacity: usize) {
        if self.try_shrink_to(capacity).is_err() {
            panic_alloc()
        }
    }

    /// Attempts to shrink the capacity of this [`Vec`] to the `capacity` lower bound.
    ///
    /// The new capacity will remain at least large as the current length and specified value.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the arena allocator failed to shrink the allocation
    #[inline]
    pub fn try_shrink_to(&mut self, capacity: usize) -> Result<(), AllocError> {
        if self.capacity() > capacity {
            unsafe {
                // Safety: we checked `capacity` to be less than `self.cap` and `self.len`
                // can't be greater than `self.cap`
                self.try_shrink_unchecked(max(capacity, self.len()))
            }
        } else {
            Ok(())
        }
    }

    /// Shrinks the capacity of this [`Vec`] as much as possible.
    ///
    /// # Panics
    ///
    /// If the arena allocator failed to shrink the allocation.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = Vec::with_capacity_in(10, &arena);
    ///
    /// vec.extend([1, 2, 3]);
    /// assert!(vec.capacity() >= 10);
    /// vec.shrink_to_fit();
    /// assert!(vec.capacity() >= 3);
    /// ```
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if self.try_shrink_to_fit().is_err() {
            panic_alloc()
        }
    }

    /// Attempts to shrink the capacity of this [`Vec`] as much as possible.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the arena allocator failed to shrink the allocation
    #[inline]
    pub fn try_shrink_to_fit(&mut self) -> Result<(), AllocError> {
        if self.capacity() > self.len() {
            unsafe {
                // Safety: `self.len` is always less or equal to `self.cap`
                self.try_shrink_unchecked(self.len())
            }
        } else {
            Ok(())
        }
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// If the arena allocation failed to grow the allocation in order
    /// to fit `other.len() * size_of::<T>()` bytes
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    ///
    /// let mut vec = vec_in![&arena; 1, 2, 3];
    /// let mut vec2 = vec_in![&arena; 4, 5, 6];
    /// vec.append(&mut vec2);
    /// assert_eq!(&*vec, &[1, 2, 3, 4, 5, 6]);
    /// assert_eq!(&*vec2, &[]);
    /// ```
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        self.reserve(other.len());
        unsafe {
            // Safety: elements are initialized, and we've reserved enough capacity to fit
            // at least `other.len` more elements
            copy_nonoverlapping(other.ptr.as_ptr(), self.ptr.add(self.len()).as_ptr(), other.len());
            self.len += other.len();

            // Safety: zero is always a valid length
            other.set_len(0);
        }
    }

    /// Shortens the vector, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len` is greater or equal to the vector's current length, this has
    /// no effect.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    ///
    /// # Examples
    ///
    /// Truncating a five element vector to two elements:
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3, 4, 5];
    /// vec.truncate(2);
    /// assert_eq!(&*vec, &[1, 2]);
    /// ```
    ///
    /// No truncation occurs when `len` is greater than the vector's current
    /// length:
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3];
    /// vec.truncate(8);
    /// assert_eq!(&*vec, &[1, 2, 3]);
    /// ```
    ///
    /// Truncating when `len == 0` is equivalent to calling [`Vec::clear`].
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3];
    /// vec.truncate(0);
    /// assert_eq!(&*vec, &[]);
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if self.len() < len {
            return;
        }

        unsafe {
            let to_drop = self.len() - len;
            // Safety: our pointer is nonnull, valid, aligned (because `Arena` guarantees alignment) and
            // elements in `self.len..self.len + to_drop` range are initialized
            let truncated_slice = from_raw_parts_mut(self.ptr.as_ptr().add(self.len()), to_drop);

            // Safety: `self.len..len` elements are initialized and we have an exclusive reference to `self`
            drop_in_place(truncated_slice);

            // Safety: `self.len` is greater or equal to `len`
            self.set_len(len);
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut v = vec_in![&arena; "foo", "bar", "baz", "qux"];
    ///
    /// assert_eq!(v.swap_remove(1), "bar");
    /// assert_eq!(&*v, &["foo", "qux", "baz"]);
    ///
    /// assert_eq!(v.swap_remove(0), "foo");
    /// assert_eq!(&*v, &["baz", "qux"]);
    /// ```
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(index < self.len(), "swap_remove index {} out of bounds", index);

        unsafe {
            // Safety: `index` was checked to not overflow `len` and we will
            // overwrite this index with a new value
            let value = self.as_ptr().add(index).read();
            let ptr = self.as_mut_ptr();

            // Safety: `index` was checked to not overflow `len`
            copy(ptr.add(self.len() - 1), ptr.add(index), 1);

            // Safety: we've checked `self.len()` to be greater than zero
            self.set_len(self.len() - 1);

            value
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 'a', 'b', 'c'];
    ///
    /// vec.insert(1, 'd');
    /// assert_eq!(&*vec, &['a', 'd', 'b', 'c']);
    /// vec.insert(4, 'e');
    /// assert_eq!(&*vec, &['a', 'd', 'b', 'c', 'e']);
    /// ```
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        self.insert_mut(index, value);
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right, and returning a reference to the new
    /// element.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    ///
    /// let mut vec = vec_in![&arena; 1, 3, 5, 9];
    /// let x = vec.insert_mut(3, 6);
    /// *x += 1;
    /// assert_eq!(&*vec, &[1, 3, 5, 7, 9]);
    /// ```
    #[inline]
    pub fn insert_mut(&mut self, index: usize, value: T) -> &mut T {
        let len = self.len();

        assert!(index <= len, "insert_mut index {} out of bounds", index);

        if len == self.capacity() {
            self.try_grow_one().unwrap();
        }

        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            if index < len {
                copy(ptr, ptr.add(1), len - index);
            }
            ptr.write(value);

            // Safety: we've shifted the elements by one
            self.set_len(len + 1);

            // Safety: value initialized above
            &mut *ptr
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut v = vec_in![&arena; 'a', 'b', 'c'];
    ///
    /// assert_eq!(v.remove(1), 'b');
    /// assert_eq!(&*v, &['a', 'c']);
    /// ```
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();

        assert!(index < len, "remove index {} out of bounds", index);

        let ptr = unsafe {
            // Safety: `index` checked to be within bounds
            self.as_mut_ptr().add(index)
        };

        unsafe {
            let value = ptr.read();

            // Safety: we've checked `index` to be within the length
            copy(ptr.add(1), ptr, len - index - 1);

            // Safety: decrementing a length is always safe and there is no leak
            // because we did copy the elements
            self.set_len(len - 1);

            value
        }
    }

    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated vector containing the elements in the range
    /// `[at, len)`. After the call, the original vector will be left containing
    /// the elements `[0, at)` with its previous capacity unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `at > len` or the arena allocator failed to create a new allocator for the other vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 'a', 'b', 'c'];
    /// let vec2 = vec.split_off(1);
    /// assert_eq!(&*vec, &['a']);
    /// assert_eq!(&*vec2, &['b', 'c']);
    /// ```
    #[must_use = "use .truncate() if you don't need the other part"]
    #[inline]
    pub fn split_off(&mut self, at: usize) -> Vec<'a, T> {
        let len = self.len();

        assert!(at <= len, "split `at` {} out of bounds", at);

        let other_len = len - at;
        let mut other = Vec::with_capacity_in(other_len, self.arena);

        unsafe {
            // Safety: `at` is checked to be less or equal to `len`
            self.set_len(at);
            // Safety: we will initialize the data
            other.set_len(other_len);

            // Safety: ranges do not overlap because `Arena` guarantees allocations to not overlap
            copy_nonoverlapping(self.as_ptr().add(at), other.as_mut_ptr(), other.len());
        }
        other
    }

    /// Resizes the [`Vec`] in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with the result of
    /// calling the closure `f`. The return values from `f` will end up
    /// in the `Vec` in the order they have been generated.
    ///
    /// If `new_len` is less than `len`, the `Vec` is simply truncated.
    ///
    /// # Panics
    ///
    /// Panics if the arena allocator failed to allocate enough capacity
    /// or `new_len` exceeds `isize::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    ///
    /// let mut vec = vec_in![&arena; 1, 2, 3];
    /// vec.resize_with(5, || 0);
    /// assert_eq!(&*vec, &[1, 2, 3, 0, 0]);
    /// ```
    #[inline]
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        let len = self.len();
        if new_len > len {
            self.extend(core::iter::repeat_with(f).take(new_len - len));
        } else {
            self.truncate(new_len);
        }
    }

    /// Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// # Panics
    ///
    /// Panics if the range has `start_bound > end_bound`, or, if the range is
    /// bounded on either end and past the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use flintnsteel::vec::{Vec, FromIteratorIn};
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3, 4];
    ///
    /// let new = [7, 8, 9];
    /// let collect = Vec::from_iter_in(vec.splice(1..3, new), &arena);
    /// assert_eq!(&*vec, &[1, 7, 8, 9, 4]);
    /// assert_eq!(&*collect, &[2, 3]);
    /// ```
    #[inline]
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, T, I::IntoIter>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        Splice {
            drain: self.drain(range),
            replace_with: replace_with.into_iter()
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3, 4];
    ///
    /// vec.retain(|&x| x % 2 == 0);
    /// assert_eq!(&*vec, &[2, 4]);
    /// ```
    ///
    /// Because the elements are visited exactly once in the original order,
    /// external state may be used to decide which elements to keep.
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3, 4, 5];
    ///
    /// let keep = [false, true, true, false, true];
    /// let mut iter = keep.iter();
    /// vec.retain(|_| *iter.next().unwrap());
    /// assert_eq!(&*vec, &[2, 3, 5]);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|a| f(a));
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    ///
    /// In other words, remove all elements `e` such that `f(&mut e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1, 2, 3, 4];
    ///
    /// vec.retain_mut(|x| if *x <= 3 {
    ///     *x += 1;
    ///     true
    /// } else {
    ///     false
    /// });
    /// assert_eq!(&*vec, &[2, 3, 4]);
    /// ```
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let len = self.len();
        if len == 0 {
            return;
        }

        struct DropGuard<'r, 'a, T> {
            vec: &'r mut Vec<'a, T>,
            original_len: usize,
            read: usize,
            write: usize,
        }

        impl<'r, 'a, T> Drop for DropGuard<'r, 'a, T> {
            #[cold]
            fn drop(&mut self) {
                let to_copy = self.original_len - self.read;
                unsafe {
                    // Safety: these elements aren't touched yet, and we guarantee
                    // `self.read` and `self.write` to be less than `original_len`
                    copy(
                        self.vec.as_ptr().add(self.read),
                        self.vec.as_mut_ptr().add(self.write),
                        to_copy,
                    );
                }

                unsafe {
                    // Safety: the `copy` invoke above will reorder the elements into contiguous array
                    self.vec.set_len(self.write + to_copy);
                }
            }
        }

        let mut read = 0;
        loop {
            let curr = unsafe {
                // Safety: we increment `read` up to `len`
                self.get_unchecked_mut(read)
            };

            #[cold]
            fn cold_path() {}

            if !f(curr) {
                cold_path();
                break;
            }

            read += 1;
            if read == len {
                // No elements to remove
                return;
            }
        }

        let mut guard = DropGuard {
            vec: self,
            original_len: len,
            read: read + 1,
            write: read
        };

        unsafe {
            // Safety: `read` is less than the length of `guard.vec`.
            // Note: I don't know why we dereference and create a new reference, but the original
            // implementation does that, so yeah. Maybe some provenance stuff :D
            drop_in_place(&mut *guard.vec.as_mut_ptr().add(read));
        }

        while guard.read < guard.original_len {
            let curr = unsafe {
                // Safety: we increment `guard.read` up to `guard.original_len`, which is
                // the length of `guard.vec`
                &mut *guard.vec.as_mut_ptr().add(guard.read)
            };
            if !f(curr) {
                // This must be done before dropping `curr` so if the `Drop` implementation panics,
                // we won't do a double drop
                guard.read += 1;
            } else {
                unsafe {
                    // Safety: we increment `guard.write` up to `guard.original_len`
                    let dst = guard.vec.as_mut_ptr().add(guard.write);

                    // Safety: `guard.read` is greater than `guard.write`
                    copy_nonoverlapping(curr, dst, 1);
                }
                guard.write += 1;
                guard.read += 1;
            }
        }

        unsafe {
            // Safety: `guard.write` is always less or equal to `guard.original_len`
            guard.vec.set_len(guard.write);
        }
        forget(guard);
    }

    /// Removes all but the first of consecutive elements in the vector that resolve to the same
    /// key.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 10, 20, 21, 30, 20];
    ///
    /// vec.dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(&*vec, &[10, 20, 30, 20]);
    /// ```
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality
    /// relation.
    ///
    /// The `f` function is passed references to two elements from the vector and
    /// must determine if the elements compare equal. The elements are passed in opposite order
    /// from their order in the slice, so if `f(a, b)` returns `true`, `a` is removed.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; "foo", "bar", "Bar", "baz", "bar"];
    ///
    /// vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(&*vec, &["foo", "bar", "baz", "bar"]);
    /// ```
    pub fn dedup_by<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        let len = self.len();
        if len <= 1 {
            return;
        }

        let start = self.as_mut_ptr();
        let mut first_duplicate_idx = 1;

        while first_duplicate_idx != len {
            let prev = unsafe {
                // Safety: `first_duplicate_idx` is initially one, and we're only increment it
                // up to `len`
                &mut *start.add(first_duplicate_idx.wrapping_sub(1))
            };
            let curr = unsafe {
                // Safety: `first_duplicate_idx` is incremented up to `len`
                &mut *start.add(first_duplicate_idx)
            };

            if f(curr, prev) {
                break;
            }
            first_duplicate_idx += 1;
        }

        // No elements to remove within the initialized part
        if first_duplicate_idx == len {
            return;
        }

        // If `f` have panicked, we're responsible for restoring the vector
        struct DropGuard<'r, 'a, T> {
            // Index of the element we want to read
            read: usize,
            // Index of the last non-duplicate element
            write: usize,
            // The vector we're working in
            vec: &'r mut Vec<'a, T>
        }

        impl<'r, 'a, T> Drop for DropGuard<'r, 'a, T> {
            fn drop(&mut self) {
                let ptr = self.vec.as_mut_ptr();
                let len = self.vec.len();

                let initialized = len.wrapping_sub(self.read);
                unsafe {
                    // Safety: we guarantee `self.read` and `self.write` to be less than `len`
                    copy(ptr.add(self.read), ptr.add(self.write), initialized);
                }

                let dropped = self.read.wrapping_sub(self.write);
                unsafe {
                    // Safety: `dropped` is less than `len` because we guarantee `self.read` to be
                    // greater than `self.write`, and truncating the length does not require
                    // any initialization
                    self.vec.set_len(len - dropped);
                }
            }
        }

        let mut guard = DropGuard {
            read: first_duplicate_idx + 1,
            write: first_duplicate_idx,
            vec: self
        };
        unsafe {
            // Safety: the loop will only increment `first_duplicate_idx` up to the length
            // of the vector. The guard will remove the element if this panics
            drop_in_place(start.add(first_duplicate_idx))
        }

        while guard.read < len {
            let curr_ptr = unsafe { start.add(guard.read) };
            let prev_ptr = unsafe { start.add(guard.write.wrapping_sub(1)) };

            unsafe {
                if f(&mut *curr_ptr, &mut *prev_ptr) {
                    guard.read += 1;

                    // Safety: `prev_ptr` is incremented by `guard.write - 1`, which is always
                    // greater or equal to 1. The index will be either overwritten or the new
                    // length will prevent further access
                    drop_in_place(prev_ptr);
                } else {
                    let dst = start.add(guard.write);

                    // Safety: we did skip at least one element in the loop checking
                    // for the first duplicate, therefore these can not overlap
                    copy_nonoverlapping(curr_ptr, dst, 1);

                    guard.write += 1;
                    guard.read += 1;
                }
            }
        }

        unsafe {
            // Safety: we increment `guard.write` up to `len`
            guard.vec.set_len(guard.write);
        }
        forget(guard);
    }

    /// Removes the subslice indicated by the given range from the vector,
    /// returning a double-ended iterator over the removed subslice.
    ///
    /// If the iterator is dropped before being fully consumed,
    /// it drops the remaining removed elements.
    ///
    /// # Panics
    ///
    /// Panics if the range has `start_bound > end_bound`, or, if the range is
    /// bounded on either end and past the length of the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::{Vec, FromIteratorIn};
    /// use flintnsteel::{Arena, vec_in};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut v = vec_in![&arena; 1, 2, 3];
    /// let u = Vec::from_iter_in(v.drain(1..), &arena);
    /// assert_eq!(&*v, &[1]);
    /// assert_eq!(&*u, &[2, 3]);
    ///
    /// // A full range clears the vector, like `clear()` does
    /// v.drain(..);
    /// assert_eq!(&*v, &[]);
    /// ```
    #[inline]
    pub fn drain<R>(&mut self, drain_range: R) -> Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let len = self.len();
        let Range { start, end } = range(drain_range, len);

        let drain_slice = unsafe {
            // Safety: `start <= len` because `range` function checks the bounds.
            // This is important to prevent access to uninitialized or moved out elements
            // of drained range if `Drain` destructor never gets to run
            self.set_len(start);

            // Safety: `range` checks both `start` and `end` to be within bounds of a slice
            // of `len` length
            from_raw_parts(self.as_ptr().add(start), end - start)
        };

        Drain {
            tail_start: end,
            tail_len: len - end,
            iter: drain_slice.iter(),
            ptr: NonNull::from_mut(self),
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut v = vec_in![&arena; 1, 2, 3];
    ///
    /// v.clear();
    ///
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        let to_drop = self.as_mut_slice() as *mut [T];
        unsafe {
            // Safety: zero is always a valid length
            self.set_len(0);

            // Safety: `to_drop` is valid because it was derived from `as_mut_slice()`. If `Drop`
            // implementation of `T` panics, remaining part will be leaked because we set
            // the length to zero
            drop_in_place(to_drop);
        }
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = Vec::new_in(&arena);
    /// assert!(vec.is_empty());
    ///
    /// vec.push(1);
    /// assert!(!vec.is_empty());
    /// ```
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let vec = vec_in![&arena; 1, 2, 3];
    /// assert_eq!(vec.len(), 3);
    /// ```
    #[inline]
    pub const fn len(&self) -> usize {
        let len = self.len;
        unsafe {
            // Safety: allocation can not be greater than `isize::MAX`
            assert_unchecked(len <= T::MAX_SLICE_LEN);
        }
        self.len
    }

    /// Returns the total number of elements the vector can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec: Vec<i32> = Vec::with_capacity_in(10, &arena);
    /// vec.push(42);
    /// assert!(vec.capacity() >= 10);
    /// ```
    ///
    /// A vector with zero-sized elements will always have a capacity of `usize::MAX`.
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// #[derive(Clone)]
    /// struct ZeroSized;
    ///
    /// fn main() {
    ///     assert_eq!(core::mem::size_of::<ZeroSized>(), 0);
    ///
    ///     let arena = Arena::new();
    ///     let v = vec_in![&arena; ZeroSized; 0];
    ///     assert_eq!(v.capacity(), usize::MAX);
    /// }
    /// ```
    #[inline]
    pub const fn capacity(&self) -> usize {
        if T::IS_ZST { usize::MAX } else { self.cap }
    }

    /// Returns a raw pointer to the allocation owned by this [`Vec`]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the allocation owned by this [`Vec`].
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe {
            // Safety: `..self.len` elements are initialized
            from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            // Safety: we have an exclusive reference to `self` and `..self.len` elements are initialized
            from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }

    /// Returns the remaining spare capacity of the vector as a slice of [`MaybeUninit`].
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`Vec::set_len`] method.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = Vec::with_capacity_in(10, &arena);
    ///
    /// // Fill in the first 3 elements.
    /// let uninit = vec.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// // Mark the first 3 elements of the vector as being initialized.
    /// unsafe {
    ///     vec.set_len(3);
    /// }
    ///
    /// assert_eq!(&*vec, &[0, 1, 2]);
    /// ```
    #[inline]
    pub const fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let spare_capacity_len = self.capacity() - self.len();
        unsafe {
            // Safety: `self.ptr` points to an allocation of `self.cap` elements, and we cast the
            // pointer to `MaybeUninit<T>`, so the elements are safe to be uninitialized
            from_raw_parts_mut(self.ptr.add(self.len()).as_ptr().cast::<MaybeUninit<T>>(), spare_capacity_len)
        }
    }

    /// Forces the length of this [`Vec`] to `new_len`.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less or equal to `self.capacity()`.
    ///
    /// - `..new_len` elements must be initialized
    #[inline]
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len;
    }
}

impl<'a, T: Clone> Vec<'a, T> {
    /// Creates a new [`Vec`] inside `arena` allocator and clones all values
    /// of `slice` onto it.
    ///
    /// # Panics
    ///
    /// If the arena allocator failed to allocate enough capacity or
    /// `Clone` implementation of `T` have panicked.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec::Vec;
    /// use flintnsteel::Arena;
    ///
    /// let slice = &[1, 2, 3, 4, 5];
    ///
    /// let arena = Arena::new();
    /// let vec = Vec::from_slice_in(slice, &arena);
    /// assert_eq!(&*vec, slice);
    /// ```
    #[inline]
    pub fn from_slice_in(slice: &[T], arena: &'a Arena) -> Self {
        let mut vec = Self::with_capacity_in(slice.len(), arena);
        vec.extend_from_slice(slice);

        vec
    }

    /// Resizes the [`Vec`] in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len`, the `Vec` is simply truncated.
    ///
    /// # Panics
    ///
    /// Panics if the arena allocator failed to allocate enough capacity
    /// or `new_len` exceeds `isize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    ///
    /// let mut vec = vec_in![&arena; "hello"];
    /// vec.resize(3, "world");
    /// assert_eq!(&*vec, &["hello", "world", "world"]);
    ///
    /// let mut vec = vec_in![&arena; 'a', 'b', 'c', 'd'];
    /// vec.resize(2, '_');
    /// assert_eq!(&*vec, &['a', 'b']);
    /// ```
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();
        if new_len > len {
            self.extend_with(new_len - len, value);
        } else {
            self.truncate(new_len);
        }
    }

    /// Clones and appends all elements in a slice to the [`Vec`].
    ///
    /// Iterates over the slice `other`, clones each element, and then appends
    /// it to this [`Vec`]. The `other` slice is traversed in-order.
    ///
    /// # Panics
    ///
    /// Panics if the arena allocator failed to allocate enough capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 1];
    /// vec.extend_from_slice(&[2, 3, 4]);
    /// assert_eq!(&*vec, &[1, 2, 3, 4]);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.extend(other.iter().cloned())
    }

    /// Given a range `src`, clones a slice of elements in that range and appends it to the end.
    ///
    /// `src` must be a range that can form a valid subslice of the [`Vec`].
    ///
    /// # Panics
    ///
    /// Panics if starting index is greater than the end index, if the index is
    /// greater than the length of the vector, or the arena allocator failed
    /// to allocate enough capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 0, 1, 2, 3, 4];
    ///
    /// vec.extend_from_within(..2);
    /// assert_eq!(&*vec, &[0, 1, 2, 3, 4, 0, 1]);
    /// ```
    pub fn extend_from_within<R: RangeBounds<usize>>(&mut self, src: R) {
        let Range { start, end } = range(src, self.len());

        let (initialized, uninitialized, len) = self.split_at_spare_capacity();
        let to_clone = &initialized[start..end];

        core::iter::zip(to_clone, uninitialized)
            .map(|(src, dst)| dst.write(src.clone()))
            .for_each(|_| *len += 1);
    }

    /// Extends this [`Vec`] with `n` more clones of `value`.
    ///
    /// # Panics
    ///
    /// If the arena allocator failed to allocate enough capacity
    /// or `n` exceeds `isize::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::vec_in;
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let mut vec = vec_in![&arena; 0, 1, 2];
    ///
    /// vec.extend_with(2, 3);
    /// assert_eq!(&*vec, &[0, 1, 2, 3, 3]);
    /// ```
    #[inline]
    pub fn extend_with(&mut self, n: usize, value: T) {
        self.reserve(n);

        for _ in 0..n.saturating_sub(1) {
            self.push(value.clone());
        }
        self.push(value);
    }
}

impl<'a, T: PartialEq> Vec<'a, T> {
    /// Removes consecutive repeated elements in the vector according to the
    /// [`PartialEq`] trait implementation.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Example
    ///
    /// ```
    /// let mut vec = vec![1, 2, 2, 3, 2];
    ///
    /// vec.dedup();
    ///
    /// assert_eq!(&*vec, &[1, 2, 3, 2]);
    /// ```
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|a, b| a == b)
    }
}

// Private methods

impl<'a, T> Vec<'a, T> {
    /// Returns `true` if this [`Vec`] is dangling, i.e. it doesn't point to a valid allocation.
    ///
    /// Dangling state is expected if the vector has zero capacity
    #[inline]
    const fn is_dangling(&self) -> bool {
        self.cap == 0
    }

    /// Returns `true` if `additional` more elements can not fit into remaining capacity
    /// of this [`Vec`], therefore we need to grow
    #[inline]
    const fn needs_to_grow(&self, additional: usize) -> bool {
        self.len() + additional > self.cap
    }

    /// Calculates [`Layout`] of the allocation this [`Vec`] is currently owning.
    ///
    /// The same as using [`Layout::from_size_align_unchecked`] but safe because we rely on
    /// [`Vec`]'s semantics
    #[inline]
    const fn calculate_layout(&self) -> Layout {
        let size = self.cap;
        unsafe {
            // Safety: already existing `Vec` will have `cap` which checked to
            // not overflow `isize::MAX` when multiplied to `T` size and aligned to `T` align
            Layout::from_size_align_unchecked(size.unchecked_mul(T::SIZE), T::ALIGN)
        }
    }

    /// Splits the initialized and uninitialized parts of the capacity and returns
    /// them with an additional mutable reference to `self.len`.
    ///
    /// This method is used in [`Vec::extend_from_within`]
    #[inline]
    const fn split_at_spare_capacity(&mut self) -> (&mut [T], &mut [MaybeUninit<T>], &mut usize) {
        let spare_capacity_len = self.capacity() - self.len;
        unsafe {
            let initialized = from_raw_parts_mut(self.ptr.as_ptr(), self.len);
            let uninitialized = from_raw_parts_mut(
                self.ptr.add(self.len()).cast::<MaybeUninit<T>>().as_ptr(),
                spare_capacity_len
            );

            (initialized, uninitialized, &mut self.len)
        }
    }

    /// Sets `self.ptr` and `self.cap` to specified values.
    ///
    /// # Safety
    ///
    /// `ptr` must point to an allocation of `cap * size_of::<T>()` bytes
    #[inline]
    const unsafe fn set_ptr_and_cap(&mut self, ptr: NonNull<T>, cap: usize) {
        self.ptr = ptr;
        self.cap = cap;
    }

    /// Attempts to create a new [`Vec`] with specified `capacity`.
    ///
    /// Returns `None` if the arena failed to allocate enough capacity
    /// or `capacity` exceeds `isize::MAX`
    #[inline]
    fn try_alloc_in(capacity: usize, arena: &'a Arena) -> Option<Self> {
        let layout = Layout::array::<T>(capacity).ok()?;
        let ptr = arena.try_alloc_layout(layout)?.cast::<T>();

        Some(Self { arena, ptr, len: 0, cap: capacity })
    }

    /// Attempts to grow this [`Vec`] by one. The growth is amortized
    #[inline]
    fn try_grow_one(&mut self) -> Result<(), AllocError> {
        unsafe {
            // Safety: the new capacity is greater by at least one
            self.try_grow_unchecked(self.cap * 2 + 1)
        }
    }

    /// Attempts to grow this [`Vec`] to `new_cap` capacity.
    ///
    /// If reallocation happens, this method will copy all existing contents to a new allocation.
    ///
    /// # Safety
    ///
    /// The current capacity must be less or equal to `new_cap`
    unsafe fn try_grow_unchecked(&mut self, new_cap: usize) -> Result<(), AllocError> {
        let new_layout = Layout::array::<T>(new_cap).map_err(|_| AllocError)?;
        
        let ptr = if self.is_dangling() {
            self.arena.try_alloc_layout(new_layout).ok_or(AllocError)?.cast::<T>()
        } else {
            let old_layout = self.calculate_layout();
            unsafe {
                // Safety: `old_layout` and `new_layout` both use the same generic
                assert_unchecked(old_layout.align() == new_layout.align());

                // Safety: `new_layout.size()` is greater than `old_layout.size()` since caller
                // guarantees `new_cap` to be greater or equal to the current capacity. `self.ptr` points
                // to a valid allocation because of `!self.is_dangling()` check
                let ptr = self.arena.grow(self.ptr.cast::<u8>(), old_layout, new_layout)
                    .ok_or(AllocError)?;
                ptr.cast::<T>()
            }
        };
        
        unsafe {
            // Safety: we did acquire `ptr` from the arena allocator
            self.set_ptr_and_cap(ptr, new_cap);
        }
        Ok(())
    }

    /// Attempts to shrink this [`Vec`] to `new_cap` capacity.
    ///
    /// # Safety
    ///
    /// The current capacity must be greater or equal to `new_cap`
    unsafe fn try_shrink_unchecked(&mut self, new_cap: usize) -> Result<(), AllocError> {
        let ptr = unsafe {
            let old_layout = self.calculate_layout();
            
            // Safety: the current layout is already larger, so a smaller one
            // can not overflow `isize::MAX` nor create an invalid `Layout`
            let new_size = new_cap.unchecked_mul(T::SIZE);
            let new_layout = Layout::from_size_align_unchecked(new_size, T::ALIGN);

            // Safety: `old_layout` and `new_layout` both use the same generic
            assert_unchecked(old_layout.align() == new_layout.align());

            // Safety: `self.ptr` is dangling only if `self.cap` is zero, but since caller guarantees
            // new capacity to be less than `self.cap`, we would have returned early if `self.cap` is zero
            self.arena.shrink(self.ptr.cast::<u8>(), old_layout, new_layout).ok_or(AllocError)?.cast::<T>()
        };
        
        unsafe {
            // Safety: we did acquire `ptr` from the arena allocator
            self.set_ptr_and_cap(ptr, new_cap);
        }
        Ok(())
    }
}

/// Conversion from an [`Iterator`] inside an [`Arena`].
///
/// By implementing [`FromIteratorIn`] for a type, you define how it will be created from an
/// iterator and an arena allocator. This is common for types which describe
/// a collection of some kind.
///
/// This trait is an extension trait of [`FromIterator`].
///
/// # Example
///
/// ```
/// use flintnsteel::vec::{Vec, FromIteratorIn};
/// use flintnsteel::{vec_in, Arena};
///
/// let five_fives = core::iter::repeat(5).take(5);
///
/// let arena = Arena::new();
/// let vec = Vec::from_iter_in(five_fives, &arena);
///
/// assert_eq!(vec, vec_in![&arena; 5, 5, 5, 5, 5])
/// ```
pub trait FromIteratorIn<'a, T>: Sized {
    fn from_iter_in<I>(iterator: I, arena: &'a Arena) -> Self
    where
        I: IntoIterator<Item = T>;
}

impl<'a, T> FromIteratorIn<'a, T> for Vec<'a, T> {
    #[inline]
    fn from_iter_in<I>(iterator: I, arena: &'a Arena) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let iter = iterator.into_iter();
        let (lower_bound, _) = iter.size_hint();
        
        let mut other = Self::with_capacity_in(lower_bound, arena);
        other.extend(iter);
        other
    }
}

/// A trait providing `.collect_in()` method, auto implemented for [`Iterator`].
///
/// # Example
///
/// ```
/// use flintnsteel::vec::{Vec, CollectIn};
/// use flintnsteel::Arena;
///
/// let arena = Arena::new();
/// let iter = core::iter::repeat(5).take(5);
///
/// let vec: Vec<_> = iter.collect_in(&arena);
/// assert_eq!(&*vec, &[5, 5, 5, 5, 5]);
/// ```
pub trait CollectIn {
    type Item;

    fn collect_in<'a, B>(self, arena: &'a Arena) -> B
    where
        B: FromIteratorIn<'a, Self::Item>;
}

impl<T> CollectIn for T
where
    T: Iterator,
{
    type Item = T::Item;

    fn collect_in<'a, B>(self, arena: &'a Arena) -> B
    where
        B: FromIteratorIn<'a, Self::Item>,
    {
        B::from_iter_in(self, arena)
    }
}

impl<'a, T> Extend<T> for Vec<'a, T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iterator = iter.into_iter();
        let size_hint = iterator.size_hint();

        self.reserve(size_hint.0);

        for item in iterator {
            self.push(item);
        }
    }
}

impl<'a, T> IntoIterator for Vec<'a, T> {
    type Item = T;
    type IntoIter = IntoIter<'a, T>;

    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        let len = self.len();
        let ptr = self.as_mut_ptr();
        let end = if T::IS_ZST {
            ptr.wrapping_byte_sub(len)
        } else {
            unsafe {
                // Safety: `Vec` guarantees `..len` to be initialized
                ptr.add(len) as *const T
            }
        };

        IntoIter {
            _phantom: PhantomData,
            end,
            ptr,
        }
    }
}

impl<'a, T> IntoIterator for &'a Vec<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vec<'a, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

impl<'a, T: Clone> Clone for Vec<'a, T> {
    fn clone(&self) -> Self {
        Vec::from_slice_in(self, self.arena)
    }
}

impl<'a, T> Deref for Vec<'a, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T> DerefMut for Vec<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a, T> AsRef<Vec<'a, T>> for Vec<'a, T> {
    #[inline]
    fn as_ref(&self) -> &Vec<'a, T> {
        self
    }
}

impl<'a, T> AsMut<Vec<'a, T>> for Vec<'a, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec<'a, T> {
        self
    }
}

impl<'a, T> AsRef<[T]> for Vec<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> AsMut<[T]> for Vec<'a, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T> From<Vec<'a, T>> for &'a mut [T] {
    #[inline]
    fn from(value: Vec<'a, T>) -> Self {
        value.into_slice()
    }
}

impl<'a, T> From<Vec<'a, T>> for Box<'a, [T]> {
    #[inline]
    fn from(value: Vec<'a, T>) -> Self {
        value.into_boxed_slice()
    }
}

impl<'a, T: Debug> Debug for Vec<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<'a, T: Hash> Hash for Vec<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

impl<'a, T, I: SliceIndex<[T]>> Index<I> for Vec<'a, T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl<'a, T, I: SliceIndex<[T]>> IndexMut<I> for Vec<'a, T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

impl<'a, T: PartialEq> PartialEq for Vec<'a, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<'a, T: PartialOrd> PartialOrd for Vec<'a, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<'a, T: Eq> Eq for Vec<'a, T> {}

impl<'a, T: Ord> Ord for Vec<'a, T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<'a, T> Drop for Vec<'a, T> {
    fn drop(&mut self) {
        unsafe {
            // Safety: we guarantee `..len` elements to be initialized and since we're
            // in a `Drop`, no access to the dropped contents of the unique allocation
            // of this vector can be performed
            drop_in_place(slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len()));
        }
    }
}