use core::ptr::{slice_from_raw_parts_mut, drop_in_place, NonNull};
use core::ops::{Deref, DerefMut};
use core::fmt::{Debug, Display};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::mem::forget;
use core::pin::Pin;

use crate::Arena;

/// A trait that allows explicit creation of duplicate value inside an [`Arena`].
///
/// See [`Clone`] documentation for more.
///
/// # Example
///
/// ```
/// use flintnsteel::Arena;
/// use flintnsteel::boxed::CloneIn;
///
/// let arena = Arena::new();
/// let boxed = arena.alloc_slice_with_boxed(|_| 5, 5);
///
/// assert_eq!(*boxed, [5, 5, 5, 5, 5]);
///
/// let cloned = boxed.clone_in(&arena);
///
/// assert_eq!(*cloned, [5, 5, 5, 5, 5]);
/// ```
pub trait CloneIn<'a> {
    type Cloned: 'a;

    /// Allocate a span of memory inside `arena` and copy a duplicate of `self` into it.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    /// use flintnsteel::boxed::CloneIn;
    ///
    /// let arena = Arena::new();
    /// let string = arena.alloc_str_copied("C'mon");
    ///
    /// let _ = string.clone_in(&arena);
    /// ```
    fn clone_in(&self, arena: &'a Arena) -> Self::Cloned;
}

impl<'a, T> CloneIn<'a> for Box<'a, T>
where
    T: Clone + 'a,
{
    type Cloned = Box<'a, T>;

    #[inline]
    fn clone_in(&self, arena: &'a Arena) -> Self::Cloned {
        arena.alloc_boxed((*self).clone())
    }
}

impl<'a, T> CloneIn<'a> for Box<'a, [T]>
where
    T: Clone + 'a,
{
    type Cloned = Box<'a, [T]>;

    #[inline]
    fn clone_in(&self, arena: &'a Arena) -> Self::Cloned {
        arena.alloc_slice_cloned_boxed(self)
    }
}

impl<'a> CloneIn<'a> for Box<'a, str> {
    type Cloned = Box<'a, str>;

    #[inline]
    fn clone_in(&self, arena: &'a Arena) -> Self::Cloned {
        arena.alloc_str_copied_boxed(self)
    }
}

impl<'a, T> CloneIn<'a> for T
where
    T: Clone + 'a,
{
    type Cloned = &'a mut T;

    #[inline]
    fn clone_in(&self, arena: &'a Arena) -> Self::Cloned {
        arena.alloc((*self).clone())
    }
}

impl<'a, T> CloneIn<'a> for [T]
where
    T: Clone + 'a,
{
    type Cloned = &'a mut [T];

    #[inline]
    fn clone_in(&self, arena: &'a Arena) -> Self::Cloned {
        arena.alloc_slice_cloned(self)
    }
}

impl<'a> CloneIn<'a> for str {
    type Cloned = &'a mut str;

    #[inline]
    fn clone_in(&self, arena: &'a Arena) -> Self::Cloned {
        arena.alloc_str_copied(self)
    }
}

/// A wrapper on top of some arena allocation, providing [`Drop`] implementation.
///
/// That's all! Box owns its allocation, calls drop on its value. You also can
/// pin a [`Box`] using [`Box::pin`] method. Unboxing a box will turn it back
/// to a reference to the allocation.
///
/// # Examples
///
/// ```rust,no_run
/// use flintnsteel::Arena;
/// use std::io::Write;
/// use std::fs::File;
///
/// let arena = Arena::new();
/// let mut file = arena.alloc_boxed(File::open("tinyme.txt").unwrap());
///
/// file.write_all("Drops™".as_bytes()).unwrap();
/// ```
///
/// Box carries `'a` lifetime bound to the allocation, so we need to drop it
/// before if we want to reset the arena:
///
/// ```
/// use flintnsteel::Arena;
///
/// let mut arena = Arena::new();
/// let boxed = arena.alloc_str_copied_boxed("Waiting for the allocator_api to become stabilized");
///
/// drop(boxed);
/// arena.reset();
/// ```
///
/// [`Box::pin`]: crate::boxed::Box::pin
/// [`Box`]: crate::boxed::Box
pub struct Box<'a, T: ?Sized> {
    ptr: NonNull<T>,
    _marker: PhantomData<(&'a (), T)>,
}

impl<'a, T: ?Sized> Box<'a, T> {
    /// Deconstructs this [`Box`] into pointer to the allocation.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let boxed = arena.alloc_boxed(0);
    ///
    /// let ptr = boxed.into_raw();
    /// assert_eq!(unsafe { ptr.read() }, 0);
    /// ```
    #[inline]
    pub const fn into_raw(self) -> NonNull<T> {
        let ptr = self.ptr;
        forget(self);

        ptr
    }

    /// Constructs a new [`Box`] from `ptr` pointing to an allocation.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid `T`.
    ///
    /// - allocation `ptr` points to must outlive newly constructed [`Box`]
    #[inline]
    pub const unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Unbox the value of this [`Box`], consuming it.
    ///
    /// Creating boxed allocation and unboxing it is basically the same as
    /// allocating a reference.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let boxed = arena.alloc_boxed(0);
    ///
    /// let unboxed = boxed.unbox();
    ///
    /// assert_eq!(*unboxed, 0);
    /// ```
    #[inline]
    pub const fn unbox(self) -> &'a mut T {
        let mut ptr = self.ptr;
        forget(self);

        unsafe {
            // Safety: we consume self, so no other mutable reference can
            // exist at this moment. `ptr` is never dangling
            ptr.as_mut()
        }
    }
    
    /// Pins `this`, creating a new [`Pin`].
    ///
    /// [`Box`] owns its allocation, and the `'a` lifetime blocks arena
    /// from dropping itself until `this` have been dropped
    #[inline]
    pub const fn pin(this: Self) -> Pin<Box<'a, T>> {
        unsafe {
            // Safety: `Box<T>` owns its value, and arena guarantees it to live for an entire
            // duration of 'a lifetime
            Pin::new_unchecked(this)
        }
    }

    /// Returns a `*const` pointer to the value
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns `*mut` pointer to the value
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<'a, T> Box<'a, MaybeUninit<T>> {
    /// Converts [`Box<MaybeUninit<T>>`] to [`Box<T>`].
    ///
    /// # Safety
    ///
    /// The value must be in initialized state, i.e. it must be a valid `T`
    #[inline]
    pub unsafe fn assume_init(self) -> Box<'a, T> {
        let ptr = self.into_raw();
        unsafe {
            // Safety: caller guarantees value to be initialized and `MaybeUninit<T>`
            // has the same layout as `T`
            Box::from_raw(ptr.cast::<T>())
        }
    }
}

impl<'a, T> Box<'a, [MaybeUninit<T>]> {
    /// Converts [`Box<[MaybeUninit<T>]>`] to [`Box<[T]>`].
    ///
    /// # Safety
    ///
    /// All values of array stored inside this [`Box`] must be initialized, i.e. valid `T`
    #[inline]
    pub unsafe fn assume_init(self) -> Box<'a, [T]> {
        let ptr = self.into_raw();
        let new_ptr = slice_from_raw_parts_mut(ptr.as_ptr().cast::<T>(), ptr.len());
        unsafe {
            // Safety: caller guarantees all values to be initialized, `MaybeUninit<T>` has the same
            // layout as `T`, `ptr` was derived from `NonNull`
            Box::from_raw(NonNull::new_unchecked(new_ptr))
        }
    }
}

impl<T: ?Sized> Deref for Box<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for Box<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: ?Sized> AsRef<T> for Box<'_, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<T: ?Sized> AsMut<T> for Box<'_, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<T: ?Sized> Borrow<T> for Box<'_, T> {
    #[inline]
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T: ?Sized + Display> Display for Box<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: ?Sized + Debug> Debug for Box<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: ?Sized + Hash> Hash for Box<'_, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Box<'_, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + Eq> Eq for Box<'_, T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Box<'_, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, other)
    }
}

impl<'a, T: ?Sized> Drop for Box<'a, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Safety: `ptr` is never dangling and safe access to the `ptr` is
            // impossible because we're in a drop method
            drop_in_place(self.ptr.as_ptr());
        }
    }
}