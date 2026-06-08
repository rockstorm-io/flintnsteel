use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::ptr::NonNull;
use core::ops::Deref;
use core::pin::Pin;

use crate::ChunkFooter;

/// An immutable reference counted autonomous allocation.
///
/// This struct is a wrapper of top of some arena allocation enforcing
/// chunk-level reference counting. The reference counting blocks deallocation of the chunk
/// where allocation lives, making the wrapper unbound from arena's lifetime.
///
/// This implementation is **really** cheap because it only adds increment of the refcount when allocated,
/// and handling of chunk deallocation when dropped.
///
/// # No Drops
///
/// Underlying `T` type will not be dropped when a [`Rc`] is. This drawback comes with a performance
/// improvement because we don't need to store a centralized counter inside the allocation.
///
/// # Examples
///
/// Creating a shared reference to an allocation:
///
/// ```
/// use flintnsteel::Arena;
///
/// let arena = Arena::new();
/// let rc = arena.alloc_rc(0);
/// let cloned = rc.clone();
///
/// assert_eq!(*rc, *cloned);
/// ```
///
/// Outliving arena:
///
/// ```
/// use flintnsteel::Arena;
///
/// #[derive(Copy, Clone)]
/// struct Bank {
///     assets: usize,
/// }
///
/// let arena = Arena::new();
/// let bank = arena.alloc(Bank { assets: 0 });
/// assert_eq!(bank.assets, 0);
///
/// bank.assets += 1;
///
/// let snapshot = arena.alloc_rc(*bank);
/// drop(arena);
///
/// assert_eq!(snapshot.assets, 1);
/// ```
///
/// [`Rc`] doesn't have [`DerefMut`] implementation, but we can use the same pattern
/// as we would do with [`alloc:rc::Rc`]:
///
/// ```
/// use core::cell::RefCell;
///
/// use flintnsteel::Arena;
///
/// let arena = Arena::new();
/// let rc = arena.alloc_with_rc(|| RefCell::new(0));
///
/// *(*rc).borrow_mut() = 1;
///
/// assert_eq!(*rc.borrow(), 1);
/// ```
///
/// # Integration with [`Arena`]
///
/// Without introducing unsafe, [`Rc`] type can be created only using
/// interface provided by [`Arena`]. Such kind of design emerges because the arena
/// must promise us to enforce lifetime of a chunk.
///
/// Reference counting is cheap due to the nature of arena allocators: just don't
/// deallocate chunk unless the count is zero. But this behavior may result in
/// unexpected memory usage for certain situations. In we, for example, have a chunk
/// of 1 MiB and just a single 64 byte allocation blocks the chunk from deallocating
/// itself, this drawback would be really unpleasant (although such situation
/// is extremely rare).
/// 
/// In general, [`Rc`] should be used either when you can't freely pass lifetime of
/// the arena and can store these counters in a group, or you want shared access.
/// This type is a merge of both cases
///
/// [`DerefMut`]: core::ops::DerefMut
/// [`Arena`]: crate::Arena
pub struct Rc<T: ?Sized> {
    ptr: NonNull<T>,
    origin: Option<NonNull<ChunkFooter>>,

    // Makes `Rc<T>` covariant in `T`
    _marker: PhantomData<*const T>,
}

// Public methods

impl<T: ?Sized> Rc<T> {
    /// Constructs a new [`Rc`] from parts describing its owned allocation.
    ///
    /// # Safety
    ///
    /// Both `ptr` and `origin` must be derived from a [`Rc`] created before.
    ///
    /// # Example
    ///
    /// ```
    /// use flintnsteel::Arena;
    /// use flintnsteel::rc::Rc;
    ///
    /// let arena = Arena::new();
    /// let rc = arena.alloc_str_copied_rc("What do you even do with raw parts?");
    ///
    /// let (ptr, origin) = rc.into_raw_parts();
    /// let new_rc = unsafe {
    ///     Rc::from_raw_parts(ptr, origin)
    /// };
    ///
    /// assert_eq!(&*new_rc, "What do you even do with raw parts?");
    /// ```
    #[inline]
    pub const unsafe fn from_raw_parts(ptr: NonNull<T>, origin: Option<NonNull<u8>>) -> Self {
        let origin = if let Some(origin) = origin {
            Some(origin.cast::<ChunkFooter>())
        } else {
            None
        };

        Self {
            origin,
            ptr,
            _marker: PhantomData
        }
    }

    /// Deconstructs this [`Rc`] into parts associated with its allocation.
    ///
    /// Calling this method will not update the refcount, so a new [`Rc`] should
    /// be constructed to avoid memory leak.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use flintnsteel::Arena;
    ///
    /// let arena = Arena::new();
    /// let rc = arena.alloc_rc(0);
    ///
    /// let (ptr, origin) = rc.into_raw_parts();
    ///
    /// // Since we didn't construct a new `Rc`, memory is leaked
    /// ```
    #[inline]
    pub const fn into_raw_parts(self) -> (NonNull<T>, Option<NonNull<u8>>) {
        let ptr = self.ptr;
        let origin = if let Some(origin) = self.origin {
            Some(origin.cast::<u8>())
        } else {
            None
        };

        core::mem::forget(self);
        (ptr, origin)
    }

    /// Returns a `'static` reference to `T` pointed by this [`Pin`], leaking the whole
    /// chunk where it lives.
    ///
    /// The chunk that hosts the allocation will not be deallocated, so an actual memory
    /// leak will be larger than size of `T`. Any [`Rc`] referencing the same `T` will remain valid
    #[inline]
    pub const fn leak(self) -> &'static T {
        let leaked = unsafe {
            self.ptr.as_ref()
        };
        core::mem::forget(self);

        leaked
    }

    /// Pins `this`, creating a new [`Pin`].
    ///
    /// This is safe because [`Rc`] owns its allocation, preventing the drop completely
    /// until at least one clone of it exist
    #[inline]
    pub const fn pin(this: Self) -> Pin<Rc<T>> {
        unsafe {
            // Safety: `Box<T>` owns its value, and arena guarantees it to live for an entire
            // duration of 'a lifetime
            Pin::new_unchecked(this)
        }
    }

    /// Returns `*const` pointer to the underlying type
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns `*mut` pointer to the underlying type
    #[inline]
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

// Private methods

impl<T: ?Sized> Rc<T> {
    /// Creates a new [`Rc`] owning an allocation of `ptr` inside `origin` chunk.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid allocation of `T` inside the arena which owns `origin`.
    ///
    /// - `origin` must be **Some** if `ptr` points to a non-zero sized type.
    ///
    /// - data must remain valid until this [`Rc`] is dropped
    #[inline]
    pub(crate) unsafe fn new_unchecked(ptr: NonNull<T>, origin: Option<NonNull<ChunkFooter>>) -> Self {
        unsafe {
            // Safety: `ptr` is guaranteed to point to a valid `T`
            if size_of_val(ptr.as_ref()) != 0 {
                debug_assert!(origin.is_some());

                // Safety: caller guarantees `origin` to be `Some` for non-zero sized value
                ChunkFooter::increment_refcount(origin.unwrap_unchecked());
            }
        }

        Self { ptr, origin, _marker: Default::default() }
    }
}

impl<T: ?Sized> Deref for Rc<T> {
    type Target = T;
    
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> AsRef<T> for Rc<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<T: ?Sized> Borrow<T> for Rc<T> {
    #[inline]
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T: ?Sized> Clone for Rc<T> {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            // Safety: the allocation will remain valid because this `Rc` isn't dropped yet,
            // and the new `Rc` will increment refcount for itself
            Rc::new_unchecked(self.ptr, self.origin)
        }
    }
}

impl<T: ?Sized + Display> Display for Rc<T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: ?Sized + Debug> Debug for Rc<T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: ?Sized + Hash> Hash for Rc<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state)
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Rc<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + Eq> Eq for Rc<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Rc<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, other)
    }
}

impl<T: ?Sized> Unpin for Rc<T> {}

impl<T: ?Sized> Drop for Rc<T> {
    fn drop(&mut self) {
        // Reference counter is not incremented for zero-sized values. It is
        // safe to drop the chunk because dereferencing dangling pointer
        // to a ZST type is not UB
        if size_of_val(unsafe { self.ptr.as_ref() }) != 0 {
            debug_assert!(self.origin.is_some());
            unsafe {
                // Safety: pointer is valid because the constructor requires that, and
                // we will not access the data because we are in a drop method
                ChunkFooter::drop_from_ptr(self.origin.unwrap_unchecked());
            }
        }
    }
}