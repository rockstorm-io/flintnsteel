use core::ptr::{NonNull, copy, drop_in_place, read, slice_from_raw_parts_mut};
use core::slice::Iter;
use core::mem::take;
use core::slice::from_raw_parts_mut;
use std::ptr::write;
use crate::vec::{SizedTypeProperties, Vec};

/// A draining iterator over a [`Vec`].
///
/// See [`Vec::drain`] documentation for more
pub struct Drain<'a, T> {
    pub(crate) tail_start: usize,
    pub(crate) tail_len: usize,
    pub(crate) iter: Iter<'a, T>,
    pub(crate) ptr: NonNull<Vec<'a, T>>
}

impl<'a, T> Drain<'a, T> {
    /// Returns an immutable slice to the iterator over drained part
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.iter.as_slice()
    }
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|i| unsafe { read(i as *const T) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|i| unsafe { read(i as *const T) })
    }
}

impl<'a, T> AsRef<[T]> for Drain<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

// Safety: we don't provide any methods to access the arena directly from `Drain`

unsafe impl<'a, T: Send> Send for Drain<'a, T> {}
unsafe impl<'a, T: Sync> Sync for Drain<'a, T> {}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // Unsafe code here relies on: methods returning `Drain` take `&mut`,
        // `tail_len` equals to `vec.len() - tail_start`
        struct DropGuard<'i, 'a, T> {
            inner: &'i mut Drain<'a, T>,
        }

        impl<'i, 'a, T> Drop for DropGuard<'i, 'a, T> {
            fn drop(&mut self) {
                if self.inner.tail_len > 0 {
                    let vec = unsafe {
                        // Safety: `Vec`'s methods returning `Drain` will always require `&mut` reference
                        self.inner.ptr.as_mut()
                    };

                    let start = vec.len();
                    let tail = self.inner.tail_start;

                    if tail != start {
                        unsafe {
                            // Safety: Tail start can not overflow `vec` because we checked the range
                            // to be within the vector
                            let src = vec.as_ptr().add(tail);

                            // Safety: we increment the pointer by the length of its vector
                            let dst = vec.as_mut_ptr().add(start);

                            copy(src, dst, self.inner.tail_len);
                        }
                    }
                    unsafe {
                        // Safety: we've initialized the data above
                        vec.set_len(start + self.inner.tail_len);
                    }
                }
            }
        }

        let iter = take(&mut self.iter);
        let len = iter.len();

        let mut ptr = self.ptr;

        // For ZST types just drop drained part
        if T::IS_ZST {
            let vec = unsafe {
                // Safety: see `DropGuard::drop`
                self.ptr.as_mut()
            };

            unsafe {
                let vec_len = vec.len();
                let to_drop = slice_from_raw_parts_mut(
                    vec.as_mut_ptr().add(vec_len + self.tail_len),
                    len
                );

                // Safety: ZST types have no identity, and we're operating in a region of valid data
                drop_in_place(to_drop);

                // Safety: tail length is produced by subtracting vec's length and tail start
                vec.set_len(vec_len + self.tail_len);
            }

            return;
        }

        let _guard = DropGuard {
            inner: self,
        };

        if len == 0 {
            return;
        }

        let drop_ptr = iter.as_slice().as_ptr();
        unsafe {
            // The comment in the alloc crate implementation of `Drain` (from which this code originates)
            // suggests that a mutable provenance for a pointer is required for `drop_in_place`
            let vec_ptr = ptr.as_mut().as_mut_ptr();
            let offset = drop_ptr.offset_from_unsigned(vec_ptr);
            let to_drop = slice_from_raw_parts_mut(vec_ptr.add(offset), len);

            // Safety: DropGuard will set a new length so the dropped part
            // is either overwritten or inaccessible
            drop_in_place(to_drop);
        }

        // DropGuard will handle move of the undrained part
    }
}

// Helper methods for `Splice::drop`

impl<T> Drain<'_, T> {
    /// Fills the range between the start of the internal [`Vec`] and the tail start with
    /// elements acquired from `replace_with` iterator.
    /// 
    /// Returns `true` if `replace_with` successfully served fill of the whole
    /// range, `false` otherwise
    pub(crate) unsafe fn fill<I>(&mut self, replace_with: &mut I) -> bool
    where
        I: Iterator<Item = T>,
    {
        let vec = unsafe {
            // Safety: we have exclusive access to self
            self.ptr.as_mut()
        };
        let range_start = vec.len;
        let range_end = self.tail_start;
        let range_slice = unsafe {
            // Safety: we have exclusive reference to `vec` and `range_start` can not overflow
            // it because it equals to vec's length
            from_raw_parts_mut(vec.as_mut_ptr().add(range_start), range_end - range_start)
        };

        for place in range_slice {
            let Some(new_item) = replace_with.next() else {
                return false;
            };
            unsafe { write(place, new_item) };
            vec.len += 1;
        }
        true
    }
    
    /// Moves tail to fit `additional` more elements before it
    pub(crate) unsafe fn move_tail(&mut self, additional: usize) {
        let vec = unsafe {
            // Safety: we have exclusive access to self
            self.ptr.as_mut()
        };
        let len = self.tail_start + self.tail_len;
        vec.reserve(len + additional);

        let new_tail_start = self.tail_start + additional;
        unsafe {
            let src = vec.as_ptr().add(self.tail_start);
            let dst = vec.as_mut_ptr().add(new_tail_start);
            copy(src, dst, self.tail_len);
        }
        self.tail_start = new_tail_start;
    }
}