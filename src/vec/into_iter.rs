use core::iter::FusedIterator;
use core::marker::PhantomData;

use crate::vec::SizedTypeProperties;
use crate::Arena;

/// An iterator which moves out of vector.
/// 
/// This struct is created by [`Vec::into_iter`] method of [`IntoIterator`] trait
/// 
/// [`Vec::into_iter`]: crate::vec::Vec::into_iter
pub struct IntoIter<'a, T> {
    pub(crate) ptr: *const T,
    pub(crate) end: *const T,
    pub(crate) _phantom: PhantomData<(T, &'a Arena)>
}

impl<'a, T> Iterator for IntoIter<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            return None;
        }

        let ptr = if T::IS_ZST {
            self.end = self.end.wrapping_byte_sub(1);
            self.ptr
        } else {
            let old_ptr = self.ptr;
            self.ptr = unsafe {
                // Safety: we checked `self.ptr` to be less than `self.end`
                self.ptr.add(1)
            };
            old_ptr
        };

        Some(unsafe {
            // Safety: for ZST types we dereference an aligned `self.ptr`, for non-ZST
            // types, the check guarantees pointer arithmetic to produce a valid pointer
            // to the allocation
            ptr.read()
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = if T::IS_ZST {
            self.end.addr().wrapping_sub(self.ptr.addr())
        } else {
            unsafe {
                // For non-ZST types `self.end` is always greater or equal to `self.ptr`
                self.end.offset_from_unsigned(self.ptr)
            }
        };
        (exact, Some(exact))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T> DoubleEndedIterator for IntoIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            return None;
        }
        
        if T::IS_ZST {
            self.end = self.end.wrapping_byte_sub(1);
            Some(unsafe { self.ptr.read() })
        } else {
            Some(unsafe {
                // Safety: the check above will prevent out of bounds arithmetic
                self.end = self.end.sub(1);
                self.end.read()
            })
        }
    }
}

impl<'a, T> ExactSizeIterator for IntoIter<'a, T> {}

impl<'a, T> FusedIterator for IntoIter<'a, T> {}

impl<'a, T> Drop for IntoIter<'a, T> {
    fn drop(&mut self) {
        self.for_each(drop);
    }
}