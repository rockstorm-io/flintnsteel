use crate::vec::drain::Drain;
use crate::vec::{CollectIn, Vec};

/// A splice iterator over a [`Vec`].
///
/// See [`Vec::splice`] documentation for more
pub struct Splice<'a, T, I>
where
    I: Iterator<Item = T> + 'a,
{
    pub(crate) drain: Drain<'a, T>,
    pub(crate) replace_with: I,
}

impl<'a, T, I> Iterator for Splice<'a, T, I>
where
    I: Iterator<Item = T> + 'a,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl<'a, T, I> DoubleEndedIterator for Splice<'a, T, I>
where
    I: Iterator<Item = T> + 'a,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

impl<'a, T, I> ExactSizeIterator for Splice<'a, T, I>
where
    I: Iterator<Item = T> + 'a,
{
}

impl<'a, T, I> Drop for Splice<'a, T, I>
where
    I: Iterator<Item = T> + 'a,
{
    fn drop(&mut self) {
        self.drain.by_ref().for_each(drop);
        self.drain.iter = [].iter();

        unsafe {
            let vec = self.drain.ptr.as_mut();
            let arena = vec.arena();

            if self.drain.tail_len == 0 {
                vec.extend(self.replace_with.by_ref());
                return;
            }

            if !self.drain.fill(&mut self.replace_with) {
                return;
            }

            let (lower_bound, _) = self.drain.size_hint();
            if lower_bound > 0 {
                self.drain.move_tail(lower_bound);
                if !self.drain.fill(&mut self.replace_with) {
                    return;
                }
            }

            let mut collected = self.replace_with.by_ref().collect_in::<Vec<_>>(arena).into_iter();
            if collected.len() > 0 {
                self.drain.move_tail(collected.len());
                self.drain.fill(&mut collected);
            }
        }
    }
}