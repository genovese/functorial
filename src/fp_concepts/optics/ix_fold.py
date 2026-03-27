"""IxFold and IxAffineFold — indexed read-only optics over multiple or optional foci.

An IxFold i s a extracts zero or more values of type a from a structure s,
each paired with an index of type i.  An IxAffineFold i s a extracts at most
one such pair.

Index accumulation follows the same _MISSING / _pack_index convention as
IxLens: the outermost action passes _MISSING as the initial accumulator;
each optic in a chain packs its own index into it, yielding a left-nested
pair for a chain of length > 1.

Dispatch:
  * IndexedForget  — indexed reading actions (ifold_map_of, icollect, ...)
  * Forget         — plain reading actions after cast_as(FOLD)  (collect, ...)

"""

from __future__  import annotations

from typing      import Callable

from ..foldable  import IndexedFoldable
from ..functions import Function, compose, fn_eval, identity
from ..list      import List
from ..monoids   import Endo, Monoid
from ..utils     import Collect

from .optic       import Optic, OpticIs, _MISSING, _pack_index
from .profunctors import Forget, IndexedForget

__all__ = [
    'IxFold',
    'IxAffineFold',
    'ifolded',
    'ifolding',
    'ifold_map_of',
    'iright_fold_of',
    'ileft_fold_of',
    'icollect',
]


class IxFold(Optic, optic_is=OpticIs.IX_FOLD):
    """A read-only indexed optic extracting zero or more indexed foci."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.IX_FOLD)


class IxAffineFold(Optic, optic_is=OpticIs.IX_AFFINE_FOLD):
    """A read-only indexed optic extracting at most one indexed focus."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.IX_AFFINE_FOLD)


#
# ifolded : IndexedFoldable f => IxFold (f i a) i a
#
# Analogous to folded, but requires IndexedFoldable (which provides ifold_map)
# rather than plain Foldable.  The structure's intrinsic indices are the optic's
# indices.
#
# Dispatches on the profunctor:
#   IndexedForget  -> uses s.ifold_map directly with index accumulation
#   Forget         -> uses s.ifold_map forgetting the index (cast_as(FOLD) path)
#

def _ifolded_fn(p):
    """Core dispatch for ifolded."""
    if isinstance(p, IndexedForget):
        f = IndexedForget.run(p)     # i_acc -> a -> r
        m = p._monoid                # pylint: disable=protected-access
        def fold_s(i_acc, s):
            return s.ifold_map(lambda ix, a: f(_pack_index(i_acc, ix), a), m)
        return IndexedForget(fold_s, m)

    if isinstance(p, Forget):
        f = Forget.run(p)
        m = p._monoid                # pylint: disable=protected-access
        return Forget(lambda s: s.ifold_map(lambda _ix, a: f(a), m), m)

    raise TypeError(
        f'ifolded: unsupported profunctor {type(p).__name__!r}; '
        f'use folded for plain Foldable structures'
    )


ifolded = IxFold(_ifolded_fn)


def ifolding(f: Callable) -> IxFold:
    """Builds an IxFold from a function s -> IndexedFoldable i a.

    ifolding :: IndexedFoldable f => (s -> f i a) -> IxFold i s a

    """
    def the_fold(p):
        if isinstance(p, IndexedForget):
            g = IndexedForget.run(p)
            m = p._monoid            # pylint: disable=protected-access
            def fold_s(i_acc, s):
                return f(s).ifold_map(lambda ix, a: g(_pack_index(i_acc, ix), a), m)
            return IndexedForget(fold_s, m)

        if isinstance(p, Forget):
            g = Forget.run(p)
            m = p._monoid            # pylint: disable=protected-access
            return Forget(lambda s: f(s).ifold_map(lambda _ix, a: g(a), m), m)

        raise TypeError(f'ifolding: unsupported profunctor {type(p).__name__!r}')

    return IxFold(the_fold)


#
# Indexed folding actions
#

def ifold_map_of(optic, f: Callable, monoid: Monoid = Collect) -> Function:
    """Maps each (index, focus) pair to a monoid value and combines them.

    ifold_map_of :: IxFold i s a -> (i -> a -> m) -> s -> m

    """
    p = IndexedForget(f, monoid)
    optic_f = optic.cast_as(OpticIs.IX_FOLD)
    g = IndexedForget.run(optic_f(p))    # i_acc -> s -> m
    return Function(lambda s: g(_MISSING, s))


def icollect(optic) -> Function:
    """Collects all (index, focus) pairs into a list.

    icollect :: IxFold i s a -> s -> [(i, a)]

    """
    return ifold_map_of(optic, lambda i, a: List.of((i, a)), Collect)


def iright_fold_of(optic, f: Callable, init) -> Function:
    """Right fold over indexed foci using an accumulating function.

    iright_fold_of :: IxFold i s a -> (i -> a -> r -> r) -> r -> s -> r

    """
    reduce = lambda i, a: lambda r: f(i, a, r)
    return Function(lambda s: fn_eval(s >> ifold_map_of(optic, reduce, Endo), init))


def ileft_fold_of(optic, f: Callable, init) -> Function:
    """Left fold over indexed foci using an accumulating function.

    ileft_fold_of :: IxFold i s a -> (r -> i -> a -> r) -> r -> s -> r

    Derived from iright_fold_of via the difference-list technique; correct
    but O(n) in stack depth. Override for strict efficiency if needed.

    """
    def reduce(i, a, cont):          # cont :: r -> r  (difference-list continuation)
        return lambda r: cont(f(r, i, a))
    return Function(lambda s: (s >> iright_fold_of(optic, reduce, identity))(init))
