"""IxFold and IxAffineFold — indexed read-only optics over multiple or optional foci.

An IxFold i s a extracts zero or more values of type a from a structure s,
each paired with an index of type i.  An IxAffineFold i s a extracts at most
one such pair.

Index accumulation follows the same _MISSING / _pack_index convention as
IxLens: the outermost action passes _MISSING as the initial accumulator;
each optic in a chain packs its own index into it, yielding a left-nested
pair for a chain of length > 1.

Dispatch is handled through the profunctor's ifold_vl method, which takes
an indexed fold VL function:

  f : Monoid r => (i -> a -> r, r) -> s -> r

This uses two principal concrete profunctors, with the second used
when the index is ignored:

  * IndexedForget  — indexed reading actions (ifold_map_of, icollect, ...)
  * Forget         — plain reading actions after cast_as(FOLD)  (collect, ...)

"""

from __future__   import annotations

from typing       import Callable

from ..functions  import Function, fn_eval, identity
from ..list       import List
from ..maybe      import Maybe, Nothing, Some
from ..monoids    import Endo, First, Monoid
from ..utils      import Collect

from .optic       import Optic, OpticIs, _MISSING
from .profunctors import IndexedForget
from .generics    import ifold_vl_

__all__ = [
    'IxFold',
    'IxAffineFold',
    'ifolded',
    'ifolding',
    'ifold_map_of',
    'iright_fold_of',
    'ileft_fold_of',
    'icollect',
    'ipreview',
]


class IxFold(Optic, optic_is=OpticIs.IX_FOLD):
    """A read-only indexed optic extracting zero or more indexed result."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.IX_FOLD)


class IxAffineFold(Optic, optic_is=OpticIs.IX_AFFINE_FOLD):
    """A read-only indexed optic extracting at most one indexed result."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.IX_AFFINE_FOLD)


#
# ifolded : IndexedFoldable f => IxFold (f i a) i a
#
# Analogous to folded, but requires IndexedFoldable (which provides ifold_map)
# rather than plain Foldable.  The structure's intrinsic indices are the optic's
# indices.
#
# Each profunctor receives the fold VL function via p.ifold_vl(f), where
# f calls the container's ifold_map.  IndexedForget accumulates indices;
# Forget ignores them.
#

ifolded = IxFold(ifold_vl_(lambda ig, m: lambda s: s.ifold_map(ig, m)))
ifolded.__doc__ = """Indexed fold over all elements of any IndexedFoldable container."""

def ifolding(f: Callable) -> IxFold:
    """Builds an IxFold from a function s -> IndexedFoldable i a.

    ifolding : IndexedFoldable f => (s -> f i a) -> IxFold i s a

    """
    return IxFold(ifold_vl_(lambda ig, m: lambda s: f(s).ifold_map(ig, m)))


#
# Indexed folding actions
#

def ifold_map_of(optic, f: Callable, monoid: Monoid = Collect) -> Function:
    """Maps each (index, focus) pair to a monoid value and combines them.

    ifold_map_of : IxFold i s a -> (i -> a -> m) -> s -> m

    """
    p = IndexedForget(f, monoid)
    optic_f = optic.cast_as(OpticIs.IX_FOLD)
    # g: i_acc -> s -> m
    g = IndexedForget.run(optic_f(p))    # type: ignore
    return Function(lambda s: g(_MISSING, s))

def icollect(optic) -> Function:
    """Collects all (index, focus) pairs into a list.

    icollect : IxFold i s a -> s -> List (i, a)

    """
    return ifold_map_of(optic, lambda i, a: List.of((i, a)), Collect)

def ipreview(optic) -> Function:
    """Optionally extracts the leftmost (index, focus) pair of an optic.

    Works with any IxFold-like optic (IxLens, IxFold, IxTraversal, ...).
    Returns a function s -> Maybe (i, a): Some((i, a)) for the first match,
    Nothing() if there is none.

    ipreview : IxFold i s a -> s -> Maybe (i, a)

    """
    def previewed(s):
        result = ifold_map_of(optic, lambda i, a: Some((i, a)), First)(s)
        # An empty structure yields First's raw munit (None) rather than
        # Nothing(), since fold_map never calls mcombine with no elements
        # to combine; normalize that here.
        return result if isinstance(result, Maybe) else Nothing()

    return Function(previewed)

def iright_fold_of(optic, f: Callable, init) -> Function:
    """Right fold over indexed foci using an accumulating function.

    iright_fold_of : IxFold i s a -> (i -> a -> r -> r) -> r -> s -> r

    """
    def reduce(i, a):
        return lambda r: f(i, a, r)

    return Function(lambda s: fn_eval(s >> ifold_map_of(optic, reduce, Endo), init))

def ileft_fold_of(optic, f: Callable, init) -> Function:
    """Left fold over indexed foci using an accumulating function.

    ileft_fold_of : IxFold i s a -> (r -> i -> a -> r) -> r -> s -> r

    Derived from iright_fold_of via the difference-list technique; correct
    but O(n) in stack depth. Override for strict efficiency if needed.

    """
    def reduce(i, a, cont):          # cont : r -> r  (difference-list continuation)
        return lambda r: cont(f(r, i, a))

    return Function(lambda s: (s >> iright_fold_of(optic, reduce, identity))(init))
