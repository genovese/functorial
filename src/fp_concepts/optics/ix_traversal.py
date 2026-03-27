"""IxTraversal and IxAffineTraversal — indexed read-write optics over multiple foci.

An IxTraversal i s t a b focuses on zero or more values of type a inside s,
each paired with an index of type i, and allows both effectful reads
(itraverse_of) and pure modifications (iover).

Indexed optic functions dispatch on the profunctor:

  * IndexedStar    — effectful traversal (itraverse_of)
  * Indexed        — pure modification (iover, iput) via Identity wrapping
  * IndexedForget  — indexed folding actions (icollect, ifold_map_of)
  * Forget         — plain folding after cast_as(FOLD)

Index accumulation follows the _MISSING / _pack_index convention.

"""

from __future__  import annotations

from typing      import Callable

from ..const     import Const, run_const, type_const
from ..functor   import lift
from ..functions import Function, compose
from ..identity  import Identity
from ..traversable import itraverse as _itraverse

from .optic       import Optic, OpticIs, _MISSING, _pack_index
from .profunctors import Forget, IndexedForget, Indexed, IndexedStar, Star
from .ix_lens     import iover, iput          # re-export

__all__ = [
    'IxTraversal',
    'IxAffineTraversal',
    'itraversal',
    'ieach',
    'itraverse_of',
    'iover',
    'iput',
]


class IxTraversal(Optic, optic_is=OpticIs.IX_TRAVERSAL):
    """An indexed optic focusing on zero or more elements."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.IX_TRAVERSAL)


class IxAffineTraversal(Optic, optic_is=OpticIs.IX_AFFINE_TRAVERSAL):
    """An indexed optic focusing on at most one element."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.IX_AFFINE_TRAVERSAL)


def itraversal(f: Callable) -> IxTraversal:
    """Builds an IxTraversal from an indexed van Laarhoven traversal function.

    f :: (i -> a -> g b) -> s -> g t   for any Applicative g

    itraversal :: (forall g. Applicative g => (i -> a -> g b) -> s -> g t)
               -> IxTraversal i s t a b

    """
    def the_traversal(p):
        if isinstance(p, IndexedStar):
            # Effectful traversal: run f with index-accumulating effectful function.
            g = IndexedStar.run(p)    # i_acc -> a -> f b
            eff = p._effect
            def go(i_acc, s):
                return f(lambda ix, a: g(_pack_index(i_acc, ix), a))(s)
            return IndexedStar(go, eff)

        if isinstance(p, Indexed):
            # Pure modification: wrap in Identity, unwrap after.
            g = Indexed.run(p)        # i_acc -> a -> b
            def modify(i_acc, s):
                return Identity.run(
                    f(lambda ix, a: Identity(g(_pack_index(i_acc, ix), a)))(s)
                )
            return Indexed(modify)

        if isinstance(p, IndexedForget):
            # Indexed fold: use f with a Const applicative to extract values.
            h = IndexedForget.run(p)  # i_acc -> a -> r
            m = p._monoid             # pylint: disable=protected-access
            cls_c = type_const(m)
            def fold_s(i_acc, s):
                g_c = lambda ix, a: Const(h(_pack_index(i_acc, ix), a), m)
                return run_const(f(g_c)(s))
            return IndexedForget(fold_s, m)

        if isinstance(p, Star):
            # Plain effectful traversal (cast_as(TRAVERSAL) path): ignore indices.
            g = Star.run(p)
            eff = p._functor
            def star_run(s):
                return f(lambda _ix, a: g(a))(s)
            return Star(star_run, eff)

        if isinstance(p, Forget):
            # Plain fold (cast_as(FOLD) path): run f forgetting indices.
            h = Forget.run(p)
            m = p._monoid             # pylint: disable=protected-access
            cls_c = type_const(m)
            def plain_fold(s):
                g_c = lambda _ix, a: Const(h(a), m)
                return run_const(f(g_c)(s))
            return Forget(plain_fold, m)

        raise TypeError(f'itraversal: unsupported profunctor {type(p).__name__!r}')

    return IxTraversal(the_traversal)


#
# ieach : IndexedTraversable f => IxTraversal i (f a) (f b) a b
#
# Indexed traversal over all elements of an IndexedTraversable container.
# Uses the container's itraverse method, which provides intrinsic indices.
#

def _ieach_fn(p):
    """Core dispatch for ieach."""
    if isinstance(p, IndexedStar):
        g = IndexedStar.run(p)    # i_acc -> a -> f b
        eff = p._effect
        def modify(i_acc, s):
            return _itraverse(lambda ix, a: g(_pack_index(i_acc, ix), a), s, eff)
        return IndexedStar(modify, eff)

    if isinstance(p, Star):
        # Plain effectful traversal (cast_as(TRAVERSAL) path): ignore indices.
        g = Star.run(p)           # a -> f b
        eff = p._functor
        def star_modify(s):
            return _itraverse(lambda _ix, a: g(a), s, eff)
        return Star(star_modify, eff)

    if isinstance(p, Indexed):
        g = Indexed.run(p)        # i_acc -> a -> b (pure)
        def pure_modify(i_acc, s):
            return Identity.run(
                _itraverse(lambda ix, a: Identity(g(_pack_index(i_acc, ix), a)), s)
            )
        return Indexed(pure_modify)

    if isinstance(p, IndexedForget):
        h = IndexedForget.run(p)
        m = p._monoid             # pylint: disable=protected-access
        cls_c = type_const(m)
        def fold_s(i_acc, s):
            g_c = lambda ix, a: Const(h(_pack_index(i_acc, ix), a), m)
            return run_const(_itraverse(g_c, s, cls_c))
        return IndexedForget(fold_s, m)

    if isinstance(p, Forget):
        h = Forget.run(p)
        m = p._monoid             # pylint: disable=protected-access
        cls_c = type_const(m)
        def plain_fold(s):
            g_c = lambda _ix, a: Const(h(a), m)
            return run_const(_itraverse(g_c, s, cls_c))
        return Forget(plain_fold, m)

    raise TypeError(f'ieach: unsupported profunctor {type(p).__name__!r}')


ieach = IxTraversal(_ieach_fn)
"""Indexed traversal over all elements of any IndexedTraversable container."""


#
# Indexed traversal action
#

def itraverse_of(optic, effect: type, g: Callable) -> Function:
    """Runs an indexed effectful function over all foci; returns s -> f t.

    itraverse_of :: Applicative f
                 => IxTraversal i s t a b -> type[f] -> (i -> a -> f b) -> s -> f t

    The effect class is passed explicitly because Python cannot infer it.

    """
    p = IndexedStar(g, effect)
    result = IndexedStar.run(optic(p))    # i_acc -> s -> f t
    return Function(lambda s: result(_MISSING, s))
