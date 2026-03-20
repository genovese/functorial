"""AffineFold -- read-only optics extracting at most one element.

An AffineFold s a can optionally extract an element of type a from s.
It is the read-only version of AffineTraversal: you can preview but
not modify.

  type AffineFold s a =
      forall p. (Strong p, Choice p, Cochoice p, Bicofunctor p) => p a a -> p s s

"""

from __future__    import annotations

from typing        import Callable

from ..bicofunctor import bicomap_
from ..either      import Left, Right
from ..maybe       import maybe, Some
from ..functions   import compose

from .choice       import into_right
from .optic        import Optic, OpticIs
from .review       import preview, preview_with   # preview re-exported

from .generics     import rphantom, visit_

__all__ = [
    'AffineFold',
    'afold',
    'afolding',
    'filtered',
    'a_or',
    'has',
    'preview_of',
    'preview',
]


class AffineFold(Optic, optic_is=OpticIs.AFFINE_FOLD):
    """A read-only optic extracting at most one element."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.AFFINE_FOLD)


def afold(f: Callable) -> AffineFold:
    """Build an AffineFold from a visit-style function (read-only).

    f :: (forall r. r -> g r) -> (a -> g b) -> s -> g t

    The rphantom wrappers at both ends make the output type phantom,
    turning the traversal into a read-only fold.
    """
    return AffineFold(compose(rphantom, visit_(f), rphantom))


def afolding(f: Callable) -> AffineFold:
    """Build an AffineFold from a partial function s -> Maybe a."""
    def af(s):
        return maybe(Left(s), Right, f(s))
    return AffineFold(compose(bicomap_(af, Left), into_right))


def filtered(predicate: Callable) -> AffineFold:
    """AffineFold that focuses only when the predicate holds.

    filtered :: (a -> bool) -> AffineFold a a

    preview (filtered p) x  =  Some(x) if p(x) else Nothing()
    """
    def fd(point, f, a):
        if predicate(a):
            return f(a)
        return point(a)
    return afold(fd)


def a_or(a: AffineFold, b: AffineFold) -> AffineFold:
    """Try the first AffineFold; if no focus, try the second.

    a_or :: AffineFold s a -> AffineFold s a -> AffineFold s a
    """
    def alt(s):
        return maybe(preview(b)(s), Some, preview(a)(s))
    return afolding(alt)


def has(optic, s) -> bool:
    """Return True if the optic has a focus in s.

    has :: AffineFold s a -> s -> bool
    """
    return maybe(False, lambda _: True, preview(optic)(s))


def preview_of(optic, f: Callable) -> Callable:
    """Extract and transform the focus if present; returns s -> Maybe b.

    preview_of :: AffineFold s a -> (a -> b) -> (s -> Maybe b)
    """
    return preview_with(optic, f)
