"""AffineTraversal -- optics focusing on *at most* one element.

An AffineTraversal s t a b focuses on zero or one element inside s,
combining the optionality of Prism with the contextual update of Lens.

  type AffineTraversal s t a b =
    forall p. (Strong p, Choice p, visit p) => p a b -> p s t

Common cases:
  + Every Lens is an AffineTraversal (always has a focus).
  + Every Prism is an AffineTraversal (optionally has a focus, no context).

"""

from __future__    import annotations

from typing        import Callable

from ..either      import Either, Left, Right
from ..maybe       import Some

from .generics     import visit_
from .optic        import Optic, OpticIs
from .review       import preview   # re-exported
from .setter       import over, put  # re-exported

__all__ = [
    'AffineTraversal',
    'affine_traversal',
    'affine_traversal_vl',
    'matching',
    'preview',
    'over',
    'put',
]


class AffineTraversal(Optic):
    """An optic focusing on at most one element."""
    def __init__(self, f):
        super().__init__(f, OpticIs.AFFINE_TRAVERSAL)


def affine_traversal(f: Callable) -> AffineTraversal:
    """Builds an AffineTraversal from a visit function.

    f :: Applicative g => (forall r. r -> g r) -> (a -> g b) -> s -> g t

    The first argument `(r -> g r)` handles the no-focus case by lifting
    the unchanged structure into the functor.

    """
    return AffineTraversal(visit_(f))


def affine_traversal_vl(
        match_fn: Callable,
        update_fn: Callable,
) -> AffineTraversal:
    """Builds an AffineTraversal from a match function and an update function.

    match_fn  : s -> Either t a
        Left t  = no focus; return t as the (unchanged) result structure
        Right a = has focus a

    update_fn : s -> b -> t
        rebuild s with a modified focus value b

    This is the most ergonomic constructor for hand-written AffineTraversals.

    Example:  first element of a non-empty list:
        affine_traversal_vl(
            lambda s: Right(s[0]) if s else Left(s),
            lambda s, b: [b] + s[1:]
        )

    """
    def visit_fn(point, f, s):
        match match_fn(s):
            case Right(a):
                return f(a).map(lambda b: update_fn(s, b))
            case Left(t):
                return point(t)
    return AffineTraversal(visit_(visit_fn))


def matching(optic, s) -> Either:
    """Returns Right(a) if the optic has a focus in s, otherwise Left(s).

    matching : AffineTraversal s t a b -> s -> Either s a

    For simple optics where s = t this is the full Either-valued preview.
    Use preview(optic)(s) for a Maybe-valued alternative.

    """
    match preview(optic)(s):
        case Some(a): return Right(a)
        case _:       return Left(s)
