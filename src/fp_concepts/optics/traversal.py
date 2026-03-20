"""Traversal - optics focusing on zero or more elements.

A Traversal s t a b focuses on zero or more elements of type a inside
an s, allowing effectful reads and pure modifications.

  type Traversal s t a b = forall p. wander p => p a b -> p s t

The canonical profunctor witness is Star f for Applicative f.
Every Lens and Prism is a Traversal.
"""

from __future__    import annotations

from typing        import Callable

from ..pair        import Pair
from ..functions   import compose
from ..traversable import traverse_

from .generics     import wander_
from .optic        import Optic, OpticIs
from .profunctors  import Star
from .setter       import over, put     # re-exported

__all__ = [
    'Traversal',
    'traversal',
    'both',
    'each',
    'traverse_of',
    'over',
    'put',
]


class Traversal(Optic, optic_is=OpticIs.TRAVERSAL):
    """An optic focusing on zero or more elements."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.TRAVERSAL)


def traversal(f: Callable) -> Traversal:
    """Build a Traversal from a van Laarhoven traversal function.

    f :: Applicative g => (a -> g b) -> s -> g t
    """
    return Traversal(wander_(f))


def traverse_of(optic, effect, f: Callable) -> Callable:
    """Run an effectful function over all foci; returns s -> f t.

    traverse_of :: Applicative f => Traversal -> type[f] -> (a -> f b) -> (s -> f t)

    The effect class is passed explicitly because Python cannot infer it.
    """
    s = Star(f, effect)
    return Star.run(optic(s))


def _both_vl(f):
    """van Laarhoven traversal over both elements of a 2-tuple or Pair."""
    def go(pair):
        a, b = pair
        return f(a).map2(Pair, f(b))
    return go


both = traversal(_both_vl)
"""Traversal over both elements of a 2-tuple or Pair."""

each = traversal(traverse_)
"""Traversal over all elements of any Traversable container."""
