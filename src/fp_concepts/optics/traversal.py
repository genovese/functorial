"""Traversal - optics focusing on zero or more elements.

A Traversal s t a b focuses on zero or more elements of type a inside
an s, allowing effectful reads and pure modifications.

  type Traversal s t a b = forall p. Wander p => p a b -> p s t

The canonical profunctor witness is Star f for Applicative f.
Every Lens and Prism is a Traversal.

"""

from __future__    import annotations

from typing        import Callable

from ..applicative import Applicative
from ..pair        import Pair
from ..traversable import traverse_
from ..wrappers    import EffectfulFunction

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
    """Builds a Traversal from a van Laarhoven traversal function.

    Parameters
    ----------
    - f : Applicative g => (a -> g b) -> s -> g t

    """
    return Traversal(wander_(f))


def traverse_of(optic, fn: Callable, effect: type[Applicative] | None = None) -> Callable:
    """Applies an effectful function to each element of a structure
    targeted by a Traversal, evaluates these actions from left to
    right, and collects the results.

    If fn is an EffectfulFunction, the corresponding effect type f
    is used, otherwise an Applicative type f should be provided in
    the effect argument. Raises an exception if the effect type
    cannot be determined.

    Returns a function s -> f t to be applied to traversable structure s.

    traverse_of : Applicative f => Traversal -> EffectfulFunction a (f b) -> (s -> f t)
                  Applicative f => Traversal -> (a -> f b) -> Type f -> (s -> f t)

    """
    if isinstance(fn, EffectfulFunction):
        effect = fn.effect
    elif effect is None:
        raise ValueError('Cannot determine effect type in traverse_of, '
                         'supply effect argument or EffectfulFunction.')

    s = Star(fn, effect)
    return Star.run(optic(s))


def _both_vl(f):
    """van Laarhoven traversal over both elements of a 2-tuple or Pair."""
    def go(pair):
        a, b = pair
        return f(a).map2(Pair, f(b))
    return go


both = traversal(_both_vl)
both.__doc__ = """Traversal over both elements of a 2-tuple or Pair."""

each = traversal(traverse_)
each.__doc__ = """Traversal over all elements of any Traversable container."""
