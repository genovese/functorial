"""Iso -- the family of optics that represent an isomorphism.

An Iso s t a b represents a reversible transformation: s <-> a and b <-> t.
It is the most specific optic and can be used as any other optic type,
since every Profunctor p supports dimap.

  type Iso s t a b = forall p. Profunctor p => p a b -> p s t

Built from two functions:
  sa : s -> a   (forward  / view  direction)
  bt : b -> t   (backward / review direction)

Laws (for simple Iso s a where s=t, a=b):
  view  (iso sa bt) s = sa s
  review (iso sa bt) b = bt b
  sa (bt b) = b
  bt (sa s) = s

"""

from __future__    import annotations

from typing        import Callable

from ..profunctor  import dilift
from ..functions   import Function, identity
from ..maybe       import maybe, Nothing, Some

from .optic        import Optic, OpticIs
from .re_          import re

__all__ = [
    'Iso',
    'iso',
    'from_',
    'involuted',
    'non',
    'coerced',
]


class Iso(Optic):
    """An optic representing an invertible transformation."""
    def __init__(self, f):
        super().__init__(f, OpticIs.ISO)


def iso(sa: Callable, bt: Callable) -> Iso:
    """Builds an Iso from a forward function and a backward function.

    sa : s -> a   (view direction: extract the focus)
    bt : b -> t   (review direction: construct from focus)

    The two functions should be mutual inverses:
        sa (bt b) = b
        bt (sa s) = s

    """
    return Iso(dilift(Function(sa), Function(bt)))


def from_(the_iso: Iso) -> Iso:
    """Reverses an Iso, swapping source/target and forward/backward.

    from_ : Iso s t a b -> Iso b a t s

    Uses Re(identity) to run the optic in reverse via the duality
    of Profunctor methods (see re_.py).

    """
    return Iso(re(the_iso))


def involuted(f: Callable) -> Iso:
    """Build an Iso from a self-inverse function (f . f = id).

    involuted :: (a -> a) -> Iso a a a a

    Example: involuted(lambda x: -x)   — negation is its own inverse
             involuted(sorted)          — only valid if sort is idempotent
    """
    return iso(f, f)


def non(default) -> Iso:
    """Creates an Iso between Maybe a and a, using a default value for Nothing.

    non :: Eq a => a -> Iso (Maybe a) (Maybe a) a a

    view  (non d) (Some a) = a
    view  (non d) Nothing  = d
    review (non d) a       = Nothing if a == default else Some(a)

    Useful for working with optional dictionary values as if they
    were always present.
    """
    forward = lambda ma: maybe(default, identity, ma)
    backward = lambda a:  Nothing() if a == default else Some(a)
    return iso(forward, backward)


def coerced(forward: Callable, backward: Callable) -> Iso:
    """Builds an Iso from explicit coercion functions (alias for iso).

    Useful when the directionality of 'forward' and 'backward' is
    clearer than 'sa' and 'bt'.

    """
    return iso(forward, backward)
