"""Iso -- the family of optics that represent invertible transformations.

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

from ..either      import Left, Right
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
    'negated',
    'swapped',
    'flipped',
]


class Iso(Optic, optic_is=OpticIs.ISO):
    """An optic representing an invertible transformation."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.ISO)


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
    """Builds an Iso from a self-inverse function (f . f = id).

    involuted : (a -> a) -> Iso a a a a

    Examples:
      - involuted(lambda x: -x)   — negation is its own inverse
      - involuted(sorted)         — only valid if sort is idempotent

    """
    return iso(f, f)

def non(default) -> Iso:
    """Creates an Iso between Maybe a and a, using a default value for Nothing.

    non : Eq a => a -> Iso (Maybe a) (Maybe a) a a

    Laws:
      - view  (non d) (Some a) == a
      - view  (non d) Nothing  == d
      - review (non d) a       == Nothing if a == default else Some(a)

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

negated: Iso = involuted(lambda x: -x)
negated.__doc__ = """Iso negating a number; self-inverse since -(-x) = x."""

swapped: Iso = involuted(lambda p: (p[1], p[0]))
swapped.__doc__ = """Iso swapping the two components of a pair; self-inverse."""

def _flip_either(e):
    match e:
        case Left(a):  return Right(a)
        case Right(b): return Left(b)

flipped: Iso = involuted(_flip_either)
flipped.__doc__ = """Iso swapping Left and Right in an Either; self-inverse."""
