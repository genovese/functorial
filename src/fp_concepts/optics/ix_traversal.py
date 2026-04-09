"""IxTraversal and IxAffineTraversal — indexed read-write optics over multiple foci.

An IxTraversal i s t a b focuses on zero or more values of type a inside s,
each paired with an index of type i, and allows both effectful reads
(itraverse_of) and pure modifications (iover).

Indexed optic functions dispatch through the profunctor's iwander method,
which encapsulates each profunctor's handling of an indexed VL traversal
function:

  * IndexedStar    — effectful traversal (itraverse_of)
  * Indexed        — pure modification (iover, iput) via Identity wrapping
  * IndexedForget  — indexed folding actions (icollect, ifold_map_of)
  * Forget         — plain folding after cast_as(FOLD)
  * Star           — plain modification after cast_as(TRAVERSAL)

Index accumulation follows the _MISSING / _pack_index convention.
See optic.py and ix_lens.py.

"""

from __future__  import annotations

from typing      import Callable

from ..applicative import Applicative
from ..functions   import Function
from ..traversable import itraverse_
from ..wrappers    import EffectfulFunction

from .optic       import Optic, OpticIs, _MISSING
from .profunctors import IndexedStar
from .generics    import iwander_
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

    f : Applicative g => (i -> a -> g b) -> s -> g t

    itraversal : (forall g. Applicative g => (i -> a -> g b) -> s -> g t)
               -> IxTraversal i s t a b

    Each profunctor p receives the VL function via p.iwander(f), which
    constructs the concrete indexed function from p's own inner function
    and tags it with the appropriate Applicative via EffectfulFunction.

    """
    return IxTraversal(iwander_(f))


#
# Indexed traversal over all elements of an IndexedTraversable container.
# Uses the container's itraverse method, which provides intrinsic indices.
#
# ieach : IndexedTraversable f => IxTraversal i (f a) (f b) a b
#
# Implementation note: itraverse_ is already curried:
#
#     itraverse_(g) : IndexedTraversable -> g t
#
# The EffectfulFunction tag on g lets itraverse_ pick the right Applicative.
#

ieach = itraversal(itraverse_)
ieach.__doc__ = """Indexed traversal over all elements of any IndexedTraversable container."""


#
# Indexed traversal action
#

def itraverse_of(optic, g: Callable, effect: type[Applicative] | None = None) -> Function:
    """Applies an indexed effectful function to each element of a structure
    targeted by an Indexed Traversal, evaluates these actions from left to
    right, and collects the results.

    If g is an EffectfulFunction, the corresponding effect type f
    is used, otherwise an Applicative type f should be provided in
    the effect argument. Raises an exception if the effect type
    cannot be determined.

    Returns a function s -> f t to be applied to traversable structure s.

    itraverse_of : Applicative f
                 => IxTraversal i s t a b
                 -> EffectfulFunction (i, a) (f b)
                 -> (s -> f t)

                   Applicative f
                 => IxTraversal i s t a b
                 -> (i -> a -> f b)
                 -> Type f
                 -> (s -> f t)

    """
    if isinstance(g, EffectfulFunction):
        effect = g.effect
    elif effect is None:
        raise ValueError('Cannot determine effect type in itraverse_of, '
                         'supply effect argument or EffectfulFunction.')

    p = IndexedStar(g, effect)
    result: Callable = IndexedStar.run(optic(p))    # i_acc -> s -> f t
    return Function(lambda s: result(_MISSING, s))  # strip index accumulator sentinel
