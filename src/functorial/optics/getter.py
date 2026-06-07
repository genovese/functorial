"""Getter -- read-only optics focusing on exactly one element.

A Getter s a extracts a single value of type a from a structure s.
It is the read-only counterpart of Lens: you can view but not modify.

  type Getter s a =
      forall p. (Profunctor p, Bicofunctor p, Cochoice p) => p a a -> p s s

Any Getter can be used as an AffineFold or Fold.

"""

from __future__    import annotations

from typing        import Callable, cast

from ..profunctor  import dilift
from ..functions   import Function, identity

from .generics     import absurd
from .optic        import Optic, OpticIs
from .profunctors  import Forget

__all__ = ['Getter', 'view', 'view_with', 'getter']


class Getter(Optic, optic_is=OpticIs.GETTER):
    """A read-only optic extracting exactly one element."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.GETTER)

idF: Forget = Forget(identity)

def view(optic) -> Function:
    """Returns the single focus of a Getter as a function s -> a."""
    return Forget.run(optic(idF))

def view_with(optic, f=identity) -> Function:
    """Returns the focus of a Getter transformed by f, as a function s -> b."""
    return Forget.run(optic(Forget(f)))

def getter[S, A](f: Callable[[S], A]) -> Getter:
    """Builds a Getter from a function mapping structure to focus.

    getter : (s -> a) -> Getter s a

    """
    return Getter(dilift(Function(f), cast(Callable[[A], S], absurd)))
