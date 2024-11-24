#
# Getter is a profunctor and bicofunctor (and cochoice).
#
# type Getter s a = Getter { runGetter :: s -> a }
#

from __future__    import annotations

from typing        import Callable, cast

from ..Bicofunctor import Bicofunctor
from ..Profunctor  import Profunctor, dilift
from ..functions   import Function, compose, fst, identity, snd

from .Forget       import Forget

__all__ = ['view', 'view_with', 'getter', 'to']


class Getter[S, A](Profunctor, Bicofunctor):  # Add Cochoice when available
    """ATTN

    """
    def __init__(self, s_to_a: Callable[[S], A]):
        self._s_to_a = Function(s_to_a)

    @classmethod
    def run(cls, fg):
        return fg._s_to_a

    def dimap(self, f, _):
        return Getter(compose(self._s_to_a._fn, f))

    def bicomap[B](self, f: Callable[[B], A], _g: Callable) -> Getter[S, B]:
        return Getter(compose(self._s_to_a._fn, f))

#    def unfirst(self):
#        return Forget(compose(self._s_to_a, fst))
#
#    def unsecond(self):
#        return Forget(compose(self._s_to_a, snd))


idF: Forget = Forget(identity)

def view(optic):
    """Returns the value pointed to by a Getter.

    """
    return Forget.run(optic(idF))

def view_with(optic, f=identity):
    """Returns the value pointed to by a Getter transformed by a function.

    """
    return Forget.run(optic(Forget(f)))

# ATTN: Make a Getter that inherits properly
# and make this a generic wrapping the constructor.
# ATTN2: Can we make the right argument error if used in appropriately
# identity below is not satisfying.
# Or at least have an ignore(*args, **kwargs) -> None

def absurd(*args, **kwargs):
    raise TypeError('Getter is read only; rmap component is a phantom.')

def to[S, A](f: Callable[[S], A]):
    """Builds and returns a Getter from a function from structure to substructure.

    """
    # A Getter is a Bicofunctor and a Profunctor, so
    # the second type argument is a "phantom" type.
    return dilift(Function(f), cast(Callable[[A], S], absurd))

# ATTN: Not sure I really like to here ^^^. It's rather too general a name
# for this special case.

def getter[S, A](f: Callable[[S], A]):
    """Builds and returns a Getter from a function from structure to substructure.

    """
    # A Getter is a Bicofunctor and a Profunctor, so
    # the second type argument is a "phantom" type.
    # Do we need to enhance this with a Bicofunctor and Cochoice methods?
    # probably...
    return dilift(Function(f), cast(Callable[[A], S], absurd))
