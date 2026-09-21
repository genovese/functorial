#
# Bifunctors that can be traversed in order on both variables
#
# trait Bitraversable (f : Type -> Type -> Type) where
#     traverse : (a -> f c) -> (b -> f d) -> t a b -> f (t c d)
#

from __future__ import annotations

from abc             import abstractmethod
from collections.abc import Callable

from .applicative    import Applicative, IdentityA
from .bifunctor      import Bifunctor
from .functions      import identity
from .traversable    import Traversable_
from .wrappers       import get_effect

__all__ = ['Bitraversable', 'Bitraversable_', 'bitraverse', 'bisequence']


class Bitraversable(Bifunctor):
    @abstractmethod
    def bitraverse(self, f: type[Applicative], g1: Callable, g2: Callable) -> Applicative:   # Hard to type properly in Python
        ...

class Bitraversable_(Bitraversable, Traversable_):
    """Bitraversable base class for inheritance that provides default implementations of extra methods.

    The Bitraversable protocol defines the primitive `bitraverse`; this gives
    access to `traverse` and `bisequence` as well with a default implementation that
    should work without extra effort. So for concrete Bitraversable classes,
    we generally prefer inheriting from this rather than just relying
    on the protocol (i.e., implementing `bitraverse` as a method).

    """
    @abstractmethod
    def bitraverse(self, f: type[Applicative], g1: Callable, g2: Callable) -> Applicative:
        ...

    def traverse(self, f: type[Applicative], g: Callable) -> Applicative:
        """A traversal over the second component only, derived from bitraverse."""
        return self.bitraverse(f, f.pure, g)

    def bisequence(self, effect: type[Applicative] = IdentityA) -> Applicative:
        """Evaluates effects on both components, collecting the results in the same shape."""
        return self.bitraverse(effect, identity, identity)


def bitraverse( g1: Callable, g2: Callable, bt: Bitraversable, effect: type[Applicative] = IdentityA) -> Applicative:
    """Evaluates effectful functions at each element of a structure, giving the same shape in an effectful context.

    Type: (Bitraversable t, Applicative f) => (a -> f c) -> (b -> f d) -> t a b -> f (t c d)

    """
    return bt.bitraverse(get_effect(g1) or get_effect(g2) or effect, g1, g2)


def bisequence(bt: Bitraversable, effect: type[Applicative] = IdentityA) -> Applicative:
    """Evaluate effects on each element of a structure, collecting the results in the effectful context.

    Type: (Bitraversable t, Applicative f) =>  t a b -> f (t a b)

    """
    return bitraverse(identity, identity, bt, effect)
