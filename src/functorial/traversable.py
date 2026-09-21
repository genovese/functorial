#
# trait Traversable (t : Type -> Type) where
#     traverse : Applicative f => (a -> f b) -> t a -> f (t b)
#
from __future__   import annotations

from abc             import abstractmethod
from collections.abc import Callable
from typing          import Protocol, cast

from .applicative import Applicative, IdentityA
from .functor     import Functor
from .functions   import identity
from .wrappers    import get_effect

__all__ = ['Traversable', 'Traversable_', 'traverse', 'sequence',
           'IndexedTraversable', 'IndexedTraversable_', 'itraverse',]


class Traversable(Functor, Protocol):
    @abstractmethod
    def traverse(self, f: type[Applicative], g: Callable) -> Applicative:   # Hard to type properly in Python
        ...

class Traversable_(Traversable):
    """Traversable base class for inheritance that provides default implementations of extra methods.

    The Traversable protocol defines the primitive `traverse`; this gives
    access to `sequence` as well with a default implementation that
    should work without extra effort. So for concrete Traversable classes,
    we generally prefer inheriting from this rather than just relying
    on the protocol (i.e., implementing traverse as a method).

    """
    @abstractmethod
    def traverse(self, f: type[Applicative], g: Callable) -> Applicative:
        ...

    def sequence(self, effect: type[Applicative] = IdentityA) -> Applicative:
        """Evaluates effects on each element, collecting the results in the same shape."""
        return self.traverse(effect, identity)

class IndexedTraversable[I, A](Traversable, Protocol):
    @abstractmethod                                                           # vv Hard to type properly in Python
    def itraverse(self, f: type[Applicative], g: Callable[[I, A], Applicative]) -> Applicative:
        ...

class IndexedTraversable_[I, A](IndexedTraversable[I, A], Traversable_):
    """IndexedTraversable base class for inheritance that provides default implementations of extra methods.

    The IndexedTraversable Traversable protocol defines the
    primitive `itraverse`; this gives access to `isequence` as well
    with a default implementation that should work without extra
    effort. So for concrete IndexedTraversable classes, we generally
    prefer inheriting from this rather than just relying on the
    protocol (i.e., implementing traverse as a method).

    """
    @abstractmethod
    def itraverse(self, f: type[Applicative], g: Callable[[I, A], Applicative]) -> Applicative:
        ...

    def traverse(self, f: type[Applicative], g: Callable) -> Applicative:
        """A traversal that ignores the index, using itraverse."""
        return self.itraverse(f, lambda _i, a: g(a))

    def isequence(self, effect: type[Applicative] = IdentityA) -> Applicative:
        """Evaluates effects on each (index, element) pair, collecting results in the same shape."""
        return self.itraverse(effect, lambda _i, a: cast(Applicative, a))


#
# Traversal over a traversable functor using an effectful function.
# A traversable functor t a can be (conceptually) decomposed into
# a shape t Unit and a List a. This applies the function to each
# value in the list putting the results in the same shape. The entire
# result is in the effectful context.
#
# traverse : (Applicative f, Traversable t) => (a -> f b) -> t a -> f (t b)
#
# Note that we pass the applicative *class* as an argument because Python
# cannot infer it from g. We only use -- at most -- the `pure` class method.
# If, however, g is an EffectfulFunction, it can hold its Applicative type,
# and this is used instead if available. The applicative is a limited
# form of Identity (IdentityA) if no other is supplied.
#

def traverse( g: Callable, t: Traversable, effect: type[Applicative] = IdentityA) -> Applicative:
    """Traversal over a functor with an effectful function.

    The traversal produces an object of the same ``shape'' in the effectful context.

    Parameter
    ---------
    g - An effectful function a -> f b. If an instance of the EffectfulFunction class,
        the Applicative instance f can be determined from g, else see `effect`.

    t - A traversable functor object. Think of this as a pairing (T (), [a])
        where t : T a

    effect - an Applicative subclass, i.e., a type. By default this is a limited
        Applicative version of Identity (IdentityA).  If g is an EffectfulFunction,
        then its effect takes precedence.

    Returns the transformed traversable in the effectful context of type f (T b).

    """
    return t.traverse(get_effect(g) or effect, g)

def traverse_( g: Callable, effect: type[Applicative] = IdentityA) -> Callable[[Traversable], Applicative]:
    """Partial evaluation of traverse without the traversable. Returns a function that does the traversal.

    The _ in a name is intended to evoke a hole to be filled in with the traversable.

    """
    return lambda t: traverse(g, t, effect)

def sequence(t: Traversable, effect: type[Applicative] = IdentityA) -> Applicative:
    """Evaluate effects on each element of a structure, collecting the results in the effectful context.

    Type: (Traversable t, Applicative f) =>  t a -> f (t a)

    """
    return traverse(identity, t, effect)

def itraverse( g: Callable, t: IndexedTraversable, effect: type[Applicative] = IdentityA) -> Applicative:
    """Traversal over a functor with an effectful function that takes an index.

    The traversal produces an object of the same ``shape'' in the effectful context.

    Parameter
    ---------
    g - An effectful function i -> a -> f b. If an instance of the EffectfulFunction class,
        the Applicative instance f can be determined from g, else see `effect`.

    t - A traversable functor object. Think of this as a pairing (T (), [a])
        where t : T a

    effect - an Applicative subclass, i.e., a type. By default this is a limited
        Applicative version of Identity (IdentityA).  If g is an EffectfulFunction,
        then its effect takes precedence.

    Returns the transformed traversable in the effectful context of type f (T b).

    """
    return t.itraverse(get_effect(g) or effect, g)

def itraverse_( g: Callable, effect: type[Applicative] = IdentityA) -> Callable[[IndexedTraversable], Applicative]:
    """Partial evaluation of itraverse without the traversable. Returns a function that does the traversal.

    The _ in a name is intended to evoke a hole to be filled in with the traversable.

    """
    return lambda t: itraverse(g, t, effect)
