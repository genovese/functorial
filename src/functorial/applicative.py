# trait Functor f => Applicative (f : Type -> Type) where
#     pure : a -> f a
#     map2 : (a -> b -> c) -> f a -> f b -> f c    -- lift2 := map2 h
#     ap   : f (a -> b) -> f a -> f b
#
#     unit : f Unit                                -- Unit equiv ()
#     combine : f a -> f b -> f (a, b)

from __future__ import annotations

from abc             import abstractmethod
from collections.abc import Callable
from typing          import Protocol, runtime_checkable

from .functor        import Functor, map                   # pylint: disable=redefined-builtin
from .functions      import compose, const, curry, identity, pair, fn_eval, eval_with

__all__ = [
    'Applicative', 'map2', 'combine', 'pure',
    'ap', 'lift2', 'ap_first', 'ap_second', 'rev_ap',
    'when', 'unless',  # ATTN: needed?
    'IdentityA',
]


#
# Applicative as a mixin
#

@runtime_checkable
class Applicative(Functor, Protocol):
    @classmethod
    def pure(cls, a):
        raise NotImplementedError

    @abstractmethod
    def map2(self, g, fb):
        ...

    @classmethod
    def unit(cls):
        return cls.pure( () )

    def combine(self, fb):
        return self.map2(pair, fb)

    def ap(self, fb):
        return self.map2(fn_eval, fb)


def map2(g, fa, fb):
    return fa.map2(g, fb)

def combine(fa, fb):
    return fa.combine(fb)

def pure(fa, a):
    return fa.pure(a)

def ap(fa_to_b: Applicative | Callable, fa: Applicative, *fs: Applicative, auto_curry=True) -> Applicative:
    if not isinstance(fa_to_b, Applicative):
        if auto_curry:
            fa_to_b = fa.pure(curry(fa_to_b))
        else:
            fa_to_b = fa.pure(fa_to_b)
    # elif auto_curry:
    #     fa_to_b = fa_to_b.map(curry)  # ATTN: PROVISIONAL

    fb = fa_to_b.ap(fa)      # type: ignore
    for fx in fs:
        fb = fb.ap(fx)
    return fb

# ATTN: by our emerging convention, this should be called map2_, though this name is good too
def lift2[A, B, C](f: Callable[[A, B], C]):
    """Lifts a two-argument function to a mapping of Applicatives.

    This is just the partial application map2(f, _, __).
    The applicatives should be the same type (technically
    one should be a subclass of the other).

    """
    def liftA2(fa: Applicative, fb: Applicative) -> Applicative:
        if not issubclass(fa.__class__, fb.__class__) and not issubclass(fb.__class__, fa.__class__):
            raise TypeError('lift2(f) should be applied to compatible applicatives.')
        return fa.map2(f, fb)

    return liftA2

map2_ = lift2   # Alias for lift2 that matches our naming convention

def rev_ap(fa: Applicative, fa_to_b: Applicative | Callable, auto_curry=True) -> Applicative:
    """A variant of ap with the arguments reversed and effects resolved in the order given.

    Note that rev_ap differs from flip(ap) in the order in which effects
    are resolved. The latter would just remap the argument order into ap,
    but this resolves fa then fa_to_b.

    Unlike ap, this only takes two arguments, but it does do automatic currying
    if auto_curry is True, which is the default.

    Returns the resulting applicative.

    """
    if not isinstance(fa_to_b, Applicative):
        if auto_curry:
            fa_to_b = fa.pure(curry(fa_to_b))
        else:
            fa_to_b = fa.pure(fa_to_b)

    return fa.map2(eval_with, fa_to_b)

# (<*) : f a -> f b -> f a
def ap_first(fa: Applicative, fb: Applicative) -> Applicative:
    "Sequence actions, disgarding the value of the second argument."
    return fa.map2(lambda a, b: a, fb)

# (*>) : f a -> f b -> f b
def ap_second(fa: Applicative, fb: Applicative) -> Applicative:
    "Sequence actions, disgarding the value of the first argument."
    return ap(map(compose(identity, const), fa), fb)

# ATTN: implement when and unless here? Are they useful at all for us, as we don't need it for??
# when : Applicative f => Bool -> f () -> f ()
def when(f: type[Applicative], condition: bool, true_case: Applicative) -> Applicative:
    return map(const(()), true_case) if condition else f.pure(())

# unless : Applicative f => Bool -> f () -> f ()
def unless(f: type[Applicative], condition: bool, false_case: Applicative) -> Applicative:
    return f.pure(()) if condition else map(const(()), false_case)

# A copy of the Identity Functor that is only an Applicative
# This is useful as a default applicative in infrastructure
# modules that would lead to circularity if loading Identity
# module. See also IdentityM in case a default Monad is needed.

class IdentityA[A](Applicative):
    """A default Applicative that mimics Identity without Monad or Traversable.

    This is useful in defaults only infrastructure modules in this package,
    like Monad and Traversable, that Identity actually loads. Users should
    not use this explicitly.

    """
    __match_args__ = ('_value',)

    def __init__(self, x: A):
        self._value = x

    def __str__(self):
        return f'IdentityA {self._value}'

    def __repr__(self):
        return f'IdentityA({self._value})'

    @classmethod
    def run(cls, fa: IdentityA[A]) -> A:
        return fa._value

    def map[B](self, g: Callable[[A], B]) -> IdentityA[B]:
        return IdentityA(g(self._value))

    @classmethod
    def pure(cls, x: A) -> IdentityA[A]:
        return IdentityA(x)

    def map2[B, C](self, g: Callable[[A, B], C], fb: IdentityA[B]) -> IdentityA[C]:
        return IdentityA(g(self._value, fb._value))
