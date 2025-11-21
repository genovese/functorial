#
# The Maybe Monad
#
# type Maybe a = Nothing | Some a
#
# This represents a context in which a value of a particular type
# may be present or may be missing. It is a Functor, an Applicative,
# and a Monad.
#
# We can destructure Maybe's with pattern matching or the maybe
# function, the functions isNone and isSome provide (type guarding)
# predicates.
#
# See List for utility functions map_maybe and cat_maybes.
#
# ruff: noqa: N801, N802, E731
#

from __future__ import annotations

from abc             import ABC, abstractmethod
from collections.abc import Callable
from typing          import TypeGuard, cast

from .Alternative import Alternative
from .Applicative import Applicative
from .Functor     import map               # pylint: disable=redefined-builtin
from .Monad       import Monad
from .Traversable import Traversable
from .Unit        import Unit

__all__ = ['Maybe', 'Nothing', 'Some', 'maybe', 'maybe_', 'isNone', 'isSome',]


# ATTN: If making Nothing a ContingentSingletonFromABC
# We'll need to rely only on the protocols here and not inheritance except for ABC.
# Can add compliance checks at the end.
#
# from singleton import ContingentSingletonFromABC
#
# We'll need to rely only on the protocols here and not inheritance except for ABC
#               vvv      vvv         vvv
class Maybe[A](Monad, Traversable, Alternative, ABC):
    @abstractmethod
    def get(self, default: A) -> A:
        ...

    @abstractmethod
    def map[B](self, g: Callable[[A], B]) -> Maybe[B]:
        ...

    @classmethod
    def pure(cls, a: A) -> Maybe[A]:
        return Some(a)

    @abstractmethod
    def map2[B, C](self, g: Callable[[A, B], C], fb: Maybe[B]) -> Maybe[C]:
        ...

    @property
    def empty(self):
        return Nothing()

    def alt(self, fb: Maybe[A]) -> Maybe[A]:   # type: ignore
        if not self:
            return fb
        return self

    @abstractmethod
    def bind[B](self, f: Callable[[A], Maybe[B]]) -> Maybe[B]:
        ...

    @classmethod
    def __do__(cls, make_generator, is_generator) -> Maybe[A]:
        if not is_generator:
            ma = make_generator()
            if isinstance(ma, cls):
                return ma
            return cls.pure(ma)

        generator = make_generator()
        try:
            x = generator.send(None)
            while True:
                if isNone(x):
                    return x
                x = x.bind(generator.send)
        except StopIteration as finished:
            return Some(finished.value)

    @abstractmethod
    def traverse(self, f: type[Applicative], g: Callable[[A], Applicative]) -> Applicative:  # g : a -> f b
        ...

class Some[A](Maybe[A]):
    __match_args__ = ('_value',)

    def __init__(self, value: A):
        self._value = value

    def __str__(self):
        return f'Some {str(self._value)}'

    def __repr__(self):
        return f'Some({repr(self._value)})'

    def __eq__(self, other):
        if isinstance(other, Some):
            return self._value == other._value
        return False

    def __bool__(self):
        return True

    def get(self, _default: A) -> A:
        return self._value

    def map[B](self, g: Callable[[A], B]) -> Maybe[B]:
        try:
            return Some(g(self._value))
        except Exception:
            return Nothing()

    def map2[B, C](self, g: Callable[[A, B], C], fb: Maybe[B]) -> Maybe[C]:
        if isinstance(fb, Nothing):
            return fb
        return Some(g(self._value, fb._value))  # type: ignore

    def bind[B](self, f: Callable[[A], Maybe[B]]) -> Maybe[B]:
        return f(self._value)

    def traverse(self, _f: type[Applicative], g: Callable[[A], Applicative]) -> Applicative:  # g : a -> f b
        return map(Some, g(self._value))

    # def itraverse[I](self, _f: type[Applicative], g: Callable[[I, A], Applicative]) -> Applicative:
    def itraverse(self, _f: type[Applicative], g: Callable[[Unit, A], Applicative]) -> Applicative:
        # g : () -> a -> f b
        return map(Some, g((), self._value))

class Nothing[A](Maybe[A]):   # The name None is already taken
    def __str__(self):
        return 'None'

    def __repr__(self):
        return 'Nothing()'

    def __eq__(self, other):
        if isinstance(other, Nothing):
            return True
        return False

    def __bool__(self):
        return False

    def get(self, default: A) -> A:
        return default

    def map[B](self, _g: Callable[[A], B]) -> Maybe[B]:
        return cast(Nothing[B], self)

    def map2[B, C](self, _g: Callable[[A, B], C], _fb: Maybe[B]) -> Maybe[C]:
        return cast(Nothing[C], self)

    def bind[B](self, _f: Callable[[A], Maybe[B]]) -> Maybe[B]:
        return cast(Nothing[B], self)

    def traverse(self, f: type[Applicative], _g: Callable[[A], Applicative]) -> Applicative:
        # g : a -> f b
        return f.pure(self)

    def itraverse[I](self, f: type[Applicative], _g: Callable[[I, A], Applicative]) -> Applicative:
        # g : () -> a -> f b
        return f.pure(self)

def isNone[A](x: Maybe[A]) -> TypeGuard[Nothing]:
    return isinstance(x, Nothing)

def isSome[A](x: Maybe[A]) -> TypeGuard[Some]:
    return isinstance(x, Some)

def maybe[A, B](default: B, f: Callable[[A], B], m: Maybe[A]) -> B:
    """Extracts a transformed value from a Maybe by case analysis.

    If given a Nothing, return the specified value (i.e., apply the
    constant function to its value); if given a Some,
    apply the function f. Returns the resulting value.

    """
    match m:
        case Nothing():
            return default
        case Some(b):
            return f(b)
        case _:
            raise TypeError('maybe applied to a non-Maybe type')

def maybe_[A, B](default: B, f: Callable[[A], B]) -> Callable[[Maybe[A]], B]:
    """Partial application of maybe on two arguments; returns the function m :--> maybe(f, g, m).

    This partial is a common use case for maybe, so this is provided as a convenience.
    The _ in the name is supposed to evoke the hole in the last argument.

    """
    return lambda m: maybe(default, f, m)
