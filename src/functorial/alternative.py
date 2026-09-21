# trait Applicative f => Alternative (f : Type -> Type) where
#     empty : f a
#     alt   : f a -> f a -> f a

from __future__   import annotations

from abc          import abstractmethod
from typing       import TYPE_CHECKING, Protocol, runtime_checkable

from .applicative import Applicative

if TYPE_CHECKING:
    from .maybe import Maybe

__all__ = ['Alternative', 'alt', 'guard', 'optional']


#
# Alternative as a mixin
#

# ATTN: Should this have a type parameter or be handled like Applicative?

@runtime_checkable
class Alternative[A](Applicative, Protocol):
    @classmethod
    def empty(cls) -> Alternative[A]:
        raise NotImplementedError

    @abstractmethod
    def alt(self, fb: Alternative[A]) -> Alternative[A]:
        ...

    def optional(self) -> Alternative[Maybe[A]]:
        """Wraps a possibly-failing computation so it always succeeds, with Some(a) or Nothing().

        optional : Alternative f => f a -> f (Maybe a)

        """
        # We use a deferred import here. Because Maybe subclasses
        # Alternative, importing it at the module level would create
        # a circular import.
        from .maybe import Nothing, Some

        return self.map(Some).alt(self.pure(Nothing()))

def alt[A](fa: Alternative[A], fb: Alternative[A]) -> Alternative[A]:
    return fa.alt(fb)

def guard(f: type[Alternative], condition: bool) -> Alternative[tuple[()]]:  # ATTN: type Unit = tuple[()]
    return f.unit() if condition else f.empty()

def optional[A](fa: Alternative[A]) -> Alternative[Maybe[A]]:
    return fa.optional()

# ATTN: Include some and many?  Can we implement them?
