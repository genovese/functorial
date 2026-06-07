#
# ATTN: Provisional
#

from __future__ import annotations

from typing          import Callable

from .applicative    import map2
from .monad          import Monad
from .maybe          import Some, Nothing, maybe
from .functions      import compose, identity

__all__ = ['Lazy',]


class Lazy[A](Monad):
    def __init__(self, thunk: Callable[[], A], value=Nothing()):
        self._thunk = thunk
        self._realized = value
        super().__init__()

    def __call__(self):
        return self.force

    @property
    def force(self):
        if not self._realized:
            self._realized = Some(self._thunk())
        return maybe(None, identity, self._realized)

    def map[B](self, g: Callable[[A], B]):
        return Lazy(compose(g, self._thunk), maybe(Nothing(), g, self._realized))  # type: ignore

    @classmethod
    def pure(cls, a: A):
        return cls(lambda: a)

    def map2(self, g, fb):
        value = map2(g, self._realized, fb._realized)
        return Lazy(lambda: g(self._thunk(), fb._thunk()), value)

    def bind(self, g):
        def thunk():
            mb = g(self._thunk())
            return mb._thunk()
        return Lazy(thunk)
