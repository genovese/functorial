#
# trait Applicative m => Monad (m : Type -> Type) where
#     bind : m a -> (a -> m b) -> m b
#     join : m (m a) -> m a
#
# ruff: noqa: E731

from __future__ import annotations

import inspect

from abc             import abstractmethod
from collections.abc import Callable
from functools       import wraps

from .applicative    import Applicative
from .functions      import identity

__all__ = ['Monad', 'join', 'bind', 'do', 'effect',]


#
# Monad as a mixin
#

class Monad(Applicative):

    @abstractmethod
    def bind(self, g: Callable):
        "Sequences monadic effects.  m a -> (a -> m b) -> m b"
        ...

    def join(self):
        "Unwraps one layer of monadic effects. m (m a) -> m a"
        return self.bind(identity)

    def then(self, mb: Monad):
        "Runs two monadic effects, retaining the second. ma -> mb -> mb"
        return self.bind(lambda _: mb)

    @classmethod
    def __do__(cls, make_generator, is_generator):
        if not is_generator:
            ma = make_generator()
            if isinstance(ma, cls):
                return ma
            return cls.pure(ma)

        generator = make_generator()
        try:
            x = generator.send(None)
            while True:
                x = x.bind(generator.send)
        except StopIteration as finished:
            return cls.pure(finished.value)

def join(mma):
    "Monadic join operation: m m a -> m a."
    return mma.join()

def bind(ma, g: Callable):
    "Monadic bind operation: m a -> (a -> m b) -> m b."
    return ma.bind(g)

def do(fn, *cls_eval_args, **eval_kwds):
    """A do block returning a Monadic value.

    This is used to decorate a (generator) function that implements
    the do syntax. The name of the function is bound to the
    **Monadic value** (not the function, see @effect) when the
    do-block passed the given arguments. It is *not* a function.

    ATTN

    """
    if isinstance(fn, type):
        return lambda f: do(f, fn, *cls_eval_args, **eval_kwds)

    if len(cls_eval_args) == 0:
        raise ValueError('@do decorator requires a Monad class argument')

    cls, *eval_args = cls_eval_args

    if not isinstance(cls, type):
        raise ValueError('@do decorator requires a Monad class argument')

    make_generator = lambda: fn(*eval_args, **eval_kwds)
    is_generator = inspect.isgeneratorfunction(fn)
    return cls.__do__(make_generator, is_generator)

    # @wraps(fn)
    # def do_do(*args, **kwds):
    #     make_generator = lambda: fn(*args, **kwds)
    #     return cls.__do__(make_generator)
    #
    # return do_do(*eval_args, **eval_kwds)

# Change name to effect
def effect(fn, cls=None):
    """Returns a function that executes a do-block.

    ATTN

    """
    if isinstance(fn, type):
        return lambda f: effect(f, fn)

    if cls is None or not isinstance(cls, type):
        raise ValueError('@effect decorator requires a Monad class argument')

    @wraps(fn)
    def do_do(*args, **kwds):
        make_generator = lambda: fn(*args, **kwds)
        is_generator = inspect.isgeneratorfunction(fn)
        return cls.__do__(make_generator, is_generator)

    return do_do
