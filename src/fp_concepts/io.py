"""Contexts that represent IO actions.

An IO action is a piece of code that may return
a value but can also have I/O side effects.

We represent these as functions.

"""
# pylint: disable=arguments-renamed

from __future__ import annotations
from typing     import Any, Callable, cast

from fp_concepts.Monad import Monad

__all__ = ['IO', 'as_io',]


def compose_nullary[T](f: Callable[[T], Any], g: Callable[[], T]) -> Callable:
    "Composes two functions, returning f after g."
    def f_after_g():
        return f(g())

    return f_after_g

def as_io(*xs):
    """Convenience method for wrapping multiple io statements and a value into a lambda.

    This is intended to be used as the body of a (nullary) lambda.
    The lambda cannot be excluded because the unevaluated code must
    be passed to the IO constructor.

    This returns the final argument (the value), which becomes the
    value of the IO object.

    Example:
        >>> x = [
            IO(lambda: as_io(print("foo"), 10)), IO(lambda: as_io(print("bar"), 20)),
            IO(lambda: as_io(print("zap"), print("ok"), 30)),
            IO(lambda: as_io(print("wow"), print("***"), 40))
        ]
        >>> x
        foo
        bar
        zap
        ok
        wow
        ***
        [10, 20, 30, 40]

    """
    return xs[-1]

class IO[A](Monad):
    """An IO-bound computation.

    We represent an IO object by a function of type:  () -> a,
    using a nullary function rather than an explicit unit.

    """
    def __init__(self, code: Callable[[], A]):
        self.code = code
        super().__init__()

    def __call__(self):
        return self.code()

    def __str__(self):
        return "<An IO-bound computation>"

    def __repr__(self):  # When printed at the repl, run it!
        return str(IO.unsafe_run(self))

    # Running an IO value, safely or otherwise

    @classmethod
    def unsafe_run(cls, io: IO[A]) -> A:
        """Actually runs an IO-bound computation.

        This is intended only to be used at the *end* of a computation,
        but this is a compromise to the capabilities of an eager Python.

        """
        return io()

    @classmethod
    def unsafe_ensure_run_io(cls, io: IO[A] | A) -> A:
        "Like unsafe_run but also accepts and returns an A values as is."
        if isinstance(io, cls):
            return cls.unsafe_run(io)
        return cast(A, io)

    @staticmethod
    def run(io: IO[A]) -> IO[A]:
        """Actually runs an IO-bound computation.

        This is intended only to be used at the *end* of a computation,
        but this is a compromise to the capabilities of an eager Python.

        """
        value = io()
        return IO.pure(value)

    # Implements the Functor Trait

    def map[B](self, f: Callable[[A], B]) -> IO[B]:
        "Maps a function over an IO computation."
        return IO(compose_nullary(f, self.code))

    # Implements the Applicative trait

    @classmethod
    def pure(cls, a: A) -> IO[A]:
        "Wraps a pure value in an IO object."
        return IO(lambda: a)

    def map2[B, C](self, g: Callable[[A, B], C], fb: IO[B]) -> IO[C]:
        "Maps a binary function over two IO objects to create a new IO object."
        def _mapped() -> C:
            a = IO.unsafe_run(self)
            b = IO.unsafe_run(fb)
            return g(a, b)

        return cast(IO[C], IO(_mapped))

    # Implements the Monad trait

    def bind[B](self, a_to_io_b: Callable[[A], IO[B]]) -> IO[B]:
        "Implements the Monad trait."
        def _bound():
            return a_to_io_b(IO.unsafe_run(self))

        return IO(_bound)

    def join(self):
        "Unwraps IO (IO a) to IO a."
        # We can do this more efficiently than does the default function.
        # Just call the outer function to get the inner IO object.
        return IO.unsafe_run(self)

    @classmethod
    def __do__(cls, make_generator, is_generator) -> IO[A]:
        "Implements do-notation for IO"

        if not is_generator:
            def _get_value():
                return cls.unsafe_ensure_run_io(make_generator())
            return IO(_get_value)

        def f():
            generator = make_generator()
            try:
                x = generator.send(None)
                while True:
                    x = generator.send(IO.unsafe_run(x))
            except StopIteration as finished:
                if isinstance(finished.value, IO):
                    return IO.unsafe_run(finished.value)
                return finished.value

        return IO(f)
