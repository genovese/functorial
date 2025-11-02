"""Contexts that represent IO actions.

An IO action is a piece of code that may return
a value but can also have I/O side effects.

We represent these as functions.

"""
# pylint: disable=arguments-renamed

from __future__ import annotations
from inspect    import isgeneratorfunction
from typing     import Any, Callable, cast

from fp_concepts.Monad     import Monad

type Unit = tuple[()]
type Context = Callable[[Unit], Any]


# ATTN: use functions.compose
def compose[T](f: Callable[[T], Any], g: Callable[..., T]) -> Callable:
    "Composes two functions, returning f after g."
    def f_after_g(*xs, **kwargs):
        return f(g(*xs, **kwargs))

    return f_after_g


class IO[A](Monad):
    """An IO-bound computation.

    We represent an IO object by a function of type:  (Unit -> a) -> Unit -> a,
    where the continuation argument is called a *context*.

    """
    def __init__(self, code: Callable[[Context], Callable[[Unit], A]]):
        self.code = code
        super().__init__()

    def __call__(self, context=lambda *_unit: ()):
        return self.code(context)

    def __str__(self):
        return "<An IO-bound computation>"

    def __repr__(self):  # When printed at the repl, run it!
        return str(IO.unsafe_run_io(self))

    @staticmethod
    def as_context(a: A) -> Callable[[Unit], A]:
        "Wraps a value in a context that just returns that value."
        return lambda *_unit: a

    @classmethod
    def unsafe_run_io(cls, io: IO[A], context=lambda *_unit: ()) -> A:
        """Actually runs an IO-bound computation.

        This is intended only to be used at the *end* of a computation,
        but this is a compromise to the capabilities of an eager Python.

        """
        return io(context)(())

    @staticmethod
    def run(io: IO[A], context=lambda *_unit: ()) -> IO[A]:
        """Actually runs an IO-bound computation.

        This is intended only to be used at the *end* of a computation,
        but this is a compromise to the capabilities of an eager Python.

        """
        value = io(context)(())
        return IO.pure(value)

    # Convert an arbitrary code block to an IO computation

    @classmethod
    def pure_py[B](cls, code: Callable[[], B]) -> IO[B]:
        """Wraps an arbitrary code block (nullary function) in an IO object.

        If the code returns None, the return value is replaced with Unit.
        (Note: this is not fully reflected in the return value type,
        which should be Unit only when B is NoneType.)

        """
        def _ret_unit(*_unit):
            v = code()
            if v is None:
                return ()
            return v

        return IO(lambda _context: _ret_unit)

    # Implements the Functor Trait

    def map[B](self, f: Callable[[A], B]) -> IO[B]:
        "Maps a function over an IO computation."
        def _mapped(context: Context) -> Callable[[Unit], B]:
            return cast(Callable[[Unit], B], compose(f, self.code(context)))

        return IO(_mapped)

    # Implements the Applicative trait

    @classmethod
    def pure(cls, a: A) -> IO[A]:
        "Wraps a pure value in an IO object."
        return IO(lambda _context: lambda *_unit: a)

    def map2[B, C](self, g: Callable[[A, B], C], fb: IO[B]) -> IO[C]:
        "Maps a binary function over two IO objects to create a new IO object."
        def _mapped(context: Callable[[Unit], C]) -> Callable[[Unit], C]:
            def _inner_mapped(*_unit):
                a = IO.unsafe_run_io(self, context)
                b = IO.unsafe_run_io(fb, context)
                return g(a, b)
            return _inner_mapped

        return cast(IO[C], IO(_mapped))

    # Implements the Monad trait

    def bind[B](self, a_to_io_b: Callable[[A], IO[B]]) -> IO[B]:
        "Implements the Monad trait."
        def _bound(context):
            return a_to_io_b(IO.unsafe_run_io(self, context))

        return IO(_bound)

    @classmethod
    def __do__(cls, make_generator) -> IO[A]:
        "Implements do-notation for IO"

        if not isgeneratorfunction(make_generator):
            # The wrapped function has no yield
            return IO.pure_py(make_generator)

        def f(context=lambda *_unit: ()):
            def _f_do_bound(*_unit):
                generator = make_generator()
                try:
                    x = generator.send(None)
                    while True:
                        x = generator.send(IO.unsafe_run_io(x, context))
                except StopIteration as finished:
                    if isinstance(finished.value, IO):
                        return IO.unsafe_run_io(finished.value)
                    return finished.value

            return _f_do_bound

        return IO(f)
