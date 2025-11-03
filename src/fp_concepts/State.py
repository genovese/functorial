#
# The State Monad
#
# newtype State s a = State { runState : s -> (s, a) }
#
# ruff: noqa: E731

from __future__      import annotations
from collections.abc import Callable

from .Monad          import Monad

__all__ = ['State',]


class GetStateDescriptor:
    "A descriptor to enable class data of type State."
    def __get__(self, obj, objtype=None):
        return objtype(lambda s: (s, s))

class State[S, A](Monad):
    "The State Monad s -> (a, s)"
    def __init__(self, state: Callable[[S], tuple[A, S]]):
        self._state = state
        super().__init__()

    #
    # Utility Constructors
    #

    get: State[S, S] = GetStateDescriptor()         # type: ignore

    @classmethod
    def put(cls, state: S) -> State[S, tuple[()]]:
        "Replaces the state, with a unit value."
        return cls(lambda _s: ((), state))          # type: ignore

    @classmethod
    def derive(cls, derivation: Callable[[S], A]) -> State[S, A]:
        "Generates a value as a function of the current state."
        return cls(lambda s: (derivation(s), s))

    @classmethod
    def modify(cls, modifier: Callable[[S], S]) -> State[S, tuple[()]]:
        "Modifies the current state with a function, giving unit value."
        return cls(lambda s: ((), modifier(s)))     # type: ignore

    #
    # ``Running'' State Transformations
    #

    def run(self, s: S) -> tuple[A, S]:
        "Runs the state monad, producing a final value and state."
        return self._state(s)

    def eval(self, s: S) -> A:
        "Runs the state, producing the final value."
        return self.run(s)[0]

    def exec(self, s: S) -> S:
        "Runs the state, producing the final state."
        return self.run(s)[1]

    #
    # Functor, Applicative, and Monad Implementations
    #

    def map[C](self, g: Callable[[A], C]) -> State[S, C]:
        def g_state(s):
            a, s_prime = self._state(s)
            return (g(a), s_prime)

        return State(g_state)

    @classmethod
    def pure(cls, a):
        return cls(lambda s: (a, s))

    def map2[B, C](self, g: Callable[[A, B], C], fb: State[S, B]) -> State[S, C]:
        def g_state(s):
            a, s1 = self._state(s)
            b, s2 = fb._state(s1)    # pylint: disable=protected-access
            return (g(a, b), s2)

        return State(g_state)

    def bind[B](self, g: Callable[[A], State[S, B]]) -> State[S, B]:
        def bind_state(s):
            a, s1 = self._state(s)
            st = g(a)
            return st._state(s1)    # pylint: disable=protected-access

        return State(bind_state)

    @classmethod
    def __do__(cls, make_generator, is_generator):
        # ATTN: Handle not is_generator case

        def threaded(s):
            generator = make_generator()
            try:
                x = generator.send(None)
                s1 = s
                while True:
                    a, s1 = x._state(s1)    # pylint: disable=protected-access
                    x = generator.send(a)
            except StopIteration as finished:
                return (finished.value, s1)

        return cls(threaded)
