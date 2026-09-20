"""Controlled scheduling for Applicative functors.

Adapted from https://hackage.haskell.org/package/tree-traversals

"""

from __future__ import annotations
from typing     import Callable

from functorial.applicative import Applicative, ap, rev_ap
from functorial.functor     import map                     # pylint: disable=redefined-builtin
from functorial.functions   import compose_, identity

__all__ = [
    'Phases',
]

# For each constructed Phases class, we store its
# corresponding Applicative, PhasesLift, and PhasesAp
phases_registry: dict[type, tuple[type, type, type]] = {}


class Phases:
    def __init__(self, f: type[Applicative]):
        self._effect = f

    def runForward(self):
        match self:
            case PhaseLift(ma):
                return ma

            case PhaseAp(mf, tx):
                return ap(mf, tx.runForward())

    def runBackward(self):
        match self:
            case PhaseLift(ma):
                return ma

            case PhaseAp(mf, tx):
                return rev_ap(tx.runBackward(), mf)

    @property
    def effect(self):
        return self._effect

    @classmethod
    def now(cls, fa: Applicative):
        """Schedules an action to run in the current phase."""
        return PhaseLift(fa)

    @classmethod
    def delay(cls, pa: Phases):
        """Delays actions by one phase."""
        f = pa._effect                     # pylint: disable=protected-access
        return PhaseAp(f.pure(identity), pa)

    @classmethod
    def later(cls, fa: Applicative):
        """Schedules an action to run in the next phase."""
        return cls.delay(cls.now(fa))

    # Functor implementation
    def map(self, g: Callable):
        match self:
            case PhaseLift(ma):
                return PhaseLift(map(g, ma))

            case PhaseAp(mf, tx):
                return PhaseAp(map(compose_(g), mf), tx)

    @classmethod
    def pure(cls, a):
        # Hmmm: need to get the applicative type
        ...

    def map2(self, g, fb):
        ...

class PhaseLift(Phases):
    __match_args__ = ('fa',)

    def __init__(self, fa: Applicative):
        self.fa = fa
        super().__init__(fa.__class__)

class PhaseAp(Phases):
    __match_args__ = ('fa_to_b', 'phase_a')

    def __init__(self, fa_to_b: Applicative, pa: Phases):
        self.fa_to_b = fa_to_b
        self.phase_a = pa
        super().__init__(fa_to_b.__class__)
