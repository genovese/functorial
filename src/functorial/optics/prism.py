"""A prism is an optic that represents a generalized constructor.

Given some data of type b (a part or representative of a structure),
a prism can "build" or "review" the structure of type t.

Given a structure of type s, a prism can "match" or "preview" to
extract a part/representatie of the structure of type a -- if it
exists -- else giving a whole structure of type t.

The canonical construction of a prism (see `prism` below) takes
a build and a match function accordingly.

"""

from __future__   import annotations

from typing       import Callable, cast

from .choice      import Choice, into_right
from ..either     import Either, Left, Right, either
from ..maybe      import Nothing, Some
from .optic       import Optic, OpticIs
from ..functions  import identity

__all__ = ['Prism', 'prism', 'left', 'right', 'some', 'nothing', 'only']


class Prism[A, B, S, T](Optic, optic_is=OpticIs.PRISM):
    """An optic representing a generalized constructor.

    We can also think of a prism as focusing on part of a sum type,
    and in this sense it is dual to a lens.

    """
    def __init__(self, cab_to_cst: Callable[[Choice[A, B]], Choice[S, T]], opt_type=None):
        self._cab_to_cst: Callable[[Choice[A, B]], Choice[S, T]] = cab_to_cst
        super().__init__(cab_to_cst, opt_type if opt_type is not None else OpticIs.PRISM)

# class SimplePrism[A, B](Prism[A, B, A, B]):
#     def __init__(self, cab_to_cab: Callable[[Choice[A, B]], Choice[A, B]]):
#         super().__init__(cab_to_cab)

def prism[A, B, S, T](
        construct: Callable[[B], T],
        match: Callable[[S], Either[T, A]]
) -> Prism[A, B, S, T]:
    """Creates a prism from build and match functions."""
    def the_prism(p_ab: Choice[A, B]) -> Choice[S, T]:
        p_sa_sb: Choice[Either[T, A], Either[T, B]] = into_right(p_ab)
        p = p_sa_sb.dimap(match, lambda esb: either(identity, construct, esb))
        return cast(Choice[S, T], p)

    return Prism(the_prism)

def _left_matcher[A, B, C](x: Either[A, B]) -> Either[Either[C, B], A]:
    match x:
        case Left(y):
            return Right(y)
        case Right(y):
            return Left(Right(y))
        case _:  # Cannot happen with right type input
            raise TypeError('Wrong type for left')

def _right_matcher[A, B, C](x: Either[A, B]) -> Either[Either[A, C], B]:
    match x:
        case Left(y):
            return Left(Left(y))
        case Right(y):
            return Right(y)
        case _:  # Cannot happen with right type input
            raise TypeError('Wrong type for right')

left: Prism = prism(Left, _left_matcher)
right: Prism = prism(Right, _right_matcher)   # why are the types not inferred here mypy??

def _some_match(ma):
    match ma:
        case Some(a):
            return Right(a)
        case _:
            return Left(ma)

some: Prism = prism(Some, _some_match)
some.__doc__ = """Prism focusing on the value inside Some, or no focus if Nothing."""


def _nothing_match(ma):
    match ma:
        case Nothing():
            return Right(None)
        case _:
            return Left(ma)

nothing: Prism = prism(lambda _: Nothing(), _nothing_match)
nothing.__doc__ = """Prism focusing on Nothing (unit focus), or no focus if Some."""


def only(v) -> Prism:
    """Prism matching only the specific value v (unit focus).

    only : Eq a => a -> Prism' a ()

    preview (only v) v  = Some(())
    preview (only v) w  = Nothing()   for w != v
    review  (only v) () = v

    """
    def match_only(x):
        if x == v:
            return Right(())
        return Left(x)

    return prism(lambda _: v, match_only)
