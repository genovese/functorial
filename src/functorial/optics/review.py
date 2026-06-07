"""Methods for Review functionality: review, preview, preview_with

A Review is a kind of reversed Getter, basically wrapping a
function b -> t. The purpose of having a class for this
would be two-fold:

  1. Ensuring for these objects that we can only use review, not
     preview.

  2. Making the type lattice complete. For instance, Iso is both a
     Review and a Lens, and this would be clear in the composition
     structure.

However, in Python, we can't prevent a user from casting a Prism to
something and calling preview on it, so #1 is not really import. #2
has pedagogical value, but we skip the extra complexity and merely
expose review() which handles most of the functionality we need.

This module provides review, preview, and the derived preview_with
that are implemented using the Tagged profunctor.

"""

from __future__    import annotations

from typing        import Callable, cast

from ..either      import Either, Left, Right
from ..functor     import Functor
from ..maybe       import Some
# from ..profunctor  import dilift
from ..functions   import Function, compose

from .choice       import Choice
from .costrong     import Costrong
from .profunctors  import ForgetM

__all__ = ['Tagged', 'review', 'preview', 'preview_with']

def absurd(*args, **kwargs):
    raise TypeError('Builder is write only; lmap component is a phantom.')


class Tagged[A, B](Choice, Costrong, Functor):
    def __init__(self, b: B):
        self._value = b

    @classmethod
    def run(cls, p):
        """Extracts the tagged value."""
        return p._value    # pylint: disable=protected-access

    def map[C](self, g: Callable[[B], C]) -> Tagged[A, C]:
        return Tagged(g(self._value))

    def dimap[C, D](self, f: Callable[[C], A], g: Callable[[B], D]) -> Tagged[C, D]:
        return cast(Tagged[C, D], Tagged(g(self._value)))

    def into_left[C](self) -> Tagged[Either[A, C], Either[B, C]]:
        return cast(Tagged[Either[A, C], Either[B, C]], Tagged(Left(self._value)))

    def into_right[C](self) -> Tagged[Either[C, A], Either[C, B]]:
        return cast(Tagged[Either[C, A], Either[C, B]], Tagged(Right(self._value)))

    def unfirst(self) -> Tagged[A, B]:
        return cast(Tagged[A, B], Tagged(self._value[0]))   # type: ignore

    def unsecond(self) -> Tagged[A, B]:
        return cast(Tagged[A, B], Tagged(self._value[1]))   # type: ignore

def review(opt):
    """Used to construct a value with the build path of an optic.

    Works with any optic that has a build path: e.g., Prism, Iso, or Review.
    Returns a function b -> t that constructs a t using the build path of
    the given optic.

      review : (Prism s t a b | Iso s t a b) -> b -> t

    Examples:
      + review(some)(42)   == Some(42)
      + review(right)('x') == Right('x')
      + review(left)(1)    == Left(1)
      + 9 >> review(left)  == Left(9)
      + 9 >> review(left) >> preview(left) == Some(9)
      + x >> review(opt) >> preview(opt) == Some(x)

    Implementation note: Runs the optic with the Tagged profunctor,
    which carries only the construction direction and ignores the
    match direction entirely.

    """
    def reviewed(b):
        return Tagged.run(opt(Tagged(b)))

    return Function(reviewed)

def preview_with(opt, f):
    """Used to optionally extract and transform the focus of an optic when it matches.

    Like preview(opt) but applies f to the focus before wrapping in Some.
    Equivalent to map(f, preview(opt)(s)) but without wrapping and unwrapping.
    Works with any Prism, AffineFold, or AffineTraversal. Returns a
    function s -> Maybe b that takes the structure and gives the
    transformed focus f(a) if available, wrapped in a Maybe, i.e.,
    Some(f(a)) or Nothing().

      preview_with : AffineFold s a -> (a -> b) -> s -> Maybe b

      Examples:
        + List.of(0, 1, 2, 3, 4) >> preview_with(ix(3), lambda x: x * 10) == Some(30)
        + preview_with(some, str)(Some(42))  == Some('42')
        + preview_with(some, str)(Nothing()) == Nothing()

    """
    def previewed(s):
        h = ForgetM.run(opt(ForgetM(compose(Some, f))))
        return h(s)

    return Function(previewed)

def preview(opt):
    """Used to optionally extract the focus of an optic when it matches.

    Works with any Prism, AffineFold, or AffineTraversal. Returns a
    function s -> Maybe a that takes the structure and gives the
    focused result if it matches. This function returns Some(a) if
    optic has a focus in s, Nothing() otherwise.

      preview : AffineFold s a -> s -> Maybe a

    Examples:
      + preview(ix(3))(List.of(0, 1, 2, 3, 4))         == Some(3)
      + preview(ix(9))(List.of(0, 1, 2, 3, 4))         == Nothing()
      + List.of(1, 2, 3, 4, 5, 6, 7) >> preview(ix(4)) == Some(5)

      + preview(some)(Some(42))  == Some(42)
      + preview(some)(Nothing()) == Nothing()

      + preview(left)(Left(1))   == Some(1)
      + preview(left)(Right(2))  == Nothing()

    Implementation note: Runs the optic with ForgetM(Some), so only
    the first focus is extracted.

    """
    def previewed(s):
        h = ForgetM.run(opt(ForgetM(Some)))
        return h(s)

    return Function(previewed)

# # Promotes a plain function to a review
# # Probably not needed, but keep until then for reference.
# def builder[B, T](f: Callable[[B], T]):
#     return dilift(cast(Callable[[T], B], absurd), Function(f))
