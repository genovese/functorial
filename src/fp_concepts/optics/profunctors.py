"""Concrete profunctors for use in optics methods.

"""
from __future__    import annotations

from typing        import Callable, Self

from ..bicofunctor import Bicofunctor
from ..const       import Const, runConst, makeConst, typeConst
from ..either      import either_, Left, Right
from ..functor     import Functor, lift, map                   # pylint: disable=redefined-builtin
from ..identity    import Identity
from ..maybe       import Nothing
from ..monoids     import Monoid
from ..pair        import Pair
from ..functions   import Function, compose, const, fst, snd
from ..utils       import MissingMonoid, eff

from .choice       import Choice
from .cochoice     import Cochoice
from .costrong     import Costrong
from .strong       import Strong

__all__ = ['Forget', 'ForgetM', 'Star', 'Costar']


#
# Forget is a profunctor whose second argument is a phantom type (ignored)
#
# newtype Forget r a b = Forget { runForget :: a -> r }
#
# This is isomorphic to Star (Const r) but arises enough that it is
# worth having a name for it.

class Forget[R, A](Strong, Cochoice, Choice, Bicofunctor):
    """A profunctor representing a mapping to a fixed type.

    The second type argument is a phantom type (i.e., ignored).

    newtype Forget r a b = Forget { runForget :: a -> r }

    This is isomorphic to Star (Const r) but arises enough that it is
    worth having a name for it.

    We extract the enclosed function with Forget.run.

    To act as a Choice, this needs a default argument supplied at
    construction. This also needs a Monoid interpretation of r for
    other constructions, so an optional Monoid can be supplied, and
    the default is then Monoid.munit. If not supplied, the Monoid is
    the generic Collect by default.

    """
    def __init__(
            self,
            a_to_r: Callable[[A], R],
            monoid: Monoid = MissingMonoid('Forget used as Choice or Bicofunctor')
    ):
        self._a_to_r = Function(a_to_r)
        self._monoid = monoid
        super().__init__()

    @classmethod
    def run(cls, fg: Self):
        """Extracts the enclosed function."""
        return fg._a_to_r    # pylint: disable=protected-access

    def dimap(self, f, _):
        return Forget(compose(self._a_to_r, f), self._monoid)

    def into_first(self):
        return Forget(compose(self._a_to_r, fst), self._monoid)

    def into_second(self):
        return Forget(compose(self._a_to_r, snd), self._monoid)

    def unleft(self):
        return Forget(compose(Left, self._a_to_r), self._monoid)

    def unright(self):
        return Forget(compose(Right, self._a_to_r), self._monoid)

    def bicomap[B](self, f: Callable[[B], A], _g: Callable) -> Forget[R, B]:
        return Forget(compose(self._a_to_r, f), self._monoid)

    def into_left(self):
        return Forget(either_(self._a_to_r, const(self._monoid.munit)), self._monoid)

    def into_right(self):
        return Forget(either_(const(self._monoid.munit), self._a_to_r), self._monoid)

    def wander(self, f):   # wander : Applicative f => (a -> f b) -> (s -> f t)
        cls = typeConst(self._monoid)
        g = eff(makeConst(self._monoid), self._a_to_r, effect=cls)
        return Forget(compose(runConst, f(g)), self._monoid)

    def visit(self, f):
        cls = typeConst(self._monoid)
        g = eff(makeConst(self._monoid), self._a_to_r, effect=cls)
        pure = Const(self._monoid.munit, self._monoid).pure  # Could use as is, but use the function
        return Forget(compose(runConst, lambda s: f(pure, g, s)), self._monoid)

class ForgetM[R, A](Strong, Cochoice, Choice, Bicofunctor):
    """A profunctor representing a mapping to a fixed type.

    The second type argument is a phantom type (i.e., ignored).

    newtype ForgetM r a b = ForgetM { runForgetM :: a -> Maybe r }

    This is isomorphic to Star (Const r) but arises enough that it is
    worth having a name for it.

    We extract the enclosed function with Forget.run.

    """
    def __init__(self, r_to_a: Callable[[A], R]):
        self._a_to_mr = Function(r_to_a)
        super().__init__()

    @classmethod
    def run(cls, fg):
        return fg._a_to_mr    # pylint: disable=protected-access

    def dimap(self, f, _):
        return ForgetM(compose(self._a_to_mr, f))

    def into_first(self):
        return ForgetM(compose(self._a_to_mr, fst))

    def into_second(self):
        return ForgetM(compose(self._a_to_mr, snd))

    def into_left(self):
        return ForgetM(either_(self._a_to_mr, const(Nothing())))

    def into_right(self):
        return ForgetM(either_(const(Nothing()), self._a_to_mr))

    def unleft(self):
        return ForgetM(compose(Left, self._a_to_mr))

    def unright(self):
        return ForgetM(compose(Right, self._a_to_mr))

    def bicomap[B](self, f: Callable[[B], A], _g: Callable) -> ForgetM[R, B]:
        return ForgetM(compose(self._a_to_mr, f))

    def visit(self, f):
        """Converts a VL affine traversal into a ForgetM profunctor.

        f : Functor g => (r -> g r) -> (a -> g b) -> s -> g t

        The focus function must be wrapped so that the .map() call inside
        the visit function (which rebuilds the structure) becomes a no-op.
        We use a local Const-like wrapper for this: its .map() ignores the
        function and returns itself, so only the extracted Maybe value matters.
        """
        class _ConstM:
            __slots__ = ('_v',)
            def __init__(self, v):      self._v = v
            def map(self, _):           return self

        fn = self._a_to_mr
        g = lambda a: _ConstM(fn(a))
        point = lambda _: _ConstM(Nothing())
        return ForgetM(lambda s: f(point, g, s)._v)

#
# Star is a profunctor that lifts an arrow a -> f b into a profunctor.
#
# newtype Star f a b = Star { runStar :: a -> f b }
#
# This is the canonical profunctor for Traversal and AffineTraversal
# optics. The Strong instance requires only that f be a Functor, but
# Choice, wander, and visit require that f be Applicative.
#

class Star[A, B](Strong, Choice):
    """Profunctor lifted from a function a -> f b for a Functor/Applicative f.

    newtype Star f a b = Star { runStar :: a -> f b }

    This is the canonical profunctor for Traversal (via wander) and
    AffineTraversal (via visit) optics.

    Strong instance requires only that f be a Functor. Choice
    (into_left/into_right), wander (used by Traversal), and visit
    (used by AffineTraversal) all require that f be an Applicative.

    Use Star.run(s) to extract the enclosed function from a Star
    object s.

    """
    def __init__(self, a_to_fb: Callable[[A], Functor[B]], effect: type[Functor] = Identity):
        self._fn = Function(a_to_fb)
        self._functor = effect
        super().__init__()

    @classmethod
    def run(cls, s: Star) -> Callable:
        """Extracts the wrapped function from a Star object."""
        return s._fn    # pylint: disable=protected-access

    def dimap(self, f, g):
        # dimap f g (Star h) = Star (fmap g . h . f)
        return Star(compose(lift(g), self._fn, f), self._functor)

    # Strong requires only a Functor f

    def into_first(self):
        # into_first (Star k) = Star $ \(a, c) -> fmap (,c) (k a)
        fn = self._fn

        def k(ac):
            a, c = ac
            return map(lambda b: Pair(b, c), fn(a))
        return Star(k, self._functor)

    def into_second(self):
        # into_second (Star k) = Star $ \(c, a) -> fmap (c,) (k a)
        fn = self._fn

        def k(ca):
            c, a = ca
            return map(lambda b: Pair(c, b), fn(a))
        return Star(k, self._functor)

    # Choice also require an Applicative f

    def into_left(self):
        # into_left (Star f) = Star $ either (fmap Left . f) (pure . Right)
        g = either_(compose(lift(Left), self._fn), compose(self._functor.pure, Right))
        return Star(g, self._functor)

    def into_right(self):
        # into_right (Star f) = Star $ either (pure . Left) (fmap Right . f)
        g = either_(compose(self._functor.pure, Left), compose(lift(Right), self._fn))
        return Star(g, self._functor)

    # Methods for Traversal and AffineTraversal support

    def wander(self, g):
        """Converts a VL traversal into a profunctor.

        g : Applicative f => (a -> f b) -> (s -> f t)

        """
        return Star(g(self._fn), self._functor)

    def visit(self, g):
        """Converts a VL affine traversal into a Star profunctor.

        g : Applicative f => (forall r. r -> f r) -> (a -> f b) -> s -> f t

        The visit function g takes point (pure for the functor), the focus
        function, and the structure s — all three arguments at once.
        """
        return Star(lambda s: g(self._functor.pure, self._fn, s), self._functor)


#
# Costar is the dual profunctor, lowering f a -> b into a profunctor.
#
# newtype Costar f a b = Costar { runCostar :: f a -> b }
#
# This is the right adjoint to Star. Its primary uses are:
#   - Grate optics (via the closed method, which requires Distributive f)
#   - Comonad-based optics (via Costrong, which requires Comonad f)
#
# Common specialisations:
#   Costar Identity a b  ≅  a -> b        (plain function)
#   Costar Maybe    a b  ≅  Maybe a -> b  (total handler for optional)
#

class Costar[F, A, B](Costrong):
    """Profunctor lowering a function f a -> b for a Functor f.

    newtype Costar f a b = Costar { runCostar :: f a -> b }

    This is the right adjoint to Star (the dual Kleisli construction).

    Profunctor:  dimap requires Functor f for the contravariant (lmap) direction.
    Costrong:    unfirst/unsecond require Comonad f (extract + extend).
    closed:      requires Distributive f; used for Grate optics.

    Use Costar.run(c) to extract the enclosed function.

    """
    def __init__(self, fa_to_b: Callable, effect: type):
        self._fn = Function(fa_to_b)
        self._functor = effect
        super().__init__()

    @classmethod
    def run(cls, c: Costar) -> Callable:
        """Extracts the enclosed function."""
        return c._fn    # pylint: disable=protected-access

    def dimap(self, f, g):
        # dimap f g (Costar k) = Costar (g . k . fmap f)
        # f : C -> A, so fmap f : F[C] -> F[A]  (contravariant in A for Costar)
        return Costar(compose(g, self._fn, lift(f)), self._functor)

    # Costrong: requires Comonad f (extract + extend)
    #
    # unfirst  :: Costar f (a, c) (b, c) -> Costar f a b
    # unsecond :: Costar f (c, a) (c, b) -> Costar f a b
    #
    # Haskell:
    #   unfirst (Costar k) = Costar $ fst . k . extend (\w -> (extract w, ???))
    #
    # The second component is a phantom: its type is fixed but its value is
    # never used by the surrounding fst. We use a sentinel object, which is
    # safe because the wrapped function k can only inspect the type, not
    # evaluate the second component in a well-typed program.

    def unfirst(self):
        f = self._functor
        if not (hasattr(f, 'extract') and hasattr(f, 'extend')):
            raise TypeError(
                f'{f.__name__} is not a Comonad; Costar.unfirst requires extract and extend'
            )
        fn = self._fn
        _phantom = object()

        def k(fa):
            extended = f.extend(lambda w: Pair(f.extract(w), _phantom), fa)
            return fst(fn(extended))
        return Costar(k, f)

    def unsecond(self):
        f = self._functor
        if not (hasattr(f, 'extract') and hasattr(f, 'extend')):
            raise TypeError(
                f'{f.__name__} is not a Comonad; Costar.unsecond requires extract and extend'
            )
        fn = self._fn
        _phantom = object()

        def k(fa):
            extended = f.extend(lambda w: Pair(_phantom, f.extract(w)), fa)
            return snd(fn(extended))
        return Costar(k, f)

    # Closed: for Grate optics, requires Distributive f
    #
    # closed :: Costar f a b -> Costar f (x -> a) (x -> b)
    # closed (Costar f) = Costar $ \g x -> f (fmap ($ x) g)
    #
    # Here g : f [x -> a] and we produce x -> b by applying each (x -> a) to x
    # via fmap, yielding f a, then feeding that into f.
    # This uses only fmap and works when f distributes over functions.

    def closed(self):
        fn = self._fn

        def k(g):              # g :: F[x -> a]
            return lambda x: fn(map(lambda h: h(x), g))  # fmap ($ x) g :: F[a]
        return Costar(k, self._functor)
