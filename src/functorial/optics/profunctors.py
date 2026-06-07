"""Concrete profunctors used to implement optics methods.

This is the key trick to the entire (sub-)library. Optics are
functions mapping p a b -> p s t for profunctors p (with various
possible constraints) and types a b s t. They work for *any*
suitable profunctor, so to create the optics methods, we simply need
a well-chosen, *concrete* profunctor.

The profunctors in this module (along with a few others, like Tagged
in review.py) serve this purpose. For instance, Forget is used for
Getters and Star for Setters.

"""
from __future__    import annotations

from typing        import Callable, Self

from ..bicofunctor import Bicofunctor
from ..const       import Const, run_const, make_const, type_const
from ..either      import either_, Left, Right
from ..functor     import Functor, lift, map                   # pylint: disable=redefined-builtin
from ..identity    import Identity
from ..maybe       import Nothing
from ..monoids     import Monoid
from ..pair        import Pair
from ..functions   import Function, compose, const, fst, snd
from ..utils       import MissingMonoid, effn

from ..wrappers    import EffectfulFunction

from .choice       import Choice
from .cochoice     import Cochoice
from .costrong     import Costrong
from .strong       import Strong
from .optic        import _pack_index

__all__ = ['Forget', 'ForgetM', 'IndexedForget', 'Indexed', 'IndexedStar', 'Star', 'Costar']


#
# Forget is a profunctor whose second argument is a phantom type (ignored)
#
# newtype Forget r a b = Forget { runForget : a -> r }
#
# This is isomorphic to Star (Const r) but arises enough that it is
# worth having a name for it.

class Forget[R, A](Strong, Cochoice, Choice, Bicofunctor):
    """A profunctor representing a mapping to a fixed type.

    The second type argument is a phantom type, i.e., ignored.

      newtype Forget r a b = Forget { runForget : a -> r }

    This is isomorphic to Star (Const r) but arises enough that it is
    worth having a distinct name/class for it.

    Use Forget.run to extract the enclosed function.

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
        cls = type_const(self._monoid)
        g = effn(make_const(self._monoid), self._a_to_r, effect=cls)
        return Forget(compose(run_const, f(g)), self._monoid)

    def visit(self, f):
        cls = type_const(self._monoid)
        g = effn(make_const(self._monoid), self._a_to_r, effect=cls)
        pure = Const(self._monoid.munit, self._monoid).pure  # Could use as is, but use the function
        return Forget(compose(run_const, lambda s: f(pure, g, s)), self._monoid)

    def iwander(self, f):
        """Applies an indexed VL traversal function, ignoring the index.

        f : (i -> a -> g b) -> s -> g t  for any Applicative g
        """
        h, m = self._a_to_r, self._monoid
        cls_c = type_const(m)

        def go(s):
            ig = EffectfulFunction(lambda _ix, a: Const(h(a), m), cls_c)
            return run_const(f(ig)(s))

        return Forget(go, m)

    def ifold_vl(self, f):
        """Applies an indexed fold VL function, ignoring the index.

        f : (i -> a -> r, Monoid r) -> s -> r
        """
        h, m = self._a_to_r, self._monoid

        def go(s):
            return f(lambda _ix, a: h(a), m)(s)

        return Forget(go, m)

#
# IndexedForget is the indexed analogue of Forget, for iview/icollect/ifold_map_of.
#
# newtype IndexedForget i r a b = IndexedForget { runIndexedForget : i -> a -> r }
#
# Like Forget, the second type argument b is a phantom type (ignored in dimap).
# Unlike Forget, it carries an index i that threads through indexed optic chains
# via _pack_index accumulation.
#

class IndexedForget[I, R, A]:
    """An indexed profunctor for reading actions: wraps i -> a -> r.

    newtype IndexedForget i r a b = IndexedForget { runIndexedForget : i -> a -> r }

    The type argument b is a phantom (ignored). This is the canonical
    profunctor for iview, icollect, and ifold_map_of — the indexed
    counterparts of view, collect, and fold_map_of.

    Like Forget, carries a Monoid for indexed fold actions.
    Use IndexedForget.run(p) to extract the enclosed function.
    """
    def __init__(
            self,
            f: Callable[[I, A], R],
            monoid: Monoid = MissingMonoid('IndexedForget used as fold without a Monoid'),
    ):
        self._f = f
        self._monoid = monoid

    @classmethod
    def run(cls, p: IndexedForget) -> Callable[[I, A], R]:
        """Extracts the enclosed function."""
        return p._f    # pylint: disable=protected-access

    def dimap(self, pre, _post):
        # dimap f _ (IndexedForget h) = IndexedForget (\i s -> h i (f s))
        # _post is phantom: b is never observed
        f = self._f
        return IndexedForget(lambda i, s: f(i, pre(s)), self._monoid)

    # Strong: into_first/into_second project the focus from a pair

    def into_first(self):
        f = self._f
        return IndexedForget(lambda i, sc: f(i, sc[0]), self._monoid)

    def into_second(self):
        f = self._f
        return IndexedForget(lambda i, cs: f(i, cs[1]), self._monoid)

    # Choice: into_left/into_right project the focus from a sum,
    # returning munit on the absent branch

    def into_left(self):
        f, m = self._f, self._monoid
        return IndexedForget(
            lambda i, ac: either_(lambda a: f(i, a), const(m.munit))(ac), m
        )

    def into_right(self):
        f, m = self._f, self._monoid
        return IndexedForget(
            lambda i, ca: either_(const(m.munit), lambda a: f(i, a))(ca), m
        )

    def iwander(self, f):
        """Applies an indexed VL traversal function with index accumulation.

        f : (i -> a -> g b) -> s -> g t  for any Applicative g
        """
        h, m = self._f, self._monoid
        cls_c = type_const(m)

        def go(i_acc, s):
            ig = EffectfulFunction(lambda ix, a: Const(h(_pack_index(i_acc, ix), a), m), cls_c)
            return run_const(f(ig)(s))

        return IndexedForget(go, m)

    def ifold_vl(self, f):
        """Applies an indexed fold VL function with index accumulation.

        f : (i -> a -> r, Monoid r) -> s -> r
        """
        h, m = self._f, self._monoid

        def go(i_acc, s):
            return f(lambda ix, a: h(_pack_index(i_acc, ix), a), m)(s)

        return IndexedForget(go, m)


class ForgetM[R, A](Strong, Cochoice, Choice, Bicofunctor):
    """A profunctor representing a mapping to a fixed type.

    The second type argument is a phantom type (i.e., ignored).

    newtype ForgetM r a b = ForgetM { runForgetM : a -> Maybe r }

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
# Indexed is a profunctor wrapping i -> a -> b.
#
# newtype Indexed i a b = Indexed { runIndexed : i -> a -> b }
#
# An indexed profunctor carries an index i alongside each value of type a.
# It threads through indexed optic composition via pair accumulation:
# composing Indexed i with Indexed j yields Indexed (i, j) via reindex.
#
# It has Strong and Choice instances (index threads through untouched),
# and a reindex combinator for changing the index type.
#

class Indexed[I, A, B](Strong, Choice):
    """An indexed profunctor wrapping i -> a -> b.

    Indexed i a b represents a profunctor where each value of type a is
    paired with an index of type i, producing a result of type b.

    newtype Indexed i a b = Indexed { runIndexed : i -> a -> b }

    When composing two indexed optics (indices I and J), the indices
    accumulate as pairs (I, J) via reindex. The Strong and Choice
    instances thread the index through untouched.

    Use Indexed.run(p) to extract the enclosed function.
    """
    def __init__(self, f: Callable[[I, A], B]) -> None:
        self._f = Function(f)
        super().__init__()

    @classmethod
    def run(cls, p: Indexed) -> Callable[[I, A], B]:
        """Extracts the enclosed function."""
        return p._f    # pylint: disable=protected-access

    def dimap(self, pre, post):
        # dimap f g (Indexed h) = Indexed (\i a -> g (h i (f a)))
        f = self._f
        return Indexed(lambda i, a: post(f(i, pre(a))))

    # Strong: index threads through untouched

    def into_first(self):
        # into_first (Indexed h) = Indexed (\i (a, c) -> (h i a, c))
        f = self._f
        return Indexed(lambda i, ac: Pair(f(i, ac[0]), ac[1]))

    def into_second(self):
        # into_second (Indexed h) = Indexed (\i (c, a) -> (c, h i a))
        f = self._f
        return Indexed(lambda i, ca: Pair(ca[0], f(i, ca[1])))

    # Choice: index threads through untouched

    def into_left(self):
        # into_left (Indexed h) = Indexed (\i -> either (Left . h i) Right)
        f = self._f
        return Indexed(lambda i, ac: either_(lambda a: Left(f(i, a)), Right)(ac))

    def into_right(self):
        # into_right (Indexed h) = Indexed (\i -> either Left (Right . h i))
        f = self._f
        return Indexed(lambda i, ca: either_(Left, lambda a: Right(f(i, a)))(ca))

    def reindex[J](self, g: Callable[[J], I]) -> Indexed[J, A, B]:
        """Changes the index type via a contravariant function J -> I.

        reindex g (Indexed h) = Indexed (\\j a -> h (g j) a)

        Used in indexed optic composition: when an outer indexed optic
        (index I) is composed with an inner indexed optic (index J), the
        combined index is (I, J). Each side uses reindex to project out
        its own component from the pair.
        """
        f = self._f
        return Indexed(lambda j, a: f(g(j), a))

    def iwander(self, f):
        """Applies an indexed VL traversal function using Identity as the Applicative.

        f : (i -> a -> g b) -> s -> g t  for any Applicative g
        """
        g = self._f

        def go(i_acc, s):
            ig = EffectfulFunction(lambda ix, a: Identity(g(_pack_index(i_acc, ix), a)), Identity)
            return Identity.run(f(ig)(s))

        return Indexed(go)


#
# IndexedStar is the indexed analogue of Star, for itraverse_of / effectful traversal.
#
# newtype IndexedStar i f a b = IndexedStar { runIndexedStar : i -> a -> f b }
#
# Like Star, it carries an effect class (for pure/Applicative operations).
# Like Indexed, it threads an accumulated index i through optic chains.
#

class IndexedStar[I, A, B](Strong, Choice):
    """Indexed profunctor for effectful traversal actions: wraps i -> a -> f b.

    newtype IndexedStar i f a b = IndexedStar { runIndexedStar : i -> a -> f b }

    The canonical profunctor for itraverse_of.  Like Star, requires
    Functor f for Strong and Applicative f for Choice and wander.

    Use IndexedStar.run(p) to extract the enclosed function.
    """
    def __init__(self, f: Callable[[I, A], Functor[B]], effect: type[Functor] = Identity):
        self._f = f
        self._effect = effect
        super().__init__()

    @classmethod
    def run(cls, p: IndexedStar) -> Callable[[I, A], Functor[B]]:
        """Extracts the enclosed function."""
        return p._f    # pylint: disable=protected-access

    def dimap(self, pre, post):
        f, eff = self._f, self._effect
        return IndexedStar(lambda i, s: map(post, f(i, pre(s))), eff)

    # Strong: index threads through untouched

    def into_first(self):
        f, eff = self._f, self._effect

        def k(i, ac):
            a, c = ac
            return map(lambda b: Pair(b, c), f(i, a))
        return IndexedStar(k, eff)

    def into_second(self):
        f, eff = self._f, self._effect

        def k(i, ca):
            c, a = ca
            return map(lambda b: Pair(c, b), f(i, a))
        return IndexedStar(k, eff)

    # Choice: requires Applicative f

    def into_left(self):
        f, eff = self._f, self._effect

        def k(i, ac):
            return either_(compose(lift(Left), f(i)), compose(eff.pure, Right))(ac)
        return IndexedStar(k, eff)

    def into_right(self):
        f, eff = self._f, self._effect

        def k(i, ca):
            return either_(compose(eff.pure, Left), compose(lift(Right), f(i)))(ca)
        return IndexedStar(k, eff)

    def iwander(self, f):
        """Applies an indexed VL traversal function with indexed effectful function.

        f : (i -> a -> g b) -> s -> g t  for any Applicative g
        """
        g, eff = self._f, self._effect

        def go(i_acc, s):
            ig = EffectfulFunction(lambda ix, a: g(_pack_index(i_acc, ix), a), eff)
            return f(ig)(s)
        return IndexedStar(go, eff)


#
# Star is a profunctor that lifts an arrow a -> f b into a profunctor.
#
# newtype Star f a b = Star { runStar : a -> f b }
#
# This is the canonical profunctor for Traversal and AffineTraversal
# optics. The Strong instance requires only that f be a Functor, but
# Choice, wander, and visit require that f be Applicative.
#

class Star[A, B](Strong, Choice):
    """Profunctor lifted from a function a -> f b for a Functor/Applicative f.

    newtype Star f a b = Star { runStar : a -> f b }

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
        fn = EffectfulFunction(self._fn, self._functor)
        return Star(g(fn), self._functor)

    def visit(self, g):
        """Converts a VL affine traversal into a Star profunctor.

        g : Applicative f => (forall r. r -> f r) -> (a -> f b) -> s -> f t

        The visit function g takes point (pure for the functor), the focus
        function, and the structure s — all three arguments at once.
        """
        return Star(lambda s: g(self._functor.pure, self._fn, s), self._functor)

    def iwander(self, f):
        """Applies an indexed VL traversal function, ignoring the index.

        f : (i -> a -> g b) -> s -> g t  for any Applicative g
        """
        g, eff = self._fn, self._functor

        def go(s):
            ig = EffectfulFunction(lambda _ix, a: g(a), eff)
            return f(ig)(s)
        return Star(go, eff)


#
# Costar is the dual profunctor, lowering f a -> b into a profunctor.
#
# newtype Costar f a b = Costar { runCostar : f a -> b }
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

    newtype Costar f a b = Costar { runCostar : f a -> b }

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
    # unfirst  : Costar f (a, c) (b, c) -> Costar f a b
    # unsecond : Costar f (c, a) (c, b) -> Costar f a b
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
    # closed : Costar f a b -> Costar f (x -> a) (x -> b)
    # closed (Costar f) = Costar $ \g x -> f (fmap ($ x) g)
    #
    # Here g : f (x -> a) and we produce x -> b by applying each (x -> a) to x
    # via fmap, yielding f a, then feeding that into f.
    # This uses only fmap and works when f distributes over functions.

    def closed(self):
        fn = self._fn

        def k(g):              # g : f (x -> a)
            return lambda x: fn(map(lambda h: h(x), g))  # fmap ($ x) g : F[a]
        return Costar(k, self._functor)
