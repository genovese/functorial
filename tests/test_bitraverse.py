#!/usr/bin/env python3.12
"""Tests for Bitraversable_ and its use by Pair and Either.

Four basic checks covered:

  1. MRO is clean.

     Pair, Left, and Right's own bitraverse/traverse implementations
     are used over the inherited Bitraversable_ defaults.

  2. bitraverse/bisequence behavior on Pair (via the Maybe applicative,
     including short-circuiting) and on Left/Right.

  3. The derived `traverse` default (leaving the first component untouched
     via f.pure), exercised directly on a minimal synthetic Bitraversable_
     instance, since Pair/Either both keep their own hand-written traverse
     and so never touch that default themselves.

  4. bisequence on an already-effectful structure.

"""

from functorial.applicative   import IdentityA, map2
from functorial.bitraversable import Bitraversable_, bisequence, bitraverse
from functorial.either        import Left, Right
from functorial.maybe         import Nothing, Some
from functorial.pair          import Pair


class MinimalBitraversable(Bitraversable_):
    """A minimal Bitraversable_ instance implementing only bitraverse.

    Used to exercise the *derived* defaults (traverse, bisequence) directly,
    since Pair/Either both override traverse for efficiency and so never
    exercise the Bitraversable_ default themselves.

    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return isinstance(other, MinimalBitraversable) and self.a == other.a and self.b == other.b

    def __repr__(self):
        return f'MinimalBitraversable({self.a!r}, {self.b!r})'

    def bimap(self, f, g):
        return MinimalBitraversable(f(self.a), g(self.b))

    def bitraverse(self, _f, g1, g2):
        return map2(MinimalBitraversable, g1(self.a), g2(self.b))


class TestOwnMethodsWinOverDefaults:
    def test_pair_methods_are_own(self):
        assert Pair.traverse is not Bitraversable_.traverse
        assert Pair.bitraverse is not Bitraversable_.bitraverse

    def test_left_methods_are_own(self):
        assert Left.traverse is not Bitraversable_.traverse
        assert Left.bitraverse is not Bitraversable_.bitraverse

    def test_right_methods_are_own(self):
        assert Right.traverse is not Bitraversable_.traverse
        assert Right.bitraverse is not Bitraversable_.bitraverse

    def test_minimal_instance_uses_derived_traverse(self):
        assert MinimalBitraversable.traverse is Bitraversable_.traverse
        assert MinimalBitraversable.bisequence is Bitraversable_.bisequence


class TestBitraverseOnPair:
    def test_bitraverse_success(self):
        p = Pair(1, 2)
        result = bitraverse(lambda a: Some(a + 10), lambda b: Some(b + 20), p)
        assert result == Some(Pair(11, 22))

    def test_bitraverse_short_circuits_on_first(self):
        p = Pair(1, 2)
        result = bitraverse(lambda _a: Nothing(), lambda b: Some(b + 20), p)
        assert result == Nothing()

    def test_bitraverse_short_circuits_on_second(self):
        p = Pair(1, 2)
        result = bitraverse(lambda a: Some(a + 10), lambda _b: Nothing(), p)
        assert result == Nothing()

    def test_bisequence_method(self):
        wrapped = Pair(IdentityA(1), IdentityA(2))
        result = wrapped.bisequence()
        assert IdentityA.run(result) == Pair(1, 2)

    def test_bisequence_free_function(self):
        wrapped = Pair(IdentityA(1), IdentityA(2))
        result = bisequence(wrapped)
        assert IdentityA.run(result) == Pair(1, 2)


class TestBitraverseOnEither:
    def test_left(self):
        result = bitraverse(lambda a: Some(a + 1), lambda b: Some(b + 1), Left(5))
        assert result == Some(Left(6))

    def test_right(self):
        result = bitraverse(lambda a: Some(a + 1), lambda b: Some(b + 1), Right(7))
        assert result == Some(Right(8))

    def test_left_ignores_second_function(self):
        # Should not call the second (right) function at all for a Left.
        def boom(_b):
            raise AssertionError('should not be called')

        result = bitraverse(lambda a: Some(a + 1), boom, Left(5))
        assert result == Some(Left(6))


class TestDerivedTraverseDefault:
    def test_traverse_leaves_first_component_untouched(self):
        bt = MinimalBitraversable(1, 2)
        result = bt.traverse(IdentityA, lambda b: IdentityA(b + 1))
        assert IdentityA.run(result) == MinimalBitraversable(1, 3)

    def test_bisequence_default(self):
        bt = MinimalBitraversable(IdentityA(1), IdentityA(2))
        result = bt.bisequence()
        assert IdentityA.run(result) == MinimalBitraversable(1, 2)
