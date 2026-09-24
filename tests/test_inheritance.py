#!/usr/bin/env python3.12
"""Tests for clean inheritance using List and Dict with IndexedTraversable_ and IndexedFoldable_.

Checks three basic issues:

  1. MRO is correct.

     In particular, List and Dict's own implementations (e.g.,
     fold_map, fold, ifold_map, ifold, traverse, itraverse) should
     win over the inherited defaults from Traversable_/IndexedTraversable_ and
     Foldable_/IndexedFoldable_.

  2. Foldable_/IndexedFoldable_ defaults now available on List and Dict.

     Specifically: to_list, to_indexed_list, find, any, all, concat_map,
     fold_right, map_maybe.

  3. Traversable_/IndexedTraversable_ defaults now available on List and Dict

     Specifically: sequence, isequence.

Some utility tests have also leaked in here because they exercise
List, Dict, and Foldable. These will eventually be moved.

"""

from functorial.applicative import IdentityA
from functorial.dict        import Dict
from functorial.foldable    import Foldable_, IndexedFoldable_
from functorial.list        import List, cat_maybes, map_maybe
from functorial.maybe       import Nothing, Some
from functorial.traversable import IndexedTraversable_


class TestOwnMethodsWinOverDefaults:
    def test_list_foldable_methods_are_own(self):
        assert List.fold_map is not Foldable_.fold_map
        assert List.fold is not Foldable_.fold
        assert List.fold_right is not Foldable_.fold_right
        assert List.ifold_map is not IndexedFoldable_.ifold_map
        assert List.ifold is not IndexedFoldable_.ifold

    def test_list_traversable_methods_are_own(self):
        assert List.traverse is not IndexedTraversable_.traverse
        assert List.itraverse is not IndexedTraversable_.itraverse

    def test_dict_foldable_methods_are_own(self):
        assert Dict.fold_map is not Foldable_.fold_map
        assert Dict.fold is not Foldable_.fold
        assert Dict.ifold_map is not IndexedFoldable_.ifold_map
        assert Dict.ifold is not IndexedFoldable_.ifold

    def test_dict_fold_right_falls_through_to_default(self):
        # Dict has no direct fold_right, so it should inherit the version
        # derived from fold_map via the Endo monoid.
        assert Dict.fold_right is Foldable_.fold_right

    def test_dict_traversable_methods_are_own(self):
        assert Dict.traverse is not IndexedTraversable_.traverse
        assert Dict.itraverse is not IndexedTraversable_.itraverse


class TestListFoldableDefaults:
    def test_to_list(self):
        assert List.of(1, 2, 3).to_list() == [1, 2, 3]

    def test_to_indexed_list(self):
        assert List.of(1, 2, 3).to_indexed_list() == [(0, 1), (1, 2), (2, 3)]

    def test_find(self):
        assert List.of(1, 2, 3).find(lambda x: x > 1) == Some(2)

    def test_any(self):
        assert List.of(1, 2, 3).any(lambda x: x > 2)
        assert not List.of(1, 2, 3).any(lambda x: x > 10)

    def test_all(self):
        assert List.of(1, 2, 3).all(lambda x: x > 0)
        assert not List.of(1, 2, 3).all(lambda x: x > 1)

    def test_concat_map(self):
        assert List.of(1, 2, 3).concat_map(lambda x: [x, x]) == [1, 1, 2, 2, 3, 3]

    def test_map_maybe(self):
        evens = List.of(1, 2, 3, 4, 5).map_maybe(lambda x: Some(x * 10) if x % 2 == 0 else Nothing())
        assert evens == [20, 40]
        assert type(evens) is list  # generic default, not List-typed


class TestDictFoldableDefaults:
    def test_to_list(self):
        assert sorted(Dict.of(('a', 1), ('b', 2)).to_list()) == [1, 2]

    def test_to_indexed_list(self):
        assert sorted(Dict.of(('a', 1), ('b', 2)).to_indexed_list()) == [('a', 1), ('b', 2)]

    def test_any(self):
        assert Dict.of(('a', 1), ('b', 2)).any(lambda v: v > 1)
        assert not Dict.of(('a', 1), ('b', 2)).any(lambda v: v > 10)

    def test_all(self):
        assert Dict.of(('a', 1), ('b', 2)).all(lambda v: v > 0)
        assert not Dict.of(('a', 1), ('b', 2)).all(lambda v: v > 1)

    def test_fold_right_default(self):
        assert Dict.of(('a', 1), ('b', 2)).fold_right(lambda v, acc: acc + v, 0) == 3

    def test_map_maybe(self):
        d = Dict.of(('a', 1), ('b', 2), ('c', 3))
        result = d.map_maybe(lambda v: Some(v * 10) if v % 2 == 0 else Nothing())
        assert sorted(result) == [20]
        assert type(result) is list  # generic default, not Dict-typed


class TestListMapMaybeUtilities:
    """Tests for list.py's own map_maybe/cat_maybes, which are List-typed."""

    def test_map_maybe_keeps_only_present(self):
        result = map_maybe(lambda x: Some(x * 10) if x % 2 == 0 else Nothing(), List.of(1, 2, 3, 4, 5))
        assert result == List.of(20, 40)
        assert type(result) is List

    def test_map_maybe_all_nothing(self):
        result = map_maybe(lambda _x: Nothing(), List.of(1, 2, 3))
        assert result == List()

    def test_map_maybe_empty_input(self):
        assert map_maybe(Some, List()) == List()

    def test_cat_maybes(self):
        result = cat_maybes(List.of(Some(1), Nothing(), Some(3), Nothing()))
        assert result == List.of(1, 3)
        assert type(result) is List

    def test_cat_maybes_all_present(self):
        assert cat_maybes(List.of(Some(1), Some(2))) == List.of(1, 2)

    def test_cat_maybes_all_absent(self):
        assert cat_maybes(List.of(Nothing(), Nothing())) == List()


class TestTraversableDefaults:
    def test_list_sequence(self):
        wrapped = List.of(IdentityA(1), IdentityA(2), IdentityA(3))
        result = wrapped.sequence()
        assert IdentityA.run(result).to_list() == [1, 2, 3]

    def test_list_isequence(self):
        wrapped = List.of(IdentityA(1), IdentityA(2), IdentityA(3))
        result = wrapped.isequence()
        assert IdentityA.run(result).to_list() == [1, 2, 3]

    def test_dict_sequence(self):
        wrapped = Dict.of(('a', IdentityA(1)), ('b', IdentityA(2)))
        result = wrapped.sequence()
        assert dict(IdentityA.run(result)) == {'a': 1, 'b': 2}
