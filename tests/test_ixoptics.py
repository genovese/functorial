#!/usr/bin/env python3.12
"""Tests for indexed optics: IxLens, IxGetter, IxFold, IxTraversal.

Covers five themes:
  1. IxLens basics: ilens, iview, iover, iput, selfIndex, index pairing
  2. IxFold on List and Dict: ifolded, ifolding, ifold_map_of, icollect,
     iright_fold_of, ileft_fold_of
  3. IxTraversal: ieach on List, Dict, RoseTree; itraverse_of with effects
  4. Composition: selfIndex @ ifolded, ilens @ ieach, nested structures
  5. Downgrade: indexed optics used as plain via cast_as
"""

import pytest

from fp_concepts.dict        import Dict
from fp_concepts.list        import List
from fp_concepts.maybe       import Some, Nothing
from fp_concepts.monoids     import Sum
from fp_concepts.trees       import RoseTree

from fp_concepts.optics.optic     import OpticIs
from fp_concepts.optics.fold      import collect, fold_map_of
from fp_concepts.optics.getter    import view
from fp_concepts.optics.setter    import over
from fp_concepts.optics.traversal import traverse_of

from fp_concepts.optics.ix_lens import (
    ilens, igetter, iview, iover, iput, selfIndex,
)
from fp_concepts.optics.ix_fold import (
    ifolded, ifolding,
    ifold_map_of, icollect, iright_fold_of, ileft_fold_of,
)
from fp_concepts.optics.ix_traversal import (
    itraversal, ieach, itraverse_of,
)


# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------

# A pair lens that records 'fst' / 'snd' as indices.
fst_ix = ilens(lambda s: ('fst', s[0]), lambda s, b: (b, s[1]))
snd_ix = ilens(lambda s: ('snd', s[1]), lambda s, b: (s[0], b))


# ---------------------------------------------------------------------------
# Theme 1 — IxLens basics
# ---------------------------------------------------------------------------

class TestIxLens:
    def test_iview_fst(self):
        assert iview(fst_ix)(('hello', 42)) == ('fst', 'hello')

    def test_iview_snd(self):
        assert iview(snd_ix)(('hello', 42)) == ('snd', 42)

    def test_iover_fst(self):
        result = iover(fst_ix, lambda _i, a: a.upper())(('hello', 42))
        assert result == ('HELLO', 42)

    def test_iover_index_used(self):
        # index is 'snd'; modify inserts index into value
        result = iover(snd_ix, lambda i, a: f'{i}:{a}')(('x', 7))
        assert result == ('x', 'snd:7')

    def test_iput(self):
        assert iput(fst_ix, 'world')(('hello', 42)) == ('world', 42)

    def test_selfindex_iview(self):
        assert iview(selfIndex)(42) == (42, 42)
        assert iview(selfIndex)('abc') == ('abc', 'abc')

    def test_selfindex_iover(self):
        # index is the whole structure; add index to focus
        assert iover(selfIndex, lambda i, a: i + a)(10) == 20

    def test_ilens_composition_index_pair(self):
        # fst_ix @ snd_ix: focus on s[0][1], index ('fst', 'snd')
        composed = fst_ix @ snd_ix
        data = (('hello', 42), 'world')
        assert iview(composed)(data) == (('fst', 'snd'), 42)

    def test_ilens_composition_iover(self):
        composed = fst_ix @ snd_ix
        data = (('hello', 42), 'world')
        result = iover(composed, lambda _i, a: a + 1)(data)
        assert result == (('hello', 43), 'world')

    def test_three_lens_composition_index(self):
        # three composed lenses yield ((i, j), k)
        l1 = ilens(lambda s: (1, s[0]), lambda s, b: (b,) + s[1:])
        l2 = ilens(lambda s: (2, s[0]), lambda s, b: (b,) + s[1:])
        l3 = ilens(lambda s: (3, s[0]), lambda s, b: (b,) + s[1:])
        composed = l1 @ l2 @ l3
        data = (((42, 'x'), 'y'), 'z')
        ix, val = iview(composed)(data)
        assert ix == ((1, 2), 3)
        assert val == 42

    def test_igetter(self):
        g = igetter(lambda s: ('key', s['key']))
        assert iview(g)({'key': 99}) == ('key', 99)


# ---------------------------------------------------------------------------
# Theme 2 — IxFold on List and Dict
# ---------------------------------------------------------------------------

class TestIxFold:
    def test_icollect_list(self):
        xs = List([10, 20, 30])
        result = icollect(ifolded)(xs)
        assert list(result) == [(0, 10), (1, 20), (2, 30)]

    def test_icollect_dict(self):
        d = Dict({'a': 1, 'b': 2, 'c': 3})
        result = set(icollect(ifolded)(d))
        assert result == {('a', 1), ('b', 2), ('c', 3)}

    def test_icollect_empty(self):
        assert list(icollect(ifolded)(List())) == []

    def test_ifold_map_of_weighted_sum(self):
        # sum of i * a: 0*10 + 1*20 + 2*30 = 80
        xs = List([10, 20, 30])
        result = ifold_map_of(ifolded, lambda i, a: i * a, Sum)(xs)
        assert result == 80

    def test_iright_fold_of(self):
        xs = List([10, 20, 30])
        result = iright_fold_of(ifolded, lambda i, a, acc: [(i, a)] + acc, [])(xs)
        assert result == [(0, 10), (1, 20), (2, 30)]

    def test_ileft_fold_of(self):
        xs = List([10, 20, 30])
        result = ileft_fold_of(ifolded, lambda acc, i, a: acc + [(i, a)], [])(xs)
        assert result == [(0, 10), (1, 20), (2, 30)]

    def test_ileft_vs_iright_order(self):
        # left fold accumulates in forward order; right fold can build forward too
        xs = List([1, 2, 3])
        left  = ileft_fold_of(ifolded,  lambda acc, _i, a: acc + [a], [])(xs)
        right = iright_fold_of(ifolded, lambda _i, a, acc: [a] + acc, [])(xs)
        assert left == right == [1, 2, 3]

    def test_ifolding_from_dict(self):
        # ifolding wraps a function s -> IndexedFoldable into an IxFold
        # Use it to treat a plain list as a dict-indexed fold
        fold_op = ifolding(lambda xs: Dict({i: v for i, v in enumerate(xs)}))
        result = set(icollect(fold_op)([100, 200, 300]))
        assert result == {(0, 100), (1, 200), (2, 300)}

    def test_ifolded_dict_ifold_map(self):
        # Collect keys whose value exceeds a threshold
        d = Dict({'a': 5, 'b': 12, 'c': 3, 'd': 8})
        result = ifold_map_of(
            ifolded,
            lambda k, v: List.of(k) if v > 6 else List(),
        )(d)
        assert set(result) == {'b', 'd'}


# ---------------------------------------------------------------------------
# Theme 3 — IxTraversal: ieach and itraverse_of
# ---------------------------------------------------------------------------

class TestIxTraversal:
    def test_ieach_icollect_list(self):
        xs = List([10, 20, 30])
        result = icollect(ieach)(xs)
        assert list(result) == [(0, 10), (1, 20), (2, 30)]

    def test_ieach_icollect_dict(self):
        d = Dict({'x': 1, 'y': 2})
        result = set(icollect(ieach)(d))
        assert result == {('x', 1), ('y', 2)}

    def test_ieach_icollect_rosetree(self):
        # RoseTree([1, [2], [3, [4]]])
        # indices: root=List(), 1st child=List([0]),
        #          2nd child=List([1]), 1st child of 2nd=List([1,0])
        t = RoseTree([1, [2], [3, [4]]])
        pairs = list(icollect(ieach)(t))

        def val_at(index):
            "Linear search — List is not hashable (mutable sequence)."
            return next(v for ix, v in pairs if ix == index)

        assert val_at(List()) == 1
        assert val_at(List([0])) == 2
        assert val_at(List([1])) == 3
        assert val_at(List([1, 0])) == 4

    def test_ieach_iover_list(self):
        xs = List([10, 20, 30])
        result = iover(ieach, lambda i, a: a + i)(xs)
        assert list(result) == [10, 21, 32]

    def test_ieach_iover_dict(self):
        d = Dict({'a': 1, 'b': 2})
        result = iover(ieach, lambda k, v: f'{k}{v}')(d)
        assert result == Dict({'a': 'a1', 'b': 'b2'})

    def test_ieach_iput_list(self):
        xs = List([1, 2, 3])
        result = iput(ieach, 0)(xs)
        assert list(result) == [0, 0, 0]

    def test_itraverse_of_list_success(self):
        # Guard: allow only positive values, double them
        xs = List([1, 2, 3])
        result = itraverse_of(ieach, Some, lambda _i, a: Some(a * 2))(xs)
        assert result == Some(List([2, 4, 6]))

    def test_itraverse_of_list_failure(self):
        # Negative value causes short-circuit to Nothing
        xs = List([1, -1, 3])
        result = itraverse_of(ieach, Some, lambda _i, a: Some(a) if a > 0 else Nothing())(xs)
        assert result == Nothing()

    def test_itraverse_of_uses_index(self):
        # Use the index in the effect: skip element at index 1
        xs = List([10, 20, 30])
        result = itraverse_of(
            ieach, Some,
            lambda i, a: Nothing() if i == 1 else Some(a)
        )(xs)
        assert result == Nothing()

    def test_itraversal_builder(self):
        # Build an IxTraversal manually from a van Laarhoven function
        # that traverses a pair's both elements with index 'left'/'right'
        def both_vl(g):
            def run(s):
                fa = g('left',  s[0])
                fb = g('right', s[1])
                return fa.map2(lambda a, b: (a, b), fb)
            return run

        both_ix = itraversal(both_vl)
        result = icollect(both_ix)((10, 20))
        assert set(result) == {('left', 10), ('right', 20)}

    def test_itraversal_iover(self):
        def both_vl(g):
            def run(s):
                fa = g('left',  s[0])
                fb = g('right', s[1])
                return fa.map2(lambda a, b: (a, b), fb)
            return run

        both_ix = itraversal(both_vl)
        result = iover(both_ix, lambda side, v: v * 10)((3, 7))
        assert result == (30, 70)


# ---------------------------------------------------------------------------
# Theme 4 — Composition
# ---------------------------------------------------------------------------

class TestComposition:
    def test_selfindex_at_ifolded(self):
        # selfIndex @ ifolded: index = (source_list, element_index)
        xs = List([10, 20])
        composed = selfIndex @ ifolded
        pairs = list(icollect(composed)(xs))
        # Each pair: ((xs, int_index), value)
        # List is not hashable, so compare as sorted list of (int_part, value)
        assert len(pairs) == 2
        # The outer index is xs itself (the source); the inner index is the int position.
        assert all(ix[0] == xs for ix, _ in pairs)
        assert sorted((ix[1], v) for ix, v in pairs) == [(0, 10), (1, 20)]

    def test_ilens_at_ieach(self):
        # fst_ix @ ieach: focus each element of s[0] with index ('fst', int)
        composed = fst_ix @ ieach
        data = (List([1, 2, 3]), 'extra')
        pairs = set(icollect(composed)(data))
        assert pairs == {(('fst', 0), 1), (('fst', 1), 2), (('fst', 2), 3)}

    def test_ilens_at_ieach_iover(self):
        composed = fst_ix @ ieach
        data = (List([1, 2, 3]), 'extra')
        result = iover(composed, lambda _ix, a: a * 10)(data)
        assert result == (List([10, 20, 30]), 'extra')

    def test_nested_list_ieach_composition(self):
        # ieach @ ieach: IxTraversal (int, int) [[a]] a
        composed = ieach @ ieach
        xss = List([List([1, 2]), List([3, 4])])
        pairs = set(icollect(composed)(xss))
        assert pairs == {((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)}

    def test_dict_of_lists_composition(self):
        # ifolded @ ieach: IxFold (str, int) (Dict str (List a)) a
        composed = ifolded @ ieach
        d = Dict({'a': List([10, 20]), 'b': List([30])})
        pairs = set(icollect(composed)(d))
        assert pairs == {(('a', 0), 10), (('a', 1), 20), (('b', 0), 30)}

    def test_rosetree_composition_with_snd(self):
        # snd_ix @ ieach: focus each node of s[1] with index ('snd', path)
        composed = snd_ix @ ieach
        t = RoseTree([1, [2], [3]])
        data = ('ignored', t)
        pairs = list(icollect(composed)(data))

        def val_at(index):
            return next(v for ix, v in pairs if ix == index)

        assert val_at(('snd', List())) == 1
        assert val_at(('snd', List([0]))) == 2
        assert val_at(('snd', List([1]))) == 3


# ---------------------------------------------------------------------------
# Theme 5 — Downgrade: indexed optics as plain via cast_as
# ---------------------------------------------------------------------------

class TestDowngrade:
    def test_ixlens_cast_as_lens_view(self):
        plain = fst_ix.cast_as(OpticIs.LENS)
        assert view(plain)(('hello', 42)) == 'hello'

    def test_ixlens_cast_as_lens_over(self):
        plain = fst_ix.cast_as(OpticIs.LENS)
        result = over(plain, str.upper)(('hello', 42))
        assert result == ('HELLO', 42)

    def test_ifolded_cast_as_fold_collect(self):
        xs = List([10, 20, 30])
        plain = ifolded.cast_as(OpticIs.FOLD)
        result = collect(plain)(xs)
        assert list(result) == [10, 20, 30]

    def test_ifolded_dict_cast_as_fold(self):
        d = Dict({'a': 1, 'b': 2})
        plain = ifolded.cast_as(OpticIs.FOLD)
        result = set(collect(plain)(d))
        assert result == {1, 2}

    def test_ieach_cast_as_traversal_over(self):
        xs = List([1, 2, 3])
        plain = ieach.cast_as(OpticIs.TRAVERSAL)
        result = over(plain, lambda x: x * 10)(xs)
        assert list(result) == [10, 20, 30]

    def test_ieach_cast_as_fold_collect(self):
        xs = List([7, 8, 9])
        plain = ieach.cast_as(OpticIs.FOLD)
        result = collect(plain)(xs)
        assert list(result) == [7, 8, 9]

    def test_ixlens_cast_as_fold(self):
        plain = fst_ix.cast_as(OpticIs.FOLD)
        result = collect(plain)(('hello', 42))
        assert list(result) == ['hello']

    def test_composition_downgrade(self):
        # (fst_ix @ ieach) composed result cast to plain fold
        composed = fst_ix @ ieach
        plain = composed.cast_as(OpticIs.FOLD)
        data = (List([10, 20, 30]), 'x')
        result = collect(plain)(data)
        assert list(result) == [10, 20, 30]
