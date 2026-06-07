#!/usr/bin/env python3.12
"""Tests for non-indexed optics: Lens, Getter, Fold, Traversal,
AffineFold, AffineTraversal, Prism, Iso, and composition.
"""

import pytest

from fp_concepts.either  import Left, Right
from fp_concepts.list    import List
from fp_concepts.maybe   import Nothing, Some
from fp_concepts.monoids import Sum, Count, Product
from fp_concepts.wrappers import EffectfulFunction

from fp_concepts.optics.lens import (
    Lens, lens, at, alongside, t_0, t_1, t_2,
)
from fp_concepts.optics.getter import getter, view, view_with
from fp_concepts.optics.setter import over, put, puts
from fp_concepts.optics.fold import (
    folded, folding,
    fold_map_of, fold_of, collect,
    sum_of, count_of, product_of, mean_of,
    left_fold_of, right_fold_of,
)
from fp_concepts.optics.traversal import (
    traversal, both, each, traverse_of,
)
from fp_concepts.optics.affine_fold import (
    afold, afolding, filtered, a_or, has, preview_of,
)
from fp_concepts.optics.affine_traversal import (
    affine_traversal_vl, ix, matching,
    preview, over, put,
)
from fp_concepts.optics.prism import (
    prism, left, right, some, nothing, only,
)
from fp_concepts.optics.review import review
from fp_concepts.optics.iso import (
    iso, from_, involuted, non,
    negated, swapped, flipped,
    to_list, to_dict, to_pair
)

#
# Helpers
#

effn = EffectfulFunction

# Simple lenses on 2-tuples
fst: Lens = lens(lambda s: s[0], lambda s, b: (b, s[1]))
snd: Lens = lens(lambda s: s[1], lambda s, b: (s[0], b))


#
# Lenses
#

class TestLens:
    def test_view(self):
        assert view(fst)((1, 2)) == 1
        assert view(snd)((1, 2)) == 2

    def test_over(self):
        assert over(fst, str)((1, 2)) == ('1', 2)

    def test_put(self):
        assert put(snd, 99)((1, 2)) == (1, 99)

    def test_at_single(self):
        assert view(at(1))([10, 20, 30]) == 20

    def test_at_put(self):
        assert put(at(0), 99)([1, 2, 3]) == [99, 2, 3]

    def test_at_negative(self):
        assert view(at(-1))([10, 20, 30]) == 30

    def test_at_slice(self):
        assert view(at(0, slice(2, None)))([0, 1, 2, 3]) == [0, 2, 3]

    def test_alongside(self):
        pair_lens = alongside(fst, snd)
        assert view(pair_lens)((('a', 1), ('b', 2))) == ('a', 2)
        assert put(pair_lens, ('x', 9))((('a', 1), ('b', 2))) == (('x', 1), ('b', 9))

    def test_tuple_lenses(self):
        triple = (10, 20, 30)
        assert view(t_0)(triple) == 10
        assert view(t_2)(triple) == 30
        assert put(t_1, 99)(triple) == (10, 99, 30)

    def test_composition(self):
        # fst @ snd: focus on s[0][1]
        assert view(fst @ snd)(((1, 2), 3)) == 2
        assert put(fst @ snd, 99)(((1, 2), 3)) == ((1, 99), 3)


#
# Getters
#

class TestGetter:
    def test_getter_view(self):
        g = getter(lambda s: s['x'])
        assert view(g)({'x': 42}) == 42

    def test_view_with(self):
        assert view_with(fst, str)((7, 8)) == '7'

    def test_getter_composition(self):
        g = getter(lambda s: s[0])
        assert view(fst @ g)(((10, 20), 5)) == 10


#
# Folds
#

class TestFold:
    def test_collect_list(self):
        assert list(collect(folded)(List([1, 2, 3]))) == [1, 2, 3]

    def test_fold_map_of(self):
        result = fold_map_of(folded, lambda x: x * 2, Sum)(List([1, 2, 3]))
        assert result == 12

    def test_sum_of(self):
        assert sum_of(folded)(List([1, 2, 3])) == 6

    def test_count_of(self):
        assert count_of(folded)(List([10, 20, 30])) == 3

    def test_product_of(self):
        assert product_of(folded)(List([2, 3, 4])) == 24

    def test_mean_of(self):
        assert mean_of(folded)(List([1, 2, 3])) == 2.0

    def test_right_fold_of(self):
        # right fold builds list in forward order when prepending
        result = right_fold_of(folded, lambda a, acc: [a] + acc, [])(List([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_left_fold_of(self):
        result = left_fold_of(folded, lambda acc, a: acc + [a], [])(List([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_folding(self):
        double_fold = folding(lambda xs: List([x * 2 for x in xs]))
        assert list(collect(double_fold)([1, 2, 3])) == [2, 4, 6]

    def test_fold_empty(self):
        assert list(collect(folded)(List())) == []
        assert count_of(folded)(List()) == 0

    def test_lens_as_fold(self):
        # A Lens can be used as a Fold
        assert list(collect(fst)((10, 20))) == [10]


#
# Traversals
#

class TestTraversal:
    def test_each_collect(self):
        assert list(collect(each)(List([1, 2, 3]))) == [1, 2, 3]

    def test_each_over(self):
        assert list(over(each, lambda x: x + 1)(List([1, 2, 3]))) == [2, 3, 4]

    def test_both_collect(self):
        assert list(collect(both)((1, 2))) == [1, 2]

    def test_both_over(self):
        assert over(both, lambda x: x * 10)((3, 4)) == (30, 40)

    def test_traverse_of_effectful_fn(self):
        result = traverse_of(each, effn(lambda x: Some(x * 2), Some))(List([1, 2, 3]))
        assert result == Some(List([2, 4, 6]))

    def test_traverse_of_bare_fn_with_effect(self):
        result = traverse_of(each, lambda x: Some(x * 2), Some)(List([1, 2, 3]))
        assert result == Some(List([2, 4, 6]))

    def test_traverse_of_both_forms_agree(self):
        f = lambda x: Some(x + 1) if x > 0 else Nothing()
        xs = List([1, 2, 3])
        assert traverse_of(each, effn(f, Some))(xs) == traverse_of(each, f, Some)(xs)

    def test_traverse_of_short_circuit(self):
        result = traverse_of(each, lambda x: Some(x) if x > 0 else Nothing(), Some)(List([1, -1, 3]))
        assert result == Nothing()

    def test_traverse_of_no_effect_raises(self):
        with pytest.raises(ValueError):
            traverse_of(each, lambda x: Some(x))

    def test_traversal_builder(self):
        def both_vl(f):
            def run(pair):
                a, b = pair
                return f(a).map2(lambda x, y: (x, y), f(b))
            return run
        both_manual = traversal(both_vl)
        assert over(both_manual, lambda x: x + 1)((3, 4)) == (4, 5)

    def test_composition_lens_traversal(self):
        # fst @ each: traverse elements of the list in the first slot
        nested = (List([1, 2, 3]), 'x')
        assert list(collect(fst @ each)(nested)) == [1, 2, 3]
        assert list(over(fst @ each, lambda x: x * 2)(nested)[0]) == [2, 4, 6]


#
# AffineFolds
#

class TestAffineFold:
    def test_afolding_some(self):
        af = afolding(lambda s: Some(s[0]) if s else Nothing())
        assert preview(af)([10, 20]) == Some(10)
        assert preview(af)([]) == Nothing()

    def test_filtered_passes(self):
        assert preview(filtered(lambda x: x > 0))(5) == Some(5)

    def test_filtered_blocks(self):
        assert preview(filtered(lambda x: x > 0))(-1) == Nothing()

    def test_has_true(self):
        assert has(filtered(lambda x: x > 0), 5) is True

    def test_has_false(self):
        assert has(filtered(lambda x: x > 0), -1) is False

    def test_a_or_first_wins(self):
        pos = filtered(lambda x: x > 0)
        neg = filtered(lambda x: x < 0)
        assert preview(a_or(pos, neg))(3) == Some(3)

    def test_a_or_fallback(self):
        pos = filtered(lambda x: x > 0)
        neg = filtered(lambda x: x < 0)
        assert preview(a_or(pos, neg))(-2) == Some(-2)

    def test_a_or_both_fail(self):
        pos = filtered(lambda x: x > 0)
        neg = filtered(lambda x: x < 0)
        assert preview(a_or(pos, neg))(0) == Nothing()

    def test_preview_of_hit(self):
        af = afolding(lambda s: Some(s) if s else Nothing())
        assert preview_of(af, lambda x: x * 2)(5) == Some(10)

    def test_preview_of_miss(self):
        af = afolding(lambda s: Some(s) if s else Nothing())
        assert preview_of(af, lambda x: x * 2)(0) == Nothing()


#
# AffineTraversals
#

class TestAffineTraversal:
    def test_ix_in_bounds(self):
        assert preview(ix(1))([10, 20, 30]) == Some(20)

    def test_ix_out_of_bounds(self):
        assert preview(ix(5))([10, 20]) == Nothing()

    def test_ix_put(self):
        assert put(ix(1), 99)([10, 20, 30]) == [10, 99, 30]

    def test_ix_out_of_bounds_put(self):
        # No focus: structure unchanged
        assert put(ix(5), 99)([10, 20]) == [10, 20]

    def test_affine_traversal_vl(self):
        head = affine_traversal_vl(
            lambda s: Right(s[0]) if s else Left(s),
            lambda s, b: [b] + s[1:]
        )
        assert preview(head)([1, 2, 3]) == Some(1)
        assert preview(head)([]) == Nothing()
        assert over(head, lambda x: x * 10)([1, 2, 3]) == [10, 2, 3]

    def test_matching_hit(self):
        assert matching(ix(0), [42]) == Right(42)

    def test_matching_miss(self):
        assert matching(ix(5), [1, 2]) == Left([1, 2])

    def test_composition(self):
        # ix(0) @ ix(1): focus on element 1 of element 0
        nested = [[10, 20, 30], [40, 50]]
        assert preview(ix(0) @ ix(1))(nested) == Some(20)
        assert put(ix(0) @ ix(1), 99)(nested) == [[10, 99, 30], [40, 50]]


#
# Prisms
#

class TestPrism:
    def test_left_preview_hit(self):
        assert preview(left)(Left(42)) == Some(42)

    def test_left_preview_miss(self):
        assert preview(left)(Right('x')) == Nothing()

    def test_right_preview_hit(self):
        assert preview(right)(Right('hello')) == Some('hello')

    def test_right_preview_miss(self):
        assert preview(right)(Left(1)) == Nothing()

    def test_left_review(self):
        assert review(left)(42) == Left(42)

    def test_right_review(self):
        assert review(right)('hello') == Right('hello')

    def test_some_preview_hit(self):
        assert preview(some)(Some(10)) == Some(10)

    def test_some_preview_miss(self):
        assert preview(some)(Nothing()) == Nothing()

    def test_some_review(self):
        assert review(some)(10) == Some(10)

    def test_nothing_preview_hit(self):
        assert preview(nothing)(Nothing()) == Some(None)

    def test_nothing_preview_miss(self):
        assert preview(nothing)(Some(1)) == Nothing()

    def test_only_match(self):
        assert preview(only(42))(42) == Some(())
        assert preview(only(42))(99) == Nothing()

    def test_only_review(self):
        assert review(only(42))(()) == 42

    def test_prism_over_hit(self):
        assert over(some, lambda x: x * 2)(Some(5)) == Some(10)

    def test_prism_over_miss(self):
        assert over(some, lambda x: x * 2)(Nothing()) == Nothing()

    def test_custom_prism(self):
        pos = prism(abs, lambda x: Right(x) if x > 0 else Left(x))
        assert preview(pos)(5) == Some(5)
        assert preview(pos)(-3) == Nothing()
        assert review(pos)(7) == 7


#
# Isos
#

class TestIso:
    def test_iso_view(self):
        double = iso(lambda x: x * 2, lambda x: x // 2)
        assert view(double)(5) == 10

    def test_iso_review(self):
        double = iso(lambda x: x * 2, lambda x: x // 2)
        assert review(double)(10) == 5

    def test_from_view(self):
        double = iso(lambda x: x * 2, lambda x: x // 2)
        assert view(from_(double))(10) == 5

    def test_from_review(self):
        double = iso(lambda x: x * 2, lambda x: x // 2)
        assert review(from_(double))(5) == 10

    def test_negated_view(self):
        assert view(negated)(3) == -3
        assert view(negated)(-3) == 3

    def test_negated_review(self):
        assert review(negated)(3) == -3

    def test_swapped_view(self):
        assert view(swapped)((1, 2)) == (2, 1)

    def test_swapped_review(self):
        assert review(swapped)((1, 2)) == (2, 1)

    def test_flipped_view(self):
        assert view(flipped)(Left(1)) == Right(1)
        assert view(flipped)(Right(2)) == Left(2)

    def test_involuted(self):
        rev_str = involuted(lambda s: s[::-1])
        assert view(rev_str)('abc') == 'cba'
        assert review(rev_str)('abc') == 'cba'

    def test_non_some(self):
        assert view(non(0))(Some(5)) == 5

    def test_non_nothing(self):
        assert view(non(0))(Nothing()) == 0

    def test_non_review_present(self):
        assert review(non(0))(5) == Some(5)

    def test_non_review_default(self):
        assert review(non(0))(0) == Nothing()

    def test_iso_as_lens(self):
        # over iso f s = bt(f(sa(s))): sa(3)=-3, f(-3)=-2, bt(-2)=2
        assert over(negated, lambda x: x + 1)(3) == 2

    def test_iso_composition(self):
        # negated @ negated = identity
        assert view(negated @ negated)(7) == 7


#
# Composition across optic types
#

class TestComposition:
    def test_lens_fold(self):
        nested = (List.of(1, 2, 3), 'x')
        assert list(collect(fst @ folded)(nested)) == [1, 2, 3]

    def test_prism_traversal(self):
        # some @ each: traverse inside a Some
        assert list(collect(some @ each)(Some(List([1, 2, 3])))) == [1, 2, 3]
        assert list(collect(some @ each)(Nothing())) == []

    def test_prism_lens(self):
        # right @ fst: focus on first element of a Right pair
        assert preview(right @ fst)(Right((10, 20))) == Some(10)

    def test_affine_traversal_fold(self):
        # ix(0) @ folded: fold over elements of the first item
        nested = [List.of(1, 2, 3), List.of(4, 5)]
        assert list(collect(ix(0) @ folded)(nested)) == [1, 2, 3]

    def test_iso_lens(self):
        # swapped @ fst: view first of swapped pair = second of original
        assert view(swapped @ fst)((1, 2)) == 2

    def test_nested(self):
        assert (List.of(1, 2, 3, 4, 5) >> over(each, lambda x: x + 1)) == [2, 3, 4, 5, 6]

        _x = List.of((1, 2), (2, 3), (4, 5), (6, 7))
        assert (_x >> over(each @ t_0, lambda x: x + 1)) == [(2, 2), (3, 3), (5, 5), (7, 7)]

        _y = List.of((1, 2, 3), (2, 3, 5), (4, 5, 6), (6, 7, 8))
        assert (_y >> over(each @ t_0, lambda x: x + 1)) == [(2, 2, 3), (3, 3, 5), (5, 5, 6), (7, 7, 8)]

        _z = List.of((List.of(1, 10, 100), 2, 3), (List.of(2, 20, 200), 3, 5),
                     (List.of(4, 40, 400), 5, 6), (List.of(6, 9, 11), 7, 8))
        assert (_z >> over(each @ t_0 @ each, lambda x: x + 1)) == [([2, 11, 101], 2, 3), ([3, 21, 201], 3, 5), ([5, 41, 401], 5, 6), ([7, 10, 12], 7, 8)]

        _w = List.of(([1, 10, 100], 2, 3), ([2, 20, 200], 3, 5), ([4, 40, 400], 5, 6), ([6, 9, 11], 7, 8))
        assert (_w >> over(each @ t_0 @ to_list @ each, lambda x: x + 1)) == [([2, 11, 101], 2, 3), ([3, 21, 201], 3, 5), ([5, 41, 401], 5, 6), ([7, 10, 12], 7, 8)]

#
# Downgrades (i.e., optics used as weaker types)
#

class TestDowngrade:
    def test_lens_as_fold(self):
        assert list(collect(fst)((10, 20))) == [10]
        assert sum_of(fst)((3, 99)) == 3

    def test_traversal_as_fold(self):
        assert list(collect(each)(List([1, 2, 3]))) == [1, 2, 3]

    def test_affine_traversal_as_fold_hit(self):
        assert list(collect(ix(0))([42, 99])) == [42]

    def test_affine_traversal_as_fold_miss(self):
        assert list(collect(ix(5))([42, 99])) == []

    def test_prism_as_fold(self):
        assert list(collect(some)(Some(7))) == [7]
        assert list(collect(some)(Nothing())) == []

    def test_iso_as_fold(self):
        assert list(collect(negated)(3)) == [-3]
