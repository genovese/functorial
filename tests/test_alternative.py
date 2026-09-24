#!/usr/bin/env python3.12
"""Tests for Alternative's optional default, exercised on Maybe and List.

optional : Alternative f => f a -> f (Maybe a)

Wraps a possibly-failing computation so that it always succeeds. What that
means concretely depends on the instance's own map/pure/alt. For Maybe it
means "Some(a) on success, Nothing() on failure, both wrapped in Some";
for List (whose Applicative/Alternative instance models nondeterministic
choice rather than at-most-one) it appends a Nothing() choice onto the
list of Some(x) choices.

"""

from functorial.alternative import Alternative, guard, optional
from functorial.list        import List
from functorial.maybe       import Maybe, Nothing, Some


class TestListIsAlternative:
    def test_list_declares_alternative(self):
        assert issubclass(List, Alternative)

    def test_own_methods_win_over_defaults(self):
        assert List.empty is not Alternative.empty
        assert List.alt is not Alternative.alt


class TestOptionalOnList:
    def test_nonempty_list(self):
        result = List.of(1, 2, 3).optional()
        assert result == List.of(Some(1), Some(2), Some(3), Nothing())

    def test_empty_list(self):
        assert optional(List.of()) == List.of(Nothing())


class TestOptionalMethod:
    def test_some_becomes_some_of_some(self):
        assert Some(5).optional() == Some(Some(5))

    def test_nothing_becomes_some_of_nothing(self):
        assert Nothing().optional() == Some(Nothing())


class TestOptionalFreeFunction:
    def test_some(self):
        assert optional(Some(5)) == Some(Some(5))

    def test_nothing(self):
        assert optional(Nothing()) == Some(Nothing())


class TestOptionalAlwaysSucceeds:
    def test_optional_result_is_never_nothing(self):
        # optional itself never fails, regardless of the input's own success/failure
        assert optional(Some(1)) != Nothing()
        assert optional(Nothing()) != Nothing()


class TestEmptyIsAClassmethod:
    """Regression test: Maybe.empty as a class method not a property.

    Maybe.empty used to be a @property, which was convenient but
    broke guard(Maybe, ...) at least, which uses the class method
    f.empty() on the bare class. This need for an empty when there
    is no instance suggests a clasmethod is better than a property
    in practice. This matches pure for instance.

    """

    def test_guard_true_on_maybe(self):
        assert guard(Maybe, True) == Some(())

    def test_guard_false_on_maybe(self):
        assert guard(Maybe, False) == Nothing()

    def test_guard_true_on_list(self):
        assert guard(List, True) == List.of(())

    def test_guard_false_on_list(self):
        assert guard(List, False) == List()
