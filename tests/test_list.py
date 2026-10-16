"""Tests for NonEmptyList's non-empty invariant and for SequenceView."""

from __future__ import annotations

import pytest

from hypothesis import given
from hypothesis import strategies as st

from functorial.list import List, NonEmptyList, SequenceView


#
# List: More tests in test_examples.py
#

@pytest.mark.parametrize('label, operation, expected', [
    ('pop',              lambda x: x.pop(),                                          [1, 2, 3, 4]),
    ('remove',           lambda x: x.remove(2),                                      [1, 3, 4, 5]),
    ('del x[0]',         lambda x: x.__delitem__(0),                                 [2, 3, 4, 5]),
    ('del x[1:]',        lambda x: x.__delitem__(slice(1, None)),                    [1]),
    ('del x[::2]',       lambda x: x.__delitem__(slice(None, None, 2)),              [2, 4]),
    ('x[:] = [7]',       lambda x: x.__setitem__(slice(None), [7]),                  [7]),
    ('x[:] = iter(...)', lambda x: x.__setitem__(slice(None), iter([7, 8])),         [7, 8]),
    ('x[1:] = []',       lambda x: x.__setitem__(slice(1, None), []),                [1]),
    ('x[::-1] = ...',    lambda x: x.__setitem__(slice(None, None, -1), [9, 8, 7, 6, 5]),  [5, 6, 7, 8, 9]),
    ('x[0] = 5',         lambda x: x.__setitem__(0, 5),                              [5, 2, 3, 4, 5]),
    ('x *= 1',           lambda x: x.__imul__(1),                                    [1, 2, 3, 4, 5]),
    ('x *= 2',           lambda x: x.__imul__(2),                                    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
])
def test_mutable_list_operations(label, operation, expected):
    x = List.of(1, 2, 3, 4, 5)
    y = List([1, 2, 3, 4, 5])
    operation(x)
    assert x == expected, label
    operation(y)
    assert y == expected, label

@pytest.mark.parametrize('label, operation, expected', [
    ('x * 0',            lambda x: x.__mul__(0),                                    List()),
    ('x * 0',            lambda x: isinstance(x.__mul__(0), List),                  True),
    ('0 * x',            lambda x: x.__imul__(0),                                   List()),
    ('0 * x',            lambda x: isinstance(x.__mul__(0), List),                  True),
    ('x * 1',            lambda x: isinstance(x.__mul__(1), List),                  True),
    ('1 * x',            lambda x: isinstance(x.__mul__(1), List),                  True),
    ('x + x',            lambda x: isinstance(x.__add__(x), List),                  True),
    ('x + x',            lambda x: x.__add__(x),                                    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
    ('[10, 20] + x',     lambda x: x.__radd__([10, 20]),                            [10, 20, 1, 2, 3, 4, 5]),
])
def test_list_operation_checks(label, operation, expected):
    x = List.of(1, 2, 3, 4, 5)
    y = List([1, 2, 3, 4, 5])
    assert operation(x) == expected, label
    assert operation(y) == expected, label


#
# NonEmptyList: operations that would empty the list raise ValueError
#

def test_construction_requires_elements():
    with pytest.raises(ValueError, match='cannot be empty'):
        NonEmptyList([])
    assert NonEmptyList([1]) == [1]

@pytest.mark.parametrize('label, operation', [
    ('clear',          lambda x: x.clear()),
    ('pop',            lambda x: x.pop()),
    ('pop(0)',         lambda x: x.pop(0)),
    ('remove',         lambda x: x.remove(1)),
    ('del x[0]',       lambda x: x.__delitem__(0)),
    ('del x[:]',       lambda x: x.__delitem__(slice(None))),
    ('del x[::-1]',    lambda x: x.__delitem__(slice(None, None, -1))),
    ('del x[::2]',     lambda x: x.__delitem__(slice(None, None, 2))),
    ('x[:] = []',      lambda x: x.__setitem__(slice(None), [])),
    ('x[0:1] = []',    lambda x: x.__setitem__(slice(0, 1), [])),
    ('x[:] = iter([])', lambda x: x.__setitem__(slice(None), iter([]))),
    ('x *= 0',         lambda x: x.__imul__(0)),
    ('x *= -1',        lambda x: x.__imul__(-1)),
])
def test_singleton_cannot_be_emptied(label, operation):
    x = NonEmptyList([1])
    with pytest.raises(ValueError):
        operation(x)
    assert x == [1], f'{label} modified the list before raising'

def test_clear_raises_even_with_many_elements():
    x = NonEmptyList([1, 2, 3])
    with pytest.raises(ValueError, match='clear'):
        x.clear()
    assert x == [1, 2, 3]

@pytest.mark.parametrize('label, operation, expected', [
    ('pop',              lambda x: x.pop(),                                          [1, 2]),
    ('remove',           lambda x: x.remove(2),                                      [1, 3]),
    ('del x[0]',         lambda x: x.__delitem__(0),                                 [2, 3]),
    ('del x[1:]',        lambda x: x.__delitem__(slice(1, None)),                    [1]),
    ('del x[::2]',       lambda x: x.__delitem__(slice(None, None, 2)),              [2]),
    ('x[:] = [7]',       lambda x: x.__setitem__(slice(None), [7]),                  [7]),
    ('x[:] = iter(...)', lambda x: x.__setitem__(slice(None), iter([7, 8])),         [7, 8]),
    ('x[1:] = []',       lambda x: x.__setitem__(slice(1, None), []),                [1]),
    ('x[::-1] = ...',    lambda x: x.__setitem__(slice(None, None, -1), [9, 8, 7]),  [7, 8, 9]),
    ('x[0] = 5',         lambda x: x.__setitem__(0, 5),                              [5, 2, 3]),
    ('x *= 1',           lambda x: x.__imul__(1),                                    [1, 2, 3]),
    ('x *= 2',           lambda x: x.__imul__(2),                                    [1, 2, 3, 1, 2, 3]),
])
def test_operations_that_keep_elements_succeed(label, operation, expected):
    x = NonEmptyList([1, 2, 3])
    operation(x)
    assert x == expected, label

@pytest.mark.parametrize('label, operation, expected', [
    ('x * 0',            lambda x: x.__mul__(0),                                    List()),
    ('x * 0',            lambda x: isinstance(x.__mul__(0), List),                  True),
    ('0 * x',            lambda x: x.__mul__(0),                                    List()),
    ('0 * x',            lambda x: isinstance(x.__mul__(0), List),                  True),
    ('x * 1',            lambda x: isinstance(x.__mul__(1), NonEmptyList),          True),
    ('1 * x',            lambda x: isinstance(x.__mul__(1), NonEmptyList),          True),
])
def test_dunder_operations_checks(label, operation, expected):
    x = NonEmptyList([1, 2, 3])
    assert operation(x) == expected, label

def test_imul_accepts_supports_index():
    class Two:
        def __index__(self):
            return 2

    x = NonEmptyList([1])
    x *= Two()
    assert x == [1, 1]
    assert isinstance(x, NonEmptyList)


# Reference checks: a NonEmptyList behaves exactly like a list, except
# that any operation that would leave it empty raises ValueError and
# leaves it unchanged.

slice_bound = st.one_of(st.none(), st.integers(-7, 7))
slices = st.builds(slice, slice_bound, slice_bound,
                   st.one_of(st.none(), st.integers(-3, 3).filter(bool)))
elements = st.lists(st.integers(0, 9), min_size=1, max_size=6)

def _check_against_list(xs, operation):
    expected = list(xs)
    try:
        operation(expected)
    except ValueError:
        expected = None     # the list operation itself fails (e.g. extended slice size mismatch)

    x = NonEmptyList(xs)
    if expected is None or len(expected) == 0:
        with pytest.raises(ValueError):
            operation(x)
        assert x == xs
    else:
        operation(x)
        assert x == expected

@given(elements, slices)
def test_delitem_slice_matches_list(xs, key):
    def delete(target):
        del target[key]
    _check_against_list(xs, delete)

@given(elements, slices, st.lists(st.integers(0, 9), max_size=4))
def test_setitem_slice_matches_list(xs, key, values):
    def assign(target):
        target[key] = values
    _check_against_list(xs, assign)

@given(elements, st.integers(-2, 3))
def test_imul_matches_list(xs, n):
    def multiply(target):
        target *= n
    _check_against_list(xs, multiply)


#
# SequenceView
#

def test_view_whole_sequence():
    xs = [1, 2, 3]
    v = SequenceView(xs)
    assert len(v) == 3
    assert list(v) == xs
    assert v == xs

@pytest.mark.parametrize('offset, length, expected', [
    (2,  5,    [2, 3, 4, 5, 6]),
    (0,  0,    []),
    (8,  10,   [8, 9]),        # length clamped to the end
    (12, None, []),            # offset past the end
    (7,  None, [7, 8, 9]),     # to the end
])
def test_view_subrange(offset, length, expected):
    assert list(SequenceView(list(range(10)), offset, length)) == expected

@pytest.mark.parametrize('offset, length', [(-1, None), (0, -1)])
def test_view_rejects_negative_bounds(offset, length):
    with pytest.raises(ValueError, match='non-negative'):
        SequenceView([1, 2, 3], offset, length)

def test_view_indexing():
    v = SequenceView(list(range(10)), 2, 5)
    assert v[0] == 2
    assert v[-1] == 6
    assert v[4] == 6
    with pytest.raises(IndexError):
        v[5]
    with pytest.raises(IndexError):
        v[-6]

def test_view_sequence_mixin_methods():
    v = SequenceView(list(range(10)), 2, 5)
    assert 4 in v
    assert 7 not in v
    assert v.index(5) == 3
    assert v.count(3) == 1
    assert list(reversed(v)) == [6, 5, 4, 3, 2]

@pytest.mark.parametrize('key, expected', [
    (slice(1, 4),        [3, 4, 5]),
    (slice(None, None, 2),  [2, 4, 6]),
    (slice(None, None, -1), [6, 5, 4, 3, 2]),
    (slice(-2, None),    [5, 6]),
    (slice(10, 20),      []),
])
def test_view_slicing_returns_views(key, expected):
    v = SequenceView(list(range(10)), 2, 5)
    sliced = v[key]
    assert isinstance(sliced, SequenceView)
    assert list(sliced) == expected

def test_nested_slices_compose():
    v = SequenceView(list(range(10)), 2, 5)
    assert list(v[1:][::2]) == [3, 5]
    assert list(v[::-1][1:3]) == [5, 4]

def test_view_of_view_flattens():
    xs = list(range(10))
    inner = SequenceView(xs, 2, 5)
    outer = SequenceView(inner, 1, 2)
    assert list(outer) == [3, 4]
    assert outer._data is xs     # pylint: disable=protected-access

def test_view_of_view_clamps_to_inner():
    inner = SequenceView(list(range(10)), 2, 5)
    assert list(SequenceView(inner, 3, 10)) == [5, 6]

def test_view_of_stepped_and_reversed_views():
    xs = list(range(10))
    assert list(SequenceView(SequenceView(xs, 1, 8)[::2], 1)) == [3, 5, 7]
    assert list(SequenceView(SequenceView(xs)[::-1], 2, 3)) == [7, 6, 5]

@pytest.mark.parametrize('data', [(1, 2, 3, 4), range(1, 5), 'abcd', List([1, 2, 3, 4])])
def test_view_of_other_sequences(data):
    assert list(SequenceView(data, 1, 2)) == list(data[1:3])

@pytest.mark.parametrize('offset, length', [
    (0, 10),          # islice path
    (40, 10),         # islice path, at the crossover
    (41, 10),         # indexing path, just past the crossover
    (900, 50),        # indexing path, large offset
])
def test_iteration_paths_agree(offset, length):
    xs = list(range(1000))
    assert list(SequenceView(xs, offset, length)) == xs[offset:offset + length]

def test_view_is_read_only():
    v = SequenceView([1, 2, 3])
    with pytest.raises(TypeError):
        v[0] = 9      # type: ignore[index]
    with pytest.raises(TypeError):
        del v[0]      # type: ignore[attr-defined]
    assert not hasattr(v, 'append')

def test_view_is_live():
    xs = [1, 2, 3, 4]
    v = SequenceView(xs, 1, 2)
    xs[1] = 20
    assert list(v) == [20, 3]

def test_view_equality():
    xs = [1, 2, 3]
    assert SequenceView(xs) == [1, 2, 3]
    assert SequenceView(xs) == (1, 2, 3)
    assert SequenceView(xs) == SequenceView([0, 1, 2, 3], 1)
    assert SequenceView(xs) != [1, 2]
    assert SequenceView(xs) != [1, 2, 4]
    assert SequenceView('abc') != 'abc'     # str is never equal to a view

def test_view_is_unhashable():
    with pytest.raises(TypeError):
        hash(SequenceView([1, 2, 3]))

def test_view_repr():
    assert repr(SequenceView([1, 2, 3], 1)) == 'SequenceView([2, 3])'

def test_thaw_makes_independent_list():
    xs = [1, 2, 3, 4]
    v = SequenceView(xs, 1, 2)
    thawed = v.thaw()
    assert isinstance(thawed, List)
    assert thawed == [2, 3]
    thawed.append(99)
    xs[1] = 20
    assert thawed == [2, 3, 99]
    assert list(v) == [20, 3]
