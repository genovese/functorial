"""make_lens -- creates field accessor lenses from classes, dataclasses, and named tuples.


"""
from __future__ import annotations

import copy
import dataclasses

from collections.abc import Hashable
from operator        import itemgetter

from ..dict import Dict
from ..list import List
from ..pair import Pair

from .lens  import Lens, lens

__all__ = [
    'field_lens',
    'make_lenses',
    'at_key',
]


def own_field_names_class(cls) -> Pair[List[str], str]:
    """Get field names and type of class, dataclass, or named tuple, excluding inherited names."""

    if dataclasses.is_dataclass(cls):
        own = vars(cls).get('__annotations__', {})
        return Pair(List([f.name for f in dataclasses.fields(cls) if f.name in own]), 'dataclass')

    if issubclass(cls, tuple) and hasattr(cls, '_fields'):  # NamedTuple
        return Pair(List(cls._fields), 'NamedTuple')

    return Pair(List(vars(cls).get('__annotations__', {}).keys()), 'other')


def own_field_names(cls) -> list[str]:
    """Get field names of class, dataclass, or named tuple, excluding inherited names."""
    return own_field_names_class(cls)[0]

def field_lens(name):
    """ ATTN:Generic """
    def getter(s):
        return getattr(s, name)

    def setter(s, a):
        if dataclasses.is_dataclass(s):
            return dataclasses.replace(s, **{name: a})      # works for frozen and non-frozen

        if hasattr(type(s), '_replace'):                    # NamedTuple
            return s._replace(**{name: a})

        s2 = copy.copy(s)
        setattr(s2, name, a)
        return s2

    return lens(getter, setter)

def make_lenses(cls, requested: list[str] | None = None) -> Dict[str, Lens] | tuple[Lens]:
    """ATTN"""
    fields, typeof = own_field_names_class(cls)

    def getter(n):
        return lambda s: getattr(s, n)  # avoid late-binding error in loop definitions

    # avoid late-binding error in loop definitions by wrapping setter
    if typeof == 'dataclass':
        def setter(n):
            # Note this works for frozen and non-frozen dataclasses
            return lambda s, a: dataclasses.replace(s, **{n: a})
    elif typeof == 'NamedTuple':
        def setter(n):
            return lambda s, a: s._replace(**{n: a})
    else:
        def setter(n):
            def _setit(s, a):
                s2 = copy.copy(s)
                setattr(s2, n, a)
                return s2
            return _setit

    lenses: Dict = Dict()
    for name in fields:
        lenses[name] = lens(getter(name), setter(name))

    if requested is not None:  # Return tuple of only requested lenses in the same order
        if not requested or not set(requested) <= set(fields):
            raise KeyError('make_lenses: requested fields empty or contains invalid name')
        return itemgetter(*requested)(lenses)

    return lenses


def at_key(*at_keys: Hashable, default=None) -> Lens:
    n = len(at_keys)
    if n == 1:
        k = at_keys[0]

        # ATTN: have case that raises a key error, make this the default??
        def _key_getter(s):
            return s.get(k, default)

        def _key_setter(s, a):
            return s.__class__(s | {k: a})
    elif n == 0:
        raise ValueError('at_key requires at least one key to be supplied')
    else:
        # ATTN: have case that raises a key error, make this the default??
        def _key_getter(s):
            return s.__class__({k: s.get(k, default) for k in at_keys})

        def _key_setter(s, a):
            update = {at_keys[i]: a[i] for i in range(len(at_keys))}
            return s.__class__(s | update)

    return lens(_key_getter, _key_setter)


# Some tests
#
# List.of(Dict({'a':9, 'c':248, 'z':0}), Dict({'a':1, 'b': 3, 'c': 8})) >> collect(each @ at_key('a'))
# => [9, 1]
#
# List.of(Dict({'a':9, 'c':248, 'z':0}), Dict({'a':1, 'b': 3, 'c': 8})) >> collect(each @ at_key('a', 'c'))
# => [{'a': 9, 'c': 248}, {'a': 1, 'c': 8}]
#
# Dict({'a': 0, 'b': 1, 'c': 2}) >> put(at_key('a', 'c'), (10, 16))
# => {'a': 10, 'b': 1, 'c': 16}
#
# Dict({'a': 0, 'b': 1, 'c': 2}) >> over(at_key('a', 'c'), lambda x: (x['a'] + 10, x['c'] + 100))
# => {'a': 10, 'b': 1, 'c': 102}
#
# List.of(1, 2, 3, 4) >> over(at(1, 3), lambda x: (x[0] + 10, x[1] + 100))
# => [1, 12, 3, 104]
#
# List.of(Dict({'a':9, 'c':248, 'z':0}), Dict({'a':1, 'b': 3, 'c': 8})) >> over(each @ at_key('a', 'c'), lambda x: (x['a'] + 10, x['c'] + 100))
# => [{'a': 19, 'c': 348, 'z': 0}, {'a': 11, 'b': 3, 'c': 108}]
