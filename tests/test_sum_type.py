"""Tests for the SumType base class and SumTypeMeta metaclass."""

from __future__ import annotations

import copy
import importlib
import pickle
import sys
import threading
import time
import types

from abc                    import ABC
from concurrent.futures     import ThreadPoolExecutor
from dataclasses            import fields, is_dataclass

import pytest

from functorial.sum_type import SumType, SumTypeMeta, abstractmethod, field
from functorial.sum_type import _normalize_subclasses


#
# Module-level sum types used across tests.  These live at module level
# so that pickling can find them.  Generated variants of top-level roots
# are bound both on the root and in this module's globals; we fetch them
# from globals explicitly to keep linters happy.
#

class Shape(SumType, data={
        'Dot':    [],
        'Circle': [('r', float)],
        'Rect':   [('w', float), ('h', float, field(default=1.0))],
        'Poly':   None,
}):
    pass

Dot, Circle, Rect = (globals()[name] for name in ('Dot', 'Circle', 'Rect'))

class Poly(Shape):
    def __init__(self, *points):
        self.points = points


class Color(SumType, data=['Red', 'Green']):
    pass

class Red(Color, singleton=True):
    pass

class Green(Color, singleton=True):
    pass


#
# Data spec normalization
#

def test_normalize_dict_passes_through():
    spec = {'A': None, 'B': []}
    assert _normalize_subclasses(spec) == spec

def test_normalize_mixed_list():
    spec = ['A', {'B': [], 'C': [('x', int)]}]
    assert _normalize_subclasses(spec) == {'A': None, 'B': [], 'C': [('x', int)]}

def test_normalize_rejects_other_types():
    with pytest.raises(TypeError, match='list or dict'):
        _normalize_subclasses('ABC')    # type: ignore[arg-type]


#
# Generated variants
#

def test_generated_variants_are_subclasses():
    for variant in (Dot, Circle, Rect):
        assert issubclass(variant, Shape)
        assert isinstance(variant, SumTypeMeta)

def test_generated_variants_have_correct_module():
    for variant in (Dot, Circle, Rect):
        assert variant.__module__ == __name__
        assert variant.__qualname__ == variant.__name__

def test_unspecified_variant_is_not_generated():
    class Haunted(SumType, data={'Ghost': None, 'Ghoul': []}):
        pass

    assert not hasattr(Haunted, 'Ghost')
    assert issubclass(Haunted.Ghoul, Haunted)
    assert not hasattr(Shape, 'Poly')
    assert Poly(1, 2).points == (1, 2)

def test_top_level_variants_bound_on_root_and_module():
    for variant in (Dot, Circle, Rect):
        assert getattr(Shape, variant.__name__) is variant
        assert globals()[variant.__name__] is variant

def test_local_root_variants_bound_only_on_root():
    class Local(SumType, data={'LocalA': [], 'LocalB': [('x', int)]}):
        pass

    assert 'LocalA' not in globals() and 'LocalB' not in globals()
    assert Local.LocalA() is Local.LocalA()
    assert Local.LocalB(3).x == 3

    match Local.LocalB(4):
        case Local.LocalA():
            matched = None
        case Local.LocalB(x):
            matched = x
    assert matched == 4

def test_class_nested_root_variants_bound_only_on_root():
    class Outer:
        class Nested(SumType, data={'NestedLeaf': []}):
            pass

    assert 'NestedLeaf' not in globals()
    assert Outer.Nested.NestedLeaf() is Outer.Nested.NestedLeaf()

def _top_level_root(name, data):
    """Creates a root that looks top-level in this module, as if at module scope."""
    return types.new_class(name, (SumType,), {'data': data},
                           lambda ns: ns.update(__module__=__name__))

def test_generation_will_not_clobber_module_names():
    with pytest.raises(TypeError, match=f'would redefine {__name__}.Dot'):
        _top_level_root('Clash', {'Dot': []})
    assert globals()['Dot'] is Shape.Dot

def test_redefining_top_level_root_is_allowed():
    # Simulates importlib.reload or re-running a notebook cell
    try:
        first = _top_level_root('Rerun', {'RerunLeaf': []})
        second = _top_level_root('Rerun', {'RerunLeaf': []})
        assert globals()['RerunLeaf'] is second.RerunLeaf
        assert second.RerunLeaf is not first.RerunLeaf
    finally:
        globals().pop('RerunLeaf', None)

def test_module_reload_regenerates_variants(tmp_path, monkeypatch):
    (tmp_path / 'reload_target.py').write_text(
        'from functorial.sum_type import SumType\n'
        'class Reloadable(SumType, data={"Fresh": [], "Boxed": [("v", int)]}):\n'
        '    pass\n'
    )
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, 'reload_target', raising=False)

    module = importlib.import_module('reload_target')
    monkeypatch.setitem(sys.modules, 'reload_target', module)  # removed on teardown
    old_fresh = module.Fresh
    module = importlib.reload(module)

    assert module.Fresh is not old_fresh
    assert module.Fresh is module.Reloadable.Fresh
    assert module.Fresh() is module.Fresh()
    assert module.Boxed(1) == module.Boxed(1)

def test_generated_dataclass_fields():
    assert is_dataclass(Circle) and is_dataclass(Rect)
    assert [f.name for f in fields(Rect)] == ['w', 'h']
    assert Rect(2.0) == Rect(2.0, 1.0)
    assert Rect(2.0, 3.0) != Rect(3.0, 2.0)
    assert repr(Circle(1.0)) == 'Circle(r=1.0)'

def test_generated_match_args():
    assert Circle.__match_args__ == ('r',)
    assert Rect.__match_args__ == ('w', 'h')

def test_pattern_matching():
    def describe(s):
        match s:
            case Dot():
                return 'dot'
            case Circle(r):
                return f'circle {r}'
            case Rect(w, h):
                return f'rect {w}x{h}'
            case Poly():
                return 'poly'
        return 'unknown'

    assert describe(Dot()) == 'dot'
    assert describe(Circle(2.0)) == 'circle 2.0'
    assert describe(Rect(1.0, 2.0)) == 'rect 1.0x2.0'
    assert describe(Poly()) == 'poly'

def test_generated_variant_respects_abstract_methods():
    class Measured(SumType, data=['Square']):
        @abstractmethod
        def area(self): ...

    class Square(Measured):
        pass

    with pytest.raises(TypeError, match="abstract method 'area'"):
        Square()


#
# Singletons
#

@pytest.mark.parametrize('variant', [Red, Green, Dot])
def test_singleton_identity(variant):
    assert variant() is variant()

def test_distinct_singletons_are_distinct():
    assert Red() is not Green()
    assert Red() != Green()

@pytest.mark.parametrize('variant', [Red, Dot])
def test_singleton_survives_copy(variant):
    assert copy.copy(variant()) is variant()
    assert copy.deepcopy(variant()) is variant()
    assert copy.deepcopy([variant(), variant()])[1] is variant()

@pytest.mark.parametrize('variant', [Red, Dot])
def test_singleton_survives_pickle(variant):
    assert pickle.loads(pickle.dumps(variant())) is variant()

def test_non_singleton_pickles_by_value():
    r = Rect(2.0, 3.0)
    r2 = pickle.loads(pickle.dumps(r))
    assert r2 == r and r2 is not r

def test_user_reduce_is_not_overridden():
    def custom_reduce(self):
        return (type(self), ())

    class Toggle(SumType, data=['On']):
        pass

    class On(Toggle, singleton=True):
        __reduce__ = custom_reduce

    assert On.__dict__['__reduce__'] is custom_reduce

def test_singleton_creation_is_thread_safe():
    created = []

    class Lazy(SumType, data=['Only']):
        pass

    class Only(Lazy, singleton=True):
        def __init__(self):
            time.sleep(0.01)    # widen the race window
            created.append(threading.get_ident())

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: Only(), range(32)))

    assert len(created) == 1
    assert all(r is results[0] for r in results)


#
# Instantiation restrictions
#

@pytest.mark.parametrize('cls', [SumType, Shape, Color])
def test_roots_cannot_be_instantiated(cls):
    with pytest.raises(TypeError, match='cannot be instantiated'):
        cls()

def test_register_is_forbidden():
    with pytest.raises(TypeError, match='closed sum type'):
        Shape.register(int)
    assert not isinstance(3, Shape)


#
# Class definition errors
#

@pytest.mark.parametrize('name, bases, kwds, message', [
    ('Zap',   (Shape,),        {},                                  'not an allowed subclass'),
    ('Sub',   (Red,),          {},                                  'variants are final'),
    ('Sub',   (Circle,),       {},                                  'variants are final'),
    ('Red',   (Shape, Color),  {},                                  'only one SumType superclass'),
    ('Red',   (Color,),        {'data': ['X']},                     'should not provide a data'),
    ('Root',  (SumType,),      {'data': ['X'], 'singleton': True},  'only valid on a sum type variant'),
    ('Lone',  (SumType,),      {'singleton': True},                 'only valid on a sum type variant'),
    ('Point', (SumType,),      {'data': ['Point', 'Other']},        'cannot list itself'),
    ('Mid',   (SumType,),      {},                                  'must specify allowed subclasses'),
    ('Root',  (SumType,),      {'data': 'XY'},                      'list or dict'),
])
def test_definition_errors(name, bases, kwds, message):
    with pytest.raises(TypeError, match=message):
        types.new_class(name, bases, kwds)

def test_disallowed_subclass_message_lists_allowed_sorted():
    with pytest.raises(TypeError, match='allowed subclasses are Circle, Dot, Poly, Rect$'):
        types.new_class('Zap', (Shape,), {})

def test_explicit_variant_can_be_redefined():
    class Recolor(SumType, data=['Blue']):
        pass

    class Blue(Recolor):        # pylint: disable=unused-variable
        pass

    class Blue(Recolor):        # pylint: disable=function-redefined
        pass

    assert issubclass(Blue, Recolor)


#
# Interaction with other classes and metaclasses
#

def test_abc_mixin_on_root():
    class Pretty(ABC):
        @abstractmethod
        def tag(self): ...

        def show(self):
            return f'<{self.tag()}>'

    class Tree(Pretty, SumType, data=['Tip', 'Node']):
        def tag(self):
            return type(self).__name__

    class Tip(Tree, singleton=True):
        pass

    class Node(Tree):
        def __init__(self, value):
            self.value = value

    assert type(Tree) is SumTypeMeta
    assert Tip().show() == '<Tip>'
    assert Node(3).show() == '<Node>'
    assert isinstance(Node(3), Pretty)
    with pytest.raises(TypeError):
        Tree()

def test_mixin_after_root_on_variant():
    class Mixin:
        def hello(self):
            return 'hello'

    class Greeting(SumType, data=['Hi']):
        pass

    class Hi(Mixin, Greeting):
        pass

    assert Hi().hello() == 'hello'

def test_cooperates_with_strict_metaclass_init():
    class StrictMeta(type):
        def __init__(cls, name, bases, ns):
            super().__init__(name, bases, ns)

    class Combined(SumTypeMeta, StrictMeta):
        pass

    class Strict(SumType, metaclass=Combined, data=['Yes']):
        pass

    class Yes(Strict, singleton=True):
        pass

    assert Yes() is Yes()

def test_generic_sum_type():
    class Opt[T](SumType, data=['Empty', 'Has']):
        pass

    class Empty(Opt, singleton=True):
        pass

    class Has[T](Opt[T]):
        def __init__(self, value: T):
            self.value = value

    assert Empty() is Empty()
    assert Has[int](3).value == 3
    assert isinstance(Has(3), Opt)
