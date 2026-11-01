"""Class and Metaclass infrastructure for sum types.

SumType is a base class with several related features:

  1. Its metaclass inherits from ABCMeta, making its
     direct inheritor an *abstract base class*.

  2. It restricts subclasses to a designated *white list*,
     via a data argument at class declaration.
     These subclasses cannot be further subclassed.

  3. It *auto-creates classes* for the "data constructors"
     on the whitelist. These classes can be overridden.

  4. It makes data constructors with no data associated
     *singletons*.

  5. It also sets =__match_args__= appropriately for
     these data-constructor subclasses.

The simplest way to create a sum type is by making
a base class that inherits from SumType and specifying
the names of the allowed subclasses with the `data`
keyword. For example,

   class MySum(SumType, data=['A', 'B', 'C']):
       ...

creates an abstract base class MySum representing the type
constructor and allowed subclasses A, B, and C representing type
constructors. In this form, the classes A, B, and C need to be
defined explicitly, but an error is raised if any subclasses of
MySum or of A, B, or C is defined. An error is also raised if MySum
is instantiated or if A, B, or C are instantiated without defining
any @abstractmethod's defined in MySum.

In general, with a different form of the `data` keyword, the
subclasses will be auto-generated. See the examples below. The
`data` value can be either be a dictionary mapping subclass names to
specifications or a list of either subclass names or specification
dictionaries.

The specification dictionaries map subclass names to either None or
to a list of tuples representingn data fields. The tuples are either
of the form (name, type) or (name, type, field_spec) using the
`field` function exported here (from dataclasses). None is
equivalent to just listing the subclass name. An empty list creates
a singleton subclass with no data. And the list of tuples create
dataclasses with the corresponding fields. All generated subclasses
are subclasses of the defined base class.

Examples
  1. Specifying allowed subclasses

    class MySum(SumType, data=['A', 'B', 'C']):
        ...

  or equivalently

    class MyEnum(SumType, data={'Foo': None, 'Bar': None, 'Zap': None}):
        ...

  As described above, MySum is an abstract base class
  with allowed subclasses A, B, and C.

  2. Simple Sums

    class MyEnum(SumType, data={'Foo': [], 'Bar': [], 'Zap': []}):
        ...

  defines MyEnum as an abstract base class with allowed subclasses
  Foo, Bar, and Zap. Moreover, these subclasses are predefined
  and set up as *singletons*.  These classes can be (re)defined explicitly
  as needed. This is equivalent to the TL1 type

     data MyEnum = Foo | Bar | Zap

  with Foo(), Bar(), and Zap() as the unique values of the types.

  3. Auto-generated general Sums

    class MyStuff(SumType, data={'Simple': [], 'Medium': [('v', int)], 'Complex': [('v', tuple[int, float])]}):
        ...

  or less concisely

    from dataclasses import field

    MyStuff_Comstructors = {
        'Simple': [],
        'Medium': [('v', int, field(default=0, init=True))],
        'Complex': [('r', int | float),
                    ('i', int | float, field(default=0, init=True))]
    }

    class MyStuff(SumType, data=MyStuff_Constructors):
        ...

    The empty list makes Simple a singleton empty class, as in example #2 above.
    Otherwise, the list
    We can optionally use `field` in the


  4. Singleton subclasses

    By specifying `singleton=True` in a subclass definition,
    the class is setup as a singleton. It should have no
    mutable fields.

    class C(MySum, singleton=True):
        pass

  Note on Auto-generation and Type Checking
  -----------------------------------------
  Note that the auto-generated classes will not be detectable by mypy,
  so as convenient as it is, the auto-generation feature is most useful
  at the REPL or in informal scripts where we don't worry about type
  checking.

"""

from __future__  import annotations
from abc         import abstractmethod
from dataclasses import make_dataclass, field
from threading   import Lock
from typing      import Any, Protocol

import sys

from functorial.singleton import FinalAttribute

__all__ = [
    'abstractmethod',  # Re-exported from abc
    'field',           # Re-exported from dataclasses
    'FinalAttribute',  # Re-exported from functorial.singleton
    'SumType',
    'SumTypeInvalidCaseException',
    'SumTypeMeta',
]

type SubClassSpec = dict[str, list[tuple[str, Any] | tuple[str, Any, Any]] | None]


def _normalize_subclasses(
        data: list[str | SubClassSpec] | SubClassSpec | None
) -> SubClassSpec:
    """ATTN
    """
    if isinstance(data, dict):
        return data
    if not isinstance(data, list):
        raise TypeError('supplied SumType data spec should be a list or dict.')

    subclss: SubClassSpec = {}
    for spec in data:
        if isinstance(spec, str):
            subclss[spec] = None  # Unspecified
        elif isinstance(spec, dict):
            subclss |= spec

    return subclss

def _is_stale_variant(obj, root_type) -> bool:
    """Checks if obj is a variant from an earlier definition of the same root type."""
    old_root = getattr(obj, '_root_type', None)
    return (isinstance(obj, SumTypeMeta)
            and old_root is not None
            and old_root is not root_type
            and old_root.__module__ == root_type.__module__
            and old_root.__qualname__ == root_type.__qualname__)


# The standard (and original) approach here is to have SumTypeMeta inherit
# from ABCMeta. But this causes a metaclass conflict for sum types that
# inherit from Protocol-based classes like Monad, Traversable, ....
# Since this is a primary use case, we change the metaclass to the
# same metaclass used for Protocol. Unfortunately, this is typing's
# *private*  metaclass _ProtocolMeta, which is also a subclass
# of ABCMeta. Deriving from it instead of ABCMeta directly solves
# our problem with no significant downside, *except* that the private
# metaclass is subject to change with Python versions. So, we have
# to detect that and use a computed base class for SumTypeMeta,
# which upsets mypy, which is why the type: ignore is used below.

class SumTypeMeta(type(Protocol)):  # type: ignore[misc]
    """Metaclass for generating true sum types.

    Types created with this metaclass accept data and singleton arguments
    to specify features of the concrete subclasses. See SumType
    for details.

    """
    def __new__(mcls, name, bases, class_dict, *, data=None, singleton=False, **kwargs):
        allowed_subclasses = None   # subclass whitelist
        sum_type_base = None        # nearest base in sum type hierarchy
        is_root_type = False        # we are defining the base sum type

        for base in bases:
            if hasattr(base, '_allowed_subclasses'):
                if sum_type_base is not None:
                    raise TypeError('A sum type subclass can have only one SumType superclass')

                allowed_subclasses = base._allowed_subclasses
                sum_type_base = base

        if sum_type_base is not None:
            if sum_type_base.__name__ in allowed_subclasses:
                emesg = f'SumType variants are final; cannot subclass {sum_type_base.__name__}'
                raise TypeError(emesg)

            if data is not None:
                raise TypeError('A sum type subclass definition should not provide a data argument')

        if allowed_subclasses is not None and name not in allowed_subclasses:
            emesg = (f'{name} is not an allowed subclass of {sum_type_base.__name__}; '
                     f'allowed subclasses are {", ".join(sorted(allowed_subclasses))}')
            raise TypeError(emesg)

        if sum_type_base is None:
            if singleton:
                raise TypeError('singleton=True is only valid on a sum type variant')
            if data is not None:
                data_spec = _normalize_subclasses(data)
                allowed_subclasses = set(data_spec.keys())
                is_root_type = True
                if name in allowed_subclasses:
                    raise TypeError(f'Sum type {name} cannot list itself as a variant')
            elif any(isinstance(base, SumTypeMeta) for base in bases):
                raise TypeError('Immediate descendants of SumType must specify allowed subclasses with data argument')
        elif singleton:
            class_dict['_is_singleton'] = FinalAttribute(True)
            class_dict.setdefault('__reduce__', lambda self: (type(self), ()))
            # Calling a subscripted generic, e.g. Nothing[int](), tries to set
            # __orig_class__ on the result. This unreadable, unsettable property
            # makes that a no-op, so the shared instance is never mutated.
            class_dict.setdefault('__orig_class__', property())

        # Ordinary variants get a metaclass with C-level construction.
        # Combined user metaclasses keep the general (slower) path.
        target = mcls
        if sum_type_base is not None and not singleton and mcls is SumTypeMeta:
            target = _VariantMeta

        cls = super().__new__(target, name, bases, class_dict, **kwargs)

        cls._is_root_type = is_root_type
        if is_root_type:
            cls._allowed_subclasses = allowed_subclasses
            cls._root_type = cls
            cls._instances = {}   # Instances for singleton subclasses
            cls._lock = Lock()    # Make singletons thread-safe

            mcls._generate_subclasses(cls, data_spec)

        return cls

    def __init__(cls, name, bases, ns, *, data=None, singleton=False, **kwargs):
        super().__init__(name, bases, ns, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls is SumType:
            raise TypeError('SumType cannot be instantiated')

        if cls._is_root_type:
            emesg = f'SumType base class {cls.__name__} cannot be instantiated'
            raise TypeError(emesg)

        if getattr(cls, '_is_singleton', False):
            # Check before locking; lock only to create the instance
            instance = cls._instances.get(cls)
            if instance is None:
                with cls._lock:
                    instance = cls._instances.get(cls)
                    if instance is None:
                        instance = super().__call__(*args, **kwargs)
                        cls._instances[cls] = instance
            return instance
        return super().__call__(*args, **kwargs)

    def register(cls, _subclass):
        emesg = f'{cls.__name__} is a closed sum type; virtual subclasses are not allowed'
        raise TypeError(emesg)

    @classmethod
    def _generate_subclasses(mcls, root_type: type, subclasses: SubClassSpec) -> None:
        module = sys.modules[root_type.__module__]

        for subclass, fields in subclasses.items():
            if fields is None:  # Do not generate
                continue

            generated: type | SumTypeMeta
            if len(fields) == 0:  # singleton, empty class
                s_dict = {'__module__': root_type.__module__, '__qualname__': subclass}
                generated = mcls(subclass, (root_type,), s_dict, singleton=True)
            else:
                # fields is a list of tuples of the form
                # (name, type) or (name, type, field(...))
                generated = make_dataclass(
                    cls_name=subclass,
                    fields=fields,
                    bases=(root_type,),
                    module=module.__name__
                )
            setattr(root_type, subclass, generated)           # Super.Sub always available
            if root_type.__qualname__ == root_type.__name__:  # top level root type
                current_binding = getattr(module, subclass, None)
                if (current_binding is not None
                    and current_binding is not generated
                    and  not _is_stale_variant(current_binding, root_type)):
                    emesg = f'Generating {subclass} would redefine {module.__name__}.{subclass}'
                    raise TypeError(emesg)
                setattr(module, subclass, generated)          # ok to bind in module

class _VariantMeta(SumTypeMeta):
    """Metaclass for ordinary (non-singleton) sum type variants.

    Using this for ordinary variants restores type's C-level
    __call__, bypassing the checks in SumTypeMeta.__call__, which
    never apply to these classes. This makes constructing variants
    about as fast as for plain classes, which can be non-trivial
    with oft-used Sum types like Maybe and Either.

    """
    __call__ = type.__call__

class SumType(metaclass=SumTypeMeta):
    """A base class for sum types.

    The basic structure of a sum type is that the *type* is represented by an
    abstract base class, which directly inherits from SumType, and
    the allowed subtypes are specified when that base class is defined.
    The concrete classes can then be defined as needed, though subclasses
    with no data are typically set as singleton's (singleton=True).
    Subclasses can be auto-generated, though this is most useful for
    simple scripts or at the REPL.

    Enumeration-style Sum Types
    ---------------------------
    For example, a TL1 type of the form

       data X = A | B | C

    is created with

       class X(SumType, data=['A', 'B', 'C'])
           ...

    This sets the allowed subtypes to be A, B, and C
    without further subclasses. We can then define

       class A(X, singleton=True):
           pass

       class B(X, singleton=True):
           pass

       class C(X, singleton=True):
           pass

    putting any desired methods (but no data).

    These can be auto-generated with

       class X(SumType, data={'A': [], 'B': [], 'C': []})
           ...

    though mypy etc will not know about these classes.

    More general variants
    ---------------------
    The subclasses can have any structure or data desired.

       class X(SumType, data=['A', 'B', 'C'])
           ...

       @dataclass
       class A(SumType):
           n: int
           s: str

       class B(SumType):
           ...

       class C(SumType):
           ...

    and so forth. Again, these can be auto-generated as
    dataclasses with a more complicated spec

       class X(SumType, data={'A': [('n', int), ('s', str)], 'B': [...], 'C': [...]})
           ...

    which will define the fields, match args, and even defaults (using field).
    But again auto-generation is most useful where type checking is not an issue
    as mypy cannot see them.

    See also the module docstring for more examples and details.

    """
    __slots__ = ()


class SumTypeInvalidCaseException(Exception):
    def __init__(self, seen, root_type: type[SumType]):
        name = root_type.__name__
        allowed = root_type._allowed_subclasses
        mesg = f'Given value {seen} for sum type {name} is not an allowed instance {allowed}.'
        self._root = root_type
        super().__init__(mesg)
