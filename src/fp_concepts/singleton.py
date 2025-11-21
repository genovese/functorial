"""Metaclass for singleton objects.

"""

from __future__ import annotations
from abc        import ABCMeta
from threading  import Lock

__all__ = [
    'FinalAttribute',
    'Singleton',
    'SingletonFromABC',
    'ContingentSingleton',
    'ContingentSingletonFromABC',
]

_MISSING = object()  # Unique object to detect missing arguments


class FinalAttribute:
    """Descriptor that makes an attribute `final`, raising an error if it is modified once set.

    If a value is passed to the descriptor, that is used as the final value
    of the attribute. This can include None.

    Usage:

         class Foo:
             a = FinalAttribute()
             b = FinalAttribute(42)

             def __init__(self, some_val):
                 self.a = some_val  # This is not final

    Of course, there are ways around this restriction, but the
    goal is to prevent innocent changes.

    """
    def __init__(self, value=_MISSING):
        self.value = value

    def __set_name__(self, owner, name):
        self.private_name = "_" + name      # pylint: disable=attribute-defined-outside-init
        if self.value is not _MISSING:
            setattr(owner, self.private_name, self.value)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if hasattr(obj, self.private_name):
            raise AttributeError(f'The {self.private_name} attribute cannot be modified once set.')
        setattr(obj, self.private_name, value)

class Singleton(type):
    """Metaclass for classes that are used as singletons.

    This ensures that only one instance of the class is ever created,
    and it uses locking to also be thread safe.

    Use as

        class Foo(Baseclass, metaclass=Singleton):
            ...

    """
    _instances: dict = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class SingletonFromABC(ABCMeta):
    """Metaclass for classes that are used as singletons with an abstract base class.

    This ensures that only one instance of the class is ever created,
    and it uses locking to also be thread safe.

    Use as

        class Foo(Baseclass, metaclass=SingletonFromABC):
            ...

    when Baseclass or its ancestor inherits from ABC.

    """
    _instances: dict = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class ContingentSingleton(type):
    """Metaclass for classes that can be used as singletons or not.

    If there is a _IS_SINGLETON class attribute that is True,
    the class will be treated as a singleton. Otherwise,
    it will behave like an ordinary type. We are strict on
    the value being True rather than just truthy for
    definiteness.

    The desired pattern is to have a class attribute

        class Foo:
            IS_SINGLETON = FinalAttribute(True)

    If set, this ensures that only one instance of the class is ever
    created, and it uses locking to also be thread safe.

    The primary use case for this is for the sum-type inheritance
    where particular types in the sums are singular and we
    want to have a single instantiation. Examples include

        Maybe a = Nothing | Some a
        BinaryTree a = Tip | Branch (BinaryTree a) a (BinaryTree a)

    Here, both Nothing and Tip are more convenient if we can define
    them singularly while having those values be data of the
    corresponding type.  We can pattern match and reuse as needed.

    The reason this dance is needed in these cases that all
    the objects in the hierarchy must have the same metaclass.
    This way, we can give this metaclass up the chain, having
    the desired effect only where needed. Now it's true that
    class data can be changed, but the point isn't to stop
    an adversary but to make normal usage, e.g., Nothing() or Tip(),
    not produce spurious objects.

    Use as

        class Foo(Baseclass, metaclass=ContingentSingleton):
            ...

    """
    _instances: dict = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        if hasattr(cls, '_IS_SINGLETON') and cls._IS_SINGLETON is True:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
            return cls._instances[cls]
        return super().__call__(*args, **kwargs)

class ContingentSingletonFromABC(ABCMeta):
    """Metaclass for classes with abstract ancestor that can be used as singletons or not.

    If there is a _IS_SINGLETON class attribute that is True,
    the class will be treated as a singleton. Otherwise,
    it will behave like an ordinary type. We are strict on
    the value being True rather than just truthy for
    definiteness.

    The desired pattern is to have a class attribute

        class Foo:
            IS_SINGLETON = FinalAttribute(True)

    If set, this ensures that only one instance of the class is ever
    created, and it uses locking to also be thread safe.

    The primary use case for this is for the sum-type inheritance
    where particular types in the sums are singular and we
    want to have a single instantiation. Examples include

        Maybe a = Nothing | Some a
        BinaryTree a = Tip | Branch (BinaryTree a) a (BinaryTree a)

    Here, both Nothing and Tip are more convenient if we can define
    them singularly while having those values be data of the
    corresponding type.  We can pattern match and reuse as needed.

    The reason this dance is needed in these cases that all
    the objects in the hierarchy must have the same metaclass.
    This way, we can give this metaclass up the chain, having
    the desired effect only where needed. Now it's true that
    class data can be changed, but the point isn't to stop
    an adversary but to make normal usage, e.g., Nothing() or Tip(),
    not produce spurious objects.

    Note that this forces all classes in the hierarchy to use
    this metaclass. This is mostly useful in specialized cases.

    Use as

        class Foo(Baseclass, metaclass=ContingentSingletonFromABC):
            ...

    Here, Baseclass or an ancestor should inherit from ABC.

    """
    _instances: dict = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        if hasattr(cls, '_IS_SINGLETON') and cls._IS_SINGLETON is True:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
            return cls._instances[cls]
        return super().__call__(*args, **kwargs)
