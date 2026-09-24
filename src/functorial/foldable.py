#
# trait Foldable (f : Type -> Type) where
#     fold_map : Monoid m => (a -> m) -> f a -> m
#     fold  : (a -> b -> a) -> a -> f b -> a
#
# trait IndexedFoldable (f : Type -> Type) where
#     ifold_map : Monoid m => (i -> a -> m) -> f a -> m
#     ifold  : (i -> a -> b -> a) -> a -> f b -> a
#
# ruff: noqa: N802
# pylint: disable=invalid-name


from __future__      import annotations

from abc             import abstractmethod
from collections.abc import Callable
from typing          import Protocol

from .applicative    import Applicative, IdentityA, ap_second
from .functions      import identity
from .maybe          import Maybe, Some, Nothing, maybe
from .monoids        import Conjunction, Disjunction, Endo, First, Monoid


__all__ = ['Foldable', 'Foldable_', 'IndexedFoldable', 'IndexedFoldable_', 'fold_map', 'fold', 'ifold_map', 'ifold',]


class Foldable[A](Protocol):
    """A structure that can be folded over, i.e., converted to a list."""
    @abstractmethod
    def fold_map[M](self, f: Callable[[A], M], monoid: Monoid) -> M:
        "fold_map : Monoid m => Self -> (a -> m) -> m"
        ...

# ATTN: should fold_right be part of the protocol, thinking not

# ATTN: This is a useful convention with the Protocol classes
# Have the basic protocol X and a class X_ that inherits from X
# but also provides default implementations of other methods.
#
# Here, we need fold_right, traverse_, and other useful things that
# follow from baseline. Instances can override these as needed.
#
class Foldable_[A](Foldable[A]):
    """Foldable base class for inheritance, with default implementations.

    Single primitive: fold_map. Everything else is derived and can be
    overridden for efficiency. Prefer inheriting from this over implementing
    Foldable directly.
    """
    @abstractmethod
    def fold_map[M](self, f: Callable[[A], M], monoid: Monoid) -> M:
        ...

    def fold_right[B](self, f: Callable[[A, B], B], initial: B) -> B:
        """Right fold derived from fold_map via the Endo monoid."""
        return self.fold_map(lambda a: lambda b: f(a, b), Endo)(initial)

    def fold[B](self, f: Callable[[B, A], B], initial: B) -> B:
        """Standard left fold. Note argument order in folding function.

        This left fold derived from fold_right via difference lists.
        This is correct but has O(n) stack depth. Override directly for
        strict efficiency (most containers should).

        Returns the folded result.

        """
        return self.fold_right(lambda a, k: lambda acc: k(f(acc, a)), identity)(initial)  # type: ignore[return-value, arg-type]

    def to_list(self) -> list[A]:
        """Collects all elements into a list in left-to-right order.

        NOTE: Returns a plain list not a List to avoid circular imports!!
        This is the same issue in to_list and map_maybe.  So these
        should be used sparingly or wrapped in List.

        """
        return self.fold_right(lambda a, acc: [a] + acc, [])

    # traverse_ : (a -> f b) -> t a -> f ()
    def traverse_(self, f: Callable[[A], Applicative], effect: type[Applicative] = IdentityA) -> Applicative:
        def act(x: A, eff: Applicative):
            return ap_second(f(x), eff)  # f x *> eff
        return self.fold_right(act, effect.pure(()))

    def find(self, pred: Callable[[A], bool]) -> Maybe[A]:
        def _find_it(x: A):
            return Some(x) if pred(x) else Nothing()

        return self.fold_map(_find_it, First)

    def any(self, pred: Callable[[A], bool]) -> bool:
        return self.fold_map(pred, Disjunction)

    def all(self, pred: Callable[[A], bool]) -> bool:
        return self.fold_map(pred, Conjunction)

    def concat_map[B](self, f: Callable[[A], list[B]]) -> list[B]:
        """Maps a list producing function over the structure concatenating into one list.

        NOTE: Returns a plain list not a List to avoid circular imports!!
        This is the same issue in to_list and map_maybe.  So these
        should be used sparingly or wrapped in List.

        """
        def _extend(acc: list[B], a: A) -> list[B]:
            acc.extend(f(a))
            return acc
        return self.fold(_extend, [])

    def map_maybe[B](self, f: Callable[[A], Maybe[B]]) -> list[B]:
        """Maps a function over the structure, keeping only the present results.

        NOTE: Returns a plain list not a List to avoid circular imports!!
        This is the same issue in to_list and concat_map.  So these
        should be used sparingly or wrapped in List.

        """
        return self.concat_map(lambda a: maybe([], lambda b: [b], f(a)))


class IndexedFoldable[I, A](Foldable[A], Protocol):
    @abstractmethod
    def ifold_map[M](self, f: Callable[[I, A], M], monoid: Monoid) -> M:
        ...


class IndexedFoldable_[I, A](IndexedFoldable[I, A], Foldable_[A]):
    """IndexedFoldable base class for inheritance, with default implementations.

    Single primitive: ifold_map. fold_map is derived by forgetting the index,
    which unblocks all Foldable_ defaults (fold_right, fold, to_list, etc.).
    Indexed variants (ifold_right, ifold, to_indexed_list) are also derived.
    """
    @abstractmethod
    def ifold_map[M](self, f: Callable[[I, A], M], monoid: Monoid) -> M:
        ...

    def fold_map[M](self, f: Callable[[A], M], monoid: Monoid) -> M:
        """Fold ignoring the index; derived from ifold_map."""
        return self.ifold_map(lambda _i, a: f(a), monoid)

    def ifold_right[B](self, f: Callable[[I, A, B], B], initial: B) -> B:
        """Indexed right fold derived from ifold_map via the Endo monoid."""
        return self.ifold_map(lambda i, a: lambda b: f(i, a, b), Endo)(initial)

    def ifold[B](self, f: Callable[[I, B, A], B], initial: B) -> B:
        """Indexed left fold derived from ifold_right via difference lists.

        This is correct but has O(n) stack depth. Override for efficiency.
        """
        return self.ifold_right(
            lambda i, a, k: lambda acc: k(f(i, acc, a)), identity  # type: ignore[return-value, arg-type]
        )(initial)

    def to_indexed_list(self) -> list[tuple[I, A]]:
        """Collects all (index, element) pairs in left-to-right order."""
        return self.ifold_right(lambda i, a, acc: [(i, a)] + acc, [])

    def itraverse_(self, f: Callable[[I, A], Applicative], effect: type[Applicative] = IdentityA) -> Applicative:
        """Effect-discarding indexed traversal, derived from ifold_right."""
        def act(i: I, a: A, eff: Applicative) -> Applicative:
            return ap_second(f(i, a), eff)
        return self.ifold_right(act, effect.pure(()))


#
# Generic Functions
#

def fold_map[A, M](f: Callable[[A], M], xs: Foldable[A], monoid: Monoid) -> M:
    """Fold over a structure, converting each component to a monoid and combining.

    This will typically operate over Functors, but it is fine to define these
    methods for any object (e.g., any Iterable).

    Parameters

    + f : a -> m -- function that maps elements of the structure to a monoid
    + xs  -- The structure to be folded.
    + monoid -- a Monoid object specifying how to combine mapped elements.

    Returns the final monoidal result, which is the unit of the monoid if the
    structure xs is empty.

    """
    return xs.fold_map(f, monoid)

def fold[A, B](f: Callable[[B, A], B], initial: B, xs: Foldable_[A]) -> B:
    """Fold over a structure accumulating a result from an initial value.

    This is a general form of functools.reduce. It is designed to work
    over a suitable Functor but the fold method can be defined for any
    type of object.

    Parameters:

    + f : (b -> a -> b) -- A function that updates an accumulator (of type b)
        given an element of the structure (of type a)
    + initial : b  -- The initial accumulator. This is returned when the
        structure is empty
    + xs - The structure to be folded over.

    Returns the final accumulator value.

    """
    return xs.fold(f, initial)

def ifold_map[I, A, M](f: Callable[[I, A], M], xs: IndexedFoldable[I, A], monoid: Monoid) -> M:
    """Fold over an indexed structure, converting each component to a monoid and combining.

    The indexes are intrinsic to the structure and are typically obtained by
    an IndexedFunctor instance, though this can be defined for more general objects.

    Parameters

    + f : i -> a -> m -- function that maps indices and elements of the structure
        to a monoid

    + xs  -- The structure to be folded. This will typically be a Functor, but
        can be defined for other objects as well (e.g., any Iterable).

    + monoid -- a Monoid object specifying how to combine mapped elements.

    Returns the final monoidal result, which is the unit of the monoid if the
    structure xs is empty.

    """
    return xs.ifold_map(f, monoid)


def ifold[I, A, B](f: Callable[[I, B, A], B], xs: IndexedFoldable_[I, A], initial: B) -> B:
    """Fold over a structure accumulating a result from an initial value.

    This is a general form of functools.reduce. It is designed to work
    over a suitable Functor but the fold method can be defined for any
    type of object.

    Parameters:

    + f : (b -> a -> b) -- A function that updates an accumulator (of type b)
        given an element of the structure (of type a)
    + initial : b  -- The initial accumulator. This is returned when the
        structure is empty
    + xs - The structure to be folded over.

    Returns the final accumulator value.

    """
    return xs.ifold(f, initial)
