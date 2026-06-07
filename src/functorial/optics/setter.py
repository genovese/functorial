"""Setter optics and modification operations.

A Setter is the most general read-write optic. Any optic that has a
wander or visit method (Traversal, AffineTraversal, Lens, Prism,
Iso) can be used for modification via over and put.

The canonical approach is to pass Star(Identity) into the optic and
unwrap. See Star in optics/profunctors.py.

"""

from __future__   import annotations

from typing       import Callable

from ..identity   import Identity
from ..functions  import Function, compose, const

from .optic       import Optic, OpticIs
from .profunctors import Star

__all__ = ['Setter', 'puts', 'over', 'put']


class Setter(Optic, optic_is=OpticIs.SETTER):
    """An optic that supports modification but not reading."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.SETTER)


def puts(f: Callable) -> Setter:
    """Builds a Setter from an update function (a -> b) -> s -> t.

    The given function shows how to update the entire structure
    given a method for updating a specific focus.

    puts : ((a -> b) -> s -> t) -> Setter s t a b

    """
    def the_setter(p):
        g = compose(Identity.run, Star.run(p))   # a -> b
        return Star(compose(Identity, f(g)), Identity)
    return Setter(the_setter)


def over(optic, f: Callable) -> Function:
    """Modifies all foci with a given function, returning a function that updates the whole structure.

    This works for Iso, Lens, Prism, AffineTraversal, Traversal, and Setter.

    over : Optic -> (a -> b) -> s -> t

    """
    s = Star(compose(Identity, f), Identity)
    return Function(compose(Identity.run, Star.run(optic(s))))


def put(optic, b) -> Function:
    """Replaces all foci with a constant value, returning a function that updates the whole structure.

    put : Optic -> b -> s -> t

    """
    return over(optic, const(b))
