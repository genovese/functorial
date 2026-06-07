# ruff: noqa: N802, N806

from __future__ import annotations

from collections.abc import Callable

from .const       import make_const, run_const, type_const
from .monoids     import Monoid
from .traversable import Traversable, traverse
from .functions   import compose as c
from .utils       import Collect

__all__ = ['fold_map_default',]

#
# fold_map_default : (Monoid m, Traversable t) => (a -> m) -> t a -> m
#
# We need to specify the Monoid (defaults to Collect) because Python
# cannot infer it automatically.
#
# In TL1, this has a simple expression
#
#   fold_map_default f = run_const . traverse (Const . f)
#

def fold_map_default(f: Callable, t: Traversable, m: Monoid = Collect):
    """An implementation of foldMap for a generic traversable.

    fold_map_default : (Traversable t, Monoid m) => (a -> m) -> t a -> m

    Parameters:

    + f :: Function producing a monoidal value for each element of the
      traversable collection.
    + t :: A traversable collection
    + m :: The monoid with which to interpret the values. Defaults
      to collecting the values in a list (Collect).

    Returns the monoidal value produced by transforming and folding over
    the traversable container.

    """
    C = type_const(m)  # Give access to pure and monoid, which is all we need in traverse
    return run_const(traverse(c(make_const(m), f), t, C))
