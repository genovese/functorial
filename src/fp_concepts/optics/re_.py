"""Re reverses a profunctor's type parameters, dualising an optic.

newtype Re p s t a b = Re { runRe :: p b a -> p t s }

Instances:
  Profunctor p  => Profunctor  (Re p s t)   dimap f g (Re h) = Re (h . dimap g f)
  Choice p      => Cochoice    (Re p s t)   unleft  (Re h)   = Re (h . into_left)
  Cochoice p    => Choice      (Re p s t)   into_left (Re h) = Re (h . unleft)
  Strong p      => Costrong    (Re p s t)   unfirst  (Re h)  = Re (h . into_first)
  Costrong p    => Strong      (Re p s t)   into_first (Re h)= Re (h . unfirst)

Primary use: the `re` function reverses an optic by passing Re(identity)
through it, accumulating the dual transformations via these instances.

"""

from __future__  import annotations

from ..functions import Function, compose, identity

from .choice     import Choice
from .cochoice   import Cochoice
from .costrong   import Costrong
from .strong     import Strong

__all__ = ['Re', 're']


class Re[A, B](Strong, Costrong, Choice, Cochoice):
    """Profunctor transformer that reverses type parameters.

    newtype Re p s t a b = Re { runRe :: p b a -> p t s }

    Re wraps a function (p b a -> p t s) and makes it act as a profunctor
    in (a, b) by delegating each method to the *dual* method on the inner
    profunctor.  The duality table is:

        Re method        inner p method called
        ---------        --------------------
        dimap f g        dimap g f       (arguments swapped)
        into_left        unleft
        into_right       unright
        unleft           into_left
        unright          into_right
        into_first       unfirst
        into_second      unsecond
        unfirst          into_first
        unsecond         into_second

    The constraints on the inner p are only checked at runtime: if the
    wrapped profunctor does not support the required dual method a
    standard AttributeError is raised.

    Use Re.run(r) to extract the wrapped function p b a -> p t s.
    Use the module-level `re` function to reverse an optic.

    """
    def __init__(self, f):
        self._f = Function(f)
        super().__init__()

    @classmethod
    def run(cls, r: Re):
        """Extracts the wrapped function."""
        return r._f     # pylint: disable=protected-access

    # --- Profunctor ---
    # dimap f g (Re h) = Re (h . dimap g f)   -- note: g and f are swapped

    def dimap(self, f, g):
        return Re(compose(self._f, lambda q: q.dimap(g, f)))

    # --- Cochoice (inner p must have Choice) ---
    # unleft  (Re h) = Re (h . into_left)
    # unright (Re h) = Re (h . into_right)

    def unleft(self):
        return Re(compose(self._f, lambda q: q.into_left()))

    def unright(self):
        return Re(compose(self._f, lambda q: q.into_right()))

    # --- Choice (inner p must have Cochoice) ---
    # into_left  (Re h) = Re (h . unleft)
    # into_right (Re h) = Re (h . unright)

    def into_left(self):
        return Re(compose(self._f, lambda q: q.unleft()))

    def into_right(self):
        return Re(compose(self._f, lambda q: q.unright()))

    # --- Costrong (inner p must have Strong) ---
    # unfirst  (Re h) = Re (h . into_first)
    # unsecond (Re h) = Re (h . into_second)

    def unfirst(self):
        return Re(compose(self._f, lambda q: q.into_first()))

    def unsecond(self):
        return Re(compose(self._f, lambda q: q.into_second()))

    # --- Strong (inner p must have Costrong) ---
    # into_first  (Re h) = Re (h . unfirst)
    # into_second (Re h) = Re (h . unsecond)

    def into_first(self):
        return Re(compose(self._f, lambda q: q.unfirst()))

    def into_second(self):
        return Re(compose(self._f, lambda q: q.unsecond()))


def re(optic):
    """Reverses an optic by running it with Re(identity).

    re : Optic (Re p) s t a b -> Optic p b a t s

    Passes Re(identity) into the optic. Each profunctor method the optic
    calls on the Re instance composes the dual method onto the identity
    function, building up the reversed transformation p b a -> p t s.

    Typical uses:
      re(prism)  -- run a Prism as a Review (or Getter over the tag)
      re(iso)    -- reverse an Iso
      re(review) -- extract the underlying builder as a Getter

    """
    return Re.run(optic(Re(identity)))
