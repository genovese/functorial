"""IxLens and IxGetter — indexed read-write and read-only optics with one focus.

An IxLens i s t a b focuses on a single value of type a within a structure
of type s, while also producing an index of type i.  It can read (iview)
and write (iover, iput).  An IxGetter i s a is the read-only restriction.

Indexed optic functions dispatch on the profunctor type they receive:

  * IndexedForget  — reading actions (iview, icollect, ifold_map_of)
  * Indexed        — writing actions (iover) and composition with other indexed optics
  * plain (Forget, Star, etc.) — fallback for use via cast_as to a plain optic type

Index accumulation: the top-level action calls start with the sentinel _MISSING.
Each optic in a composition chain packs its own index into the accumulator via
_pack_index, producing a left-nested pair for a chain of length > 1:

    ilens_i @ ilens_j @ ilens_k  =>  index ((i, j), k)

"""

from __future__   import annotations

from typing       import Callable

from ..functions  import Function, identity

from .generics    import absurd
from .optic       import Optic, OpticIs, _MISSING, _pack_index
from .profunctors import IndexedForget, Indexed  # IndexedForget used in iview, not for dispatch in optic builders

__all__ = [
    'IxLens',
    'IxGetter',
    'ilens',
    'igetter',
    'iview',
    'iover',
    'iput',
    'selfIndex',
]


class IxLens(Optic, optic_is=OpticIs.IX_LENS):
    """An indexed optic with exactly one focus and an associated index."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.IX_LENS)


class IxGetter(Optic, optic_is=OpticIs.IX_GETTER):
    """A read-only indexed optic with exactly one focus and an associated index."""
    def __init__(self, f, opt_type=None):
        super().__init__(f, opt_type if opt_type is not None else OpticIs.IX_GETTER)


def ilens[I, S, T, A, B](
        get_with_index: Callable[[S], tuple[I, A]],
        setter: Callable[[S, B], T],
) -> IxLens:
    """Builds an IxLens from an indexed getter and a setter.

    get_with_index : s -> (i, a)   -- extracts the index and focus from s
    setter         : (s, b) -> t   -- rebuilds the structure from s and a new focus

    ilens : (s -> (i, a)) -> (s -> b -> t) -> IxLens i s t a b

    """
    plain_get = lambda s: get_with_index(s)[1]

    def the_lens(p):
        if isinstance(p, Indexed):
            # Writing/composition path: produce b from (accumulated index, a),
            # then rebuild structure via setter.
            f = Indexed.run(p)

            def write(i_acc, s):
                ix, a = get_with_index(s)
                return setter(s, f(_pack_index(i_acc, ix), a))

            return Indexed(write)

        if hasattr(p, 'ifold_vl'):
            # Reading path (IndexedForget, Forget): extract index and focus
            # via a single-element fold; the monoid is irrelevant for one element.
            return p.ifold_vl(lambda ig, _m: lambda s: ig(*(get_with_index(s))))

        # Plain profunctor (Star, etc.): act as a plain lens, forgetting the index.
        # Enables cast_as(LENS) and use with view, over, etc.
        p_first = p.into_first()
        return p_first.dimap(lambda s: (plain_get(s), s),
                             lambda bs: setter(bs[1], bs[0]))

    return IxLens(the_lens, OpticIs.IX_LENS)

def igetter[I, S, A](get_with_index: Callable[[S], tuple[I, A]]) -> IxGetter:
    """Builds an IxGetter from a function s -> (i, a).

    igetter : (s -> (i, a)) -> IxGetter i s a

    """
    def the_getter(p):
        if isinstance(p, Indexed):
            f = Indexed.run(p)

            def forward(i_acc, s):
                ix, a = get_with_index(s)
                return f(_pack_index(i_acc, ix), a)

            return Indexed(forward)

        if hasattr(p, 'ifold_vl'):
            return p.ifold_vl(lambda ig, _m: lambda s: ig(*(get_with_index(s))))

        # Plain profunctor: project out the value, forget the index.
        return p.dimap(lambda s: get_with_index(s)[1], absurd)

    return IxGetter(the_getter, OpticIs.IX_GETTER)


#
# Utility primitive: An indexed lens whose focus IS the whole structure
#
# selfIndex : IxLens s s s s
#

def _self_index_fn(p):
    if isinstance(p, Indexed):
        f = Indexed.run(p)
        return Indexed(lambda i_acc, s: f(_pack_index(i_acc, s), s))
    if hasattr(p, 'ifold_vl'):
        # selfIndex: focus IS the structure; index IS the structure.
        return p.ifold_vl(lambda ig, _m: lambda s: ig(s, s))
    # Plain profunctor: identity lens (no-op structurally)
    return p.dimap(identity, identity)


selfIndex = IxLens(_self_index_fn, OpticIs.IX_LENS)
selfIndex.__doc__ = """An indexed lens whose focus IS the whole structure, and whose index IS the structure itself.

Primarily useful as a prefix in composition:

  selfIndex @ some_lens   =>  IxLens s s a  (index = s, focus = view some_lens s)
  selfIndex @ some_fold   =>  IxFold  s s a  (index = s, focus = each element)

iview  (selfIndex) s          = (s, s)
iview  (selfIndex @ l) s      = (s, view l s)
iover  (selfIndex @ l) f s    = over l (f s) s

"""


#
# Indexed actions
#

def iview(optic) -> Function:
    """Gets the index and focus of an indexed lens on a structure.

    Parameters
    ----------
    * optic -- an IxGetter or IxLens

    Returns a function that takes a structure and returns an index, value pair.

    iview : IxGetter i s a -> s -> (i, a)

    """
    p: IndexedForget = IndexedForget(lambda i, a: (i, a))
    result: Callable = IndexedForget.run(optic(p))   # i_acc -> s -> (i, a)
    return Function(lambda s: result(_MISSING, s))

def iover(optic, f: Callable) -> Function:
    """Modifies the focus with a function that takes an index and the focus.

    Parameters
    ----------
    * optic -- an indexed lens
    * f -- a function that takes an index and a focus value and returns a
           transformed value

    Returns a function that maps the structure to an updated structure.

    iover : IxLens i s t a b -> (i -> a -> b) -> s -> t

    """
    p = Indexed(f)
    result: Callable = Indexed.run(optic(p))         # i_acc -> s -> t
    return Function(lambda s: result(_MISSING, s))

def iput(optic, b) -> Function:
    """Replaces the focus of an indexed lens with a constant value.

    Parameters
    ----------
    * optic -- an indexed lens

    Returns a function that takes a structure and returns the modified
    structure.

    iput : IxLens i s t a b -> b -> s -> t

    """
    return iover(optic, lambda _i, _a: b)
