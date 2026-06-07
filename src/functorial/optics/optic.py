"""Generic wrapper for all optics."""

from __future__   import annotations

import re

from enum         import StrEnum

from ..functions  import Function, compose

__all__ = ['Optic', 'OpticIs', 'OpticTypeError', 'composed_optic_is', 'cast_optic_is']

class OpticTypeError(Exception):
    "Exception that indicates incompatible optic types for cast or composition."

class OpticIs(StrEnum):
    "Optic types"
    ISO = "Iso"
    LENS = "Lens"
    PRISM = "Prism"
    SETTER = "Setter"
    GETTER = "Getter"
    REVIEW = "Review"
    AFFINE_TRAVERSAL = "Affine Traversal"
    TRAVERSAL = "Traversal"
    AFFINE_FOLD = "Affine Fold"
    FOLD = "Fold"
    # Indexed variants — carry an index type alongside the focus
    IX_LENS             = "IxLens"
    IX_PRISM            = "IxPrism"
    IX_AFFINE_TRAVERSAL = "IxAffineTraversal"
    IX_TRAVERSAL        = "IxTraversal"
    IX_GETTER           = "IxGetter"
    IX_AFFINE_FOLD      = "IxAffineFold"
    IX_FOLD             = "IxFold"
    IX_SETTER           = "IxSetter"

# Index accumulation sentinel and combinator for indexed optics.
#
# _MISSING marks "no index accumulated yet." The top-level iview/iover
# calls start with _MISSING, allowing the outermost indexed optic to
# return its index bare (not wrapped in a pair).
#
# _pack_index combines an incoming accumulated index with a new index
# produced by the current optic:
#   _pack_index(_MISSING, ix) = ix          -- first optic in chain
#   _pack_index(acc,      ix) = (acc, ix)   -- subsequent optics pair up
#
# Three composed optics i, j, k yield ((i, j), k) — left-nested,
# matching the evaluation order of compose(f1, f2, f3).
#
_MISSING = object()

def _pack_index(acc, ix):
    return ix if acc is _MISSING else (acc, ix)


class Optic(Function):
    """An optic is a first-class profunctor transformer p a b -> p s t.

    The _type field tracks which optic in the subtype lattice this is,
    allowing composition and casting to be checked and dispatched correctly.

    _class_map maps each OpticIs value to its corresponding subclass so
    that composition returns the correct Python type. Each subclass
    registers itself automatically via __init_subclass__ when its module
    is imported, using the optic_is keyword argument in the class header.

    Currently no index specification is included, though that is planned.
    We also allow a generic data map to allow appropriate operations,
    such as holding the Monoid or Applicative to use, but that is
    currently inconsistently handled and needs to be fleshed out.
    ATTN:TODOS
    """
    _class_map: dict = {}

    def __init_subclass__(cls, optic_is=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if optic_is is not None:
            Optic._class_map[optic_is] = cls

    def __init__(self, f, o_type: OpticIs, **data):
        self._type = o_type
        self._data = data
        super().__init__(f)

    def __str__(self):
        return f'{_optic_desc(self._type)} Optic {repr(self)}'

    def __matmul__(self, other):
        "Composes two optics, returning the correct subtype."
        # ATTN: We do not yet handle the **data argument generally.
        # This needs to be unified across all optics, assuming
        # that it even proves useful. The idea is to allow
        # extra specification info that cannot be inferred in Python,
        # like a Monoid or Applicative to use.
        if isinstance(other, Optic):
            opt_type = composed_optic_is(self._type, other._type)
            opt_data = self._data | other._data
            cls = Optic._class_map.get(opt_type, Optic)
            return cls(compose(self._fn, other._fn), opt_type, **opt_data)

        if callable(other):
            cls = Optic._class_map.get(self._type, Optic)
            return cls(compose(self._fn, other), self._type, **self._data)

        return NotImplemented

    def __rmatmul__(self, other):
        "Composes a plain callable on the left with this optic."
        if callable(other):
            cls = Optic._class_map.get(self._type, Optic)
            return cls(compose(other, self._fn), self._type, **self._data)
        return NotImplemented

    def cast_as(self, o_type: OpticIs):
        cls = Optic._class_map.get(o_type, Optic)
        return cls(self._fn, cast_optic_is(self._type, o_type), **self._data)

def _optic_desc(o_type: OpticIs, start_sentence=True) -> str:
    a = 'A' if start_sentence else 'a'
    n = ''
    if re.match(r'[AEIOU]', o_type.name, re.IGNORECASE):
        n = 'n'
    return f'{a}{n} {o_type.name}'

#
# Optic Composition: composed_optic_is(outer, inner) -> composed result type.
#
# Hasse diagram of the Optics subtype lattice with most specific at the top:
#
#                           Iso
#                          /   \
#                       Lens   Prism
#                      /   \   /   \
#                  Getter   AT    Review
#                      \   / \
#                      AF   Traversal
#                        \ /        \
#                        Fold      Setter
#
# where AT = AffineTraversal and AF = AffineFold.
#
# An edge in the lattice means ``can be used as.'' So, for instance,
# a Traversal can be used as an AffineTraversal or a Lens or an Iso,
# but a Lens cannot necessarily be used as a Traversal.
#
# This is encoded in the table _OPTIC_COMPOSITIONS below.
# To reading the table: outer ∘ inner means "apply outer optic to get an
# intermediate focus, then apply inner optic to reach the final focus".
# The result is the least specific (i.e., most general) type consistent with
# both optics.  Pairs not listed are incompatible and attempting such
# compositions will raise an OpticTypeError.
#

_OPTIC_COMPOSITIONS: dict[tuple[OpticIs, OpticIs], OpticIs] = {
    # ISO is the identity element of the composition lattice
    (OpticIs.ISO, OpticIs.ISO):                      OpticIs.ISO,
    (OpticIs.ISO, OpticIs.LENS):                     OpticIs.LENS,
    (OpticIs.ISO, OpticIs.PRISM):                    OpticIs.PRISM,
    (OpticIs.ISO, OpticIs.AFFINE_TRAVERSAL):         OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.ISO, OpticIs.TRAVERSAL):                OpticIs.TRAVERSAL,
    (OpticIs.ISO, OpticIs.AFFINE_FOLD):              OpticIs.AFFINE_FOLD,
    (OpticIs.ISO, OpticIs.FOLD):                     OpticIs.FOLD,
    (OpticIs.ISO, OpticIs.GETTER):                   OpticIs.GETTER,
    (OpticIs.ISO, OpticIs.REVIEW):                   OpticIs.REVIEW,
    (OpticIs.ISO, OpticIs.SETTER):                   OpticIs.SETTER,

    # LENS
    (OpticIs.LENS, OpticIs.ISO):                     OpticIs.LENS,
    (OpticIs.LENS, OpticIs.LENS):                    OpticIs.LENS,
    (OpticIs.LENS, OpticIs.PRISM):                   OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.LENS, OpticIs.AFFINE_TRAVERSAL):        OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.LENS, OpticIs.TRAVERSAL):               OpticIs.TRAVERSAL,
    (OpticIs.LENS, OpticIs.AFFINE_FOLD):             OpticIs.AFFINE_FOLD,
    (OpticIs.LENS, OpticIs.FOLD):                    OpticIs.FOLD,
    (OpticIs.LENS, OpticIs.GETTER):                  OpticIs.GETTER,
    (OpticIs.LENS, OpticIs.SETTER):                  OpticIs.SETTER,

    # PRISM
    (OpticIs.PRISM, OpticIs.ISO):                    OpticIs.PRISM,
    (OpticIs.PRISM, OpticIs.LENS):                   OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.PRISM, OpticIs.PRISM):                  OpticIs.PRISM,
    (OpticIs.PRISM, OpticIs.AFFINE_TRAVERSAL):       OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.PRISM, OpticIs.TRAVERSAL):              OpticIs.TRAVERSAL,
    (OpticIs.PRISM, OpticIs.AFFINE_FOLD):            OpticIs.AFFINE_FOLD,
    (OpticIs.PRISM, OpticIs.FOLD):                   OpticIs.FOLD,
    (OpticIs.PRISM, OpticIs.GETTER):                 OpticIs.AFFINE_FOLD,
    (OpticIs.PRISM, OpticIs.REVIEW):                 OpticIs.REVIEW,
    (OpticIs.PRISM, OpticIs.SETTER):                 OpticIs.SETTER,

    # AFFINE_TRAVERSAL
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.ISO):              OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.LENS):             OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.PRISM):            OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.AFFINE_TRAVERSAL): OpticIs.AFFINE_TRAVERSAL,
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.TRAVERSAL):        OpticIs.TRAVERSAL,
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.AFFINE_FOLD):      OpticIs.AFFINE_FOLD,
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.FOLD):             OpticIs.FOLD,
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.GETTER):           OpticIs.AFFINE_FOLD,
    (OpticIs.AFFINE_TRAVERSAL, OpticIs.SETTER):           OpticIs.SETTER,

    # TRAVERSAL
    (OpticIs.TRAVERSAL, OpticIs.ISO):                OpticIs.TRAVERSAL,
    (OpticIs.TRAVERSAL, OpticIs.LENS):               OpticIs.TRAVERSAL,
    (OpticIs.TRAVERSAL, OpticIs.PRISM):              OpticIs.TRAVERSAL,
    (OpticIs.TRAVERSAL, OpticIs.AFFINE_TRAVERSAL):   OpticIs.TRAVERSAL,
    (OpticIs.TRAVERSAL, OpticIs.TRAVERSAL):          OpticIs.TRAVERSAL,
    (OpticIs.TRAVERSAL, OpticIs.AFFINE_FOLD):        OpticIs.FOLD,
    (OpticIs.TRAVERSAL, OpticIs.FOLD):               OpticIs.FOLD,
    (OpticIs.TRAVERSAL, OpticIs.GETTER):             OpticIs.FOLD,
    (OpticIs.TRAVERSAL, OpticIs.SETTER):             OpticIs.SETTER,

    # GETTER (read-only, always has exactly one focus)
    (OpticIs.GETTER, OpticIs.ISO):                   OpticIs.GETTER,
    (OpticIs.GETTER, OpticIs.LENS):                  OpticIs.GETTER,
    (OpticIs.GETTER, OpticIs.PRISM):                 OpticIs.AFFINE_FOLD,
    (OpticIs.GETTER, OpticIs.AFFINE_TRAVERSAL):      OpticIs.AFFINE_FOLD,
    (OpticIs.GETTER, OpticIs.TRAVERSAL):             OpticIs.FOLD,
    (OpticIs.GETTER, OpticIs.GETTER):                OpticIs.GETTER,
    (OpticIs.GETTER, OpticIs.AFFINE_FOLD):           OpticIs.AFFINE_FOLD,
    (OpticIs.GETTER, OpticIs.FOLD):                  OpticIs.FOLD,

    # AFFINE_FOLD (read-only, zero or one focus)
    (OpticIs.AFFINE_FOLD, OpticIs.ISO):              OpticIs.AFFINE_FOLD,
    (OpticIs.AFFINE_FOLD, OpticIs.LENS):             OpticIs.AFFINE_FOLD,
    (OpticIs.AFFINE_FOLD, OpticIs.PRISM):            OpticIs.AFFINE_FOLD,
    (OpticIs.AFFINE_FOLD, OpticIs.AFFINE_TRAVERSAL): OpticIs.AFFINE_FOLD,
    (OpticIs.AFFINE_FOLD, OpticIs.TRAVERSAL):        OpticIs.FOLD,
    (OpticIs.AFFINE_FOLD, OpticIs.GETTER):           OpticIs.AFFINE_FOLD,
    (OpticIs.AFFINE_FOLD, OpticIs.AFFINE_FOLD):      OpticIs.AFFINE_FOLD,
    (OpticIs.AFFINE_FOLD, OpticIs.FOLD):             OpticIs.FOLD,

    # FOLD (read-only, zero or more foci)
    (OpticIs.FOLD, OpticIs.ISO):                     OpticIs.FOLD,
    (OpticIs.FOLD, OpticIs.LENS):                    OpticIs.FOLD,
    (OpticIs.FOLD, OpticIs.PRISM):                   OpticIs.FOLD,
    (OpticIs.FOLD, OpticIs.AFFINE_TRAVERSAL):        OpticIs.FOLD,
    (OpticIs.FOLD, OpticIs.TRAVERSAL):               OpticIs.FOLD,
    (OpticIs.FOLD, OpticIs.GETTER):                  OpticIs.FOLD,
    (OpticIs.FOLD, OpticIs.AFFINE_FOLD):             OpticIs.FOLD,
    (OpticIs.FOLD, OpticIs.FOLD):                    OpticIs.FOLD,

    # REVIEW (write-only constructor; only composes with optics that have review)
    (OpticIs.REVIEW, OpticIs.ISO):                   OpticIs.REVIEW,
    (OpticIs.REVIEW, OpticIs.PRISM):                 OpticIs.REVIEW,
    (OpticIs.REVIEW, OpticIs.REVIEW):                OpticIs.REVIEW,

    # SETTER (write-only modifier)
    (OpticIs.SETTER, OpticIs.ISO):                   OpticIs.SETTER,
    (OpticIs.SETTER, OpticIs.LENS):                  OpticIs.SETTER,
    (OpticIs.SETTER, OpticIs.PRISM):                 OpticIs.SETTER,
    (OpticIs.SETTER, OpticIs.AFFINE_TRAVERSAL):      OpticIs.SETTER,
    (OpticIs.SETTER, OpticIs.TRAVERSAL):             OpticIs.SETTER,
    (OpticIs.SETTER, OpticIs.SETTER):                OpticIs.SETTER,
}

# Mappings between plain and indexed optic types.
#
# ISO as indexed degrades to IX_LENS: an iso carries no index of its own,
# so when combined with an indexed optic the symmetric structure is lost.
# REVIEW has no indexed variant: write-only optics do not expose an index.
#
_PLAIN_TO_IX: dict[OpticIs, OpticIs] = {
    OpticIs.ISO:               OpticIs.IX_LENS,
    OpticIs.LENS:              OpticIs.IX_LENS,
    OpticIs.PRISM:             OpticIs.IX_PRISM,
    OpticIs.AFFINE_TRAVERSAL:  OpticIs.IX_AFFINE_TRAVERSAL,
    OpticIs.TRAVERSAL:         OpticIs.IX_TRAVERSAL,
    OpticIs.GETTER:            OpticIs.IX_GETTER,
    OpticIs.AFFINE_FOLD:       OpticIs.IX_AFFINE_FOLD,
    OpticIs.FOLD:              OpticIs.IX_FOLD,
    OpticIs.SETTER:            OpticIs.IX_SETTER,
}

_IX_TO_PLAIN: dict[OpticIs, OpticIs] = {
    OpticIs.IX_LENS:              OpticIs.LENS,
    OpticIs.IX_PRISM:             OpticIs.PRISM,
    OpticIs.IX_AFFINE_TRAVERSAL:  OpticIs.AFFINE_TRAVERSAL,
    OpticIs.IX_TRAVERSAL:         OpticIs.TRAVERSAL,
    OpticIs.IX_GETTER:            OpticIs.GETTER,
    OpticIs.IX_AFFINE_FOLD:       OpticIs.AFFINE_FOLD,
    OpticIs.IX_FOLD:              OpticIs.FOLD,
    OpticIs.IX_SETTER:            OpticIs.SETTER,
}


def composed_optic_is(opt1: OpticIs, opt2: OpticIs) -> OpticIs:
    """Returns the type of the optic formed by composing opt1 (outer) with opt2 (inner).

    For plain @ plain, uses the composition table directly.

    For any composition involving at least one indexed optic, the result is
    the indexed version of what the plain composition would give.  The index
    accumulates as a pair (I, J) when both sides carry one.

    Raises OpticTypeError for incompatible combinations (e.g. Getter ∘ Setter).

    """
    result = _OPTIC_COMPOSITIONS.get((opt1, opt2))
    if result is not None:
        return result

    # At least one side is indexed: compose the plain bases, then lift.
    base1 = _IX_TO_PLAIN.get(opt1, opt1)
    base2 = _IX_TO_PLAIN.get(opt2, opt2)
    if (opt1 in _IX_TO_PLAIN) or (opt2 in _IX_TO_PLAIN):
        base_result = _OPTIC_COMPOSITIONS.get((base1, base2))
        if base_result is not None:
            return _PLAIN_TO_IX.get(base_result, base_result)

    raise OpticTypeError(
        f'Cannot compose {_optic_desc(opt1, False)} (outer) with {_optic_desc(opt2, False)} (inner): '
        f'incompatible optic types'
    )


#
# Optic Casting:  cast_optic_is(from, to) -> valid cast type
#
# This is encoded in the subtype table _OPTIC_SUBTYPES below.
# _OPTIC_SUBTYPES[T] is the set of types that T can be cast to.
#
# A type T is a subtype of S when T is more specific (T can do
# everything S can). See the Hasse diagram above.
#

_OPTIC_SUBTYPES: dict[OpticIs, frozenset[OpticIs]] = {
    OpticIs.ISO: frozenset({
        OpticIs.ISO, OpticIs.LENS, OpticIs.PRISM,
        OpticIs.AFFINE_TRAVERSAL, OpticIs.TRAVERSAL,
        OpticIs.GETTER, OpticIs.AFFINE_FOLD, OpticIs.FOLD,
        OpticIs.REVIEW, OpticIs.SETTER,
    }),
    OpticIs.LENS: frozenset({
        OpticIs.LENS, OpticIs.AFFINE_TRAVERSAL, OpticIs.TRAVERSAL,
        OpticIs.GETTER, OpticIs.AFFINE_FOLD, OpticIs.FOLD, OpticIs.SETTER,
    }),
    OpticIs.PRISM: frozenset({
        OpticIs.PRISM, OpticIs.AFFINE_TRAVERSAL, OpticIs.TRAVERSAL,
        OpticIs.AFFINE_FOLD, OpticIs.FOLD, OpticIs.REVIEW, OpticIs.SETTER,
    }),
    OpticIs.AFFINE_TRAVERSAL: frozenset({
        OpticIs.AFFINE_TRAVERSAL, OpticIs.TRAVERSAL,
        OpticIs.AFFINE_FOLD, OpticIs.FOLD, OpticIs.SETTER,
    }),
    OpticIs.TRAVERSAL: frozenset({
        OpticIs.TRAVERSAL, OpticIs.FOLD, OpticIs.SETTER,
    }),
    OpticIs.GETTER: frozenset({
        OpticIs.GETTER, OpticIs.AFFINE_FOLD, OpticIs.FOLD,
    }),
    OpticIs.AFFINE_FOLD: frozenset({
        OpticIs.AFFINE_FOLD, OpticIs.FOLD,
    }),
    OpticIs.FOLD:   frozenset({OpticIs.FOLD}),
    OpticIs.REVIEW: frozenset({OpticIs.REVIEW}),
    OpticIs.SETTER: frozenset({OpticIs.SETTER}),

    # Indexed variants: each IX_X can be cast to indexed subtypes
    # and also to the corresponding plain subtypes (forgetting the index).
    OpticIs.IX_LENS: frozenset({
        OpticIs.IX_LENS, OpticIs.IX_AFFINE_TRAVERSAL, OpticIs.IX_TRAVERSAL,
        OpticIs.IX_GETTER, OpticIs.IX_AFFINE_FOLD, OpticIs.IX_FOLD, OpticIs.IX_SETTER,
        OpticIs.LENS, OpticIs.AFFINE_TRAVERSAL, OpticIs.TRAVERSAL,
        OpticIs.GETTER, OpticIs.AFFINE_FOLD, OpticIs.FOLD, OpticIs.SETTER,
    }),
    OpticIs.IX_PRISM: frozenset({
        OpticIs.IX_PRISM, OpticIs.IX_AFFINE_TRAVERSAL, OpticIs.IX_TRAVERSAL,
        OpticIs.IX_AFFINE_FOLD, OpticIs.IX_FOLD, OpticIs.IX_SETTER,
        OpticIs.PRISM, OpticIs.AFFINE_TRAVERSAL, OpticIs.TRAVERSAL,
        OpticIs.AFFINE_FOLD, OpticIs.FOLD, OpticIs.SETTER,
    }),
    OpticIs.IX_AFFINE_TRAVERSAL: frozenset({
        OpticIs.IX_AFFINE_TRAVERSAL, OpticIs.IX_TRAVERSAL,
        OpticIs.IX_AFFINE_FOLD, OpticIs.IX_FOLD, OpticIs.IX_SETTER,
        OpticIs.AFFINE_TRAVERSAL, OpticIs.TRAVERSAL,
        OpticIs.AFFINE_FOLD, OpticIs.FOLD, OpticIs.SETTER,
    }),
    OpticIs.IX_TRAVERSAL: frozenset({
        OpticIs.IX_TRAVERSAL, OpticIs.IX_FOLD, OpticIs.IX_SETTER,
        OpticIs.TRAVERSAL, OpticIs.FOLD, OpticIs.SETTER,
    }),
    OpticIs.IX_GETTER: frozenset({
        OpticIs.IX_GETTER, OpticIs.IX_AFFINE_FOLD, OpticIs.IX_FOLD,
        OpticIs.GETTER, OpticIs.AFFINE_FOLD, OpticIs.FOLD,
    }),
    OpticIs.IX_AFFINE_FOLD: frozenset({
        OpticIs.IX_AFFINE_FOLD, OpticIs.IX_FOLD,
        OpticIs.AFFINE_FOLD, OpticIs.FOLD,
    }),
    OpticIs.IX_FOLD: frozenset({
        OpticIs.IX_FOLD,
        OpticIs.FOLD,
    }),
    OpticIs.IX_SETTER: frozenset({
        OpticIs.IX_SETTER,
        OpticIs.SETTER,
    }),
}

def cast_optic_is(opt_from: OpticIs, opt_to: OpticIs) -> OpticIs:
    """Checks that opt_from can be used as opt_to, or raises OpticTypeError if not.

    A cast is valid when opt_from is a subtype of opt_to — i.e., when
    opt_from is at least as specific as opt_to in the optic lattice.

    Returns opt_to on success.

    """
    if opt_to not in _OPTIC_SUBTYPES.get(opt_from, frozenset()):
        raise OpticTypeError(f'Cannot cast {_optic_desc(opt_from, False)} as {_optic_desc(opt_to, False)}')
    return opt_to
