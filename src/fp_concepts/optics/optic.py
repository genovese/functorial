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

class Optic(Function):
    """Class representing the generic family of optics.

    This is a wrapper for all the specific optic subtypes that
    enables unified handling (e.g., composition, casting) and querying
    (e.g., types or descriptive label).

    Currently no index specification is included, though that is planned.
    We also allow a generic data map to allow appropriate operations,
    such as holding the Monoid or Applicative to use, but that is
    currently inconsistently handled and needs to be fleshed out.
    ATTN:TODOS

    """
    # ATTN: Add index type here as optional argument with NoIx = Unit
    def __init__(self, f, o_type: OpticIs, **data):
        self._type = o_type
        self._data = data    # ATTN: needed? how used? e.g., Monoid to use etc.
        super().__init__(f)

    def __str__(self):
        return f'{_optic_desc(self._type)} Optic {repr(self)}'

    # ATTN: add data accessor, e.g., to wrap with the right Monoid

    def __matmul__(self, other):
        "Composes two optics."
        if isinstance(other, Optic):
            opt_type = composed_optic_is(self._type, other._type)
            opt_data = self._data | other._data

            # ATTN: Need to standardize the args to the non-trivial optic classes
            # We will have them take **data in second argument but not a type
            # and instead of .__class__ get the class from the composition
            # return self.__class__(compose(self._fn, other._fn), opt_type, **opt_data)
            return self.__class__(compose(self._fn, other._fn), opt_type)

        if callable(other):  # Viable case?
            return self.__class__(compose(self._fn, other), self._type, **self._data)

        return NotImplemented

    def __rmatmul__(self, other):
        "Composes two optics."
        if callable(other):  # Viable case?
            return self.__class__(compose(other, self._fn), self._type, **self._data)
        return NotImplemented

    def cast_as(self, o_type: OpticIs):
        return Optic(self._fn, cast_optic_is(self._type, o_type), **self._data)

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

def composed_optic_is(opt1: OpticIs, opt2: OpticIs) -> OpticIs:
    """Returns the type of the optic formed by composing opt1 (outer) with opt2 (inner).

    Raises OpticTypeError for incompatible combinations (e.g. Getter ∘ Setter).

    """
    result = _OPTIC_COMPOSITIONS.get((opt1, opt2))
    if result is None:
        raise OpticTypeError(
            f'Cannot compose {_optic_desc(opt1, False)} (outer) with {_optic_desc(opt2, False)} (inner): '
            f'incompatible optic types'
        )
    return result


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
