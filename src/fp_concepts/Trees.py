# ruff: noqa: N801, N806, E731, EM102
# pylint: disable=protected-access, invalid-name

""" Various forms of trees implementing the relevant protocols

The Tree types implemented here include:

1. Rose Trees - data in the nodes, arbitrary number of children

   data RoseTree a = Node a (List (RoseTree a))

2. (Tipped) Binary Trees - data in the branch nodes not at the leaves
   This is the default form and allows empty trees.

   data BinaryTree a = Tip | Node (BinaryTree a) a (BinaryTree a)

3. Leafy Binary Trees - data at the leaves but not at the branch nodes

   data LeafyBinaryTree a = Leaf a | Branch (LeafyBinaryTree a) (LeafyBinaryTree a)

4. Heterogeneous Binary Trees - data at the leaves and branches of different types

   data HetBinaryTree b a = Leaf a
                          | Unary b (HetBinaryTree b a)
                          | Binary (HetBinaryTree b a) b (HetBinaryTree b a)

5. Tries (aka Prefix Trees)

   data Trie = record Trie whee
       key        : Maybe a
       children   : Map k (Trie k m a)
       annotation : m   -- an associated value or monoidal annotation

6. Tulip Trees - like a Rose tree but with different types on leaf and branches

   data TulipTree b l = Leaf l | Branch b (NonEmptyList HTree b l)

All of these tree types implement appropriate traits such as Functor,
Applicative, Foldable, Traversable, and their indexed counterparts.
They also all support the creation of zippers for navigating and
functionally modifying them.


"""

from __future__ import annotations

from abc             import ABC, abstractmethod
from collections.abc import Callable
from functools       import partial
from typing          import TYPE_CHECKING, TypeGuard, cast

from .Applicative import Applicative, map2, ap
from .Either      import Either, Left, Right
from .Functor     import map, imap  # pylint: disable=redefined-builtin
from .List        import List
from .Monoids     import Monoid
from .Traversable import traverse, itraverse
from .singleton   import ContingentSingletonFromABC, FinalAttribute

if TYPE_CHECKING:
    from .Functor import IndexedFunctor

__all__ = [
    'BinaryTree', 'Tip', 'BinaryTreeType',
    'RoseTree', 'LeafyBinaryTree',
    'SExp', 'NoSExp',
    'is_binary_tree', 'binary_tree', 'complete_btree',
]


#
# Helpers
#

class NoSExp:
    "A marker class to prevent s-expression conversion in recursive lists."

    __match_args__ = ('_value',)

    def __init__(self, value):
        self._value = value

    @property
    def unwrapped(self):
        "Unwraps and retuns the underlying value that has been marked NoSExp."
        return self._value

class SExp(list):
    "A Marker class to distinguish lists as values from lists as s-expressions."
    def __init__(self, xs):
        super().__init__(xs)

    @classmethod
    def of(cls, *xs):
        "Converts zero or more arguments into a SExp list."
        return cls(list(xs))

    @classmethod
    def recurse(cls, lst):
        """Recursively an s-expression list to SExp lists at every level.

        Values should be wrapped in NoSExp to be excluded from this conversion.

        As an example

            SExp.recurse([4, [(10, 20), [30, NoSExp([40, 50])]]])

        converts all lists (or subclasses) -- but not tuples or other
        sequences -- to be SExp lists except for the marked [40, 50],
        which is itself in the returned list.

        """
        def convert(xs):
            match xs:
                case list() if isinstance(xs, list):
                    return cls.recurse(xs)
                case NoSExp(value):
                    return value
                case _:
                    return xs

        return SExp(map(convert, lst))


#
# Common Structure
#

class TreeLike(ABC):
    """Common structure for all tree classes.

    This allows identifying node values and subtrees
    and provids a basis for a general traversal method.

    """

    @abstractmethod
    def node_value(self):  # -> Maybe[A]
        "Returns Some(val) if val is the data in the root node of a subtree, else Nothing()."
        ...

    @abstractmethod
    def subtrees(self):
        "Returns a List of subtrees for the root node of this tree."
        ...

    @staticmethod
    @abstractmethod
    def tree_traverse(effect: type[Applicative], node_fn: Callable, subtree_fn: Callable, tree):
        "Generic tree traversal, applying node_fn at nodes and subtree_fn at subtree branches."
        ...


#
# Rose Trees
#
# data RoseTree a = Node a (List (RoseTree a))
#

class RoseTree[A](Applicative):
    """A Rose Tree which holds data and an arbitrary number of children in each node.

    These have type
        data RoseTree a = Node a (List (RoseTree a))

    """
    def __init__(self, sexp: list):
        "Creates a Rose Tree from s-expression input. See `to_sexp`."
        if len(sexp) == 0:
            raise ValueError('Rose tree requires a nonempty root node.')

        val, *children = sexp
        self._value = val
        self._subtrees: List = List(RoseTree(child) for child in children)  # Note: mypy forces hint

    def to_sexp(self):
        """Converts a rose tree to s-expression format.

        A tree is represented by a list [data, children], where
        children are sub-trees specified by lists in s-expression
        format. Leaf nodes are thus given by singleton lists
        [v] with data v.

        Example: [1, [2, [3], [4], [5]], [6, [7, [8, [9], [10]]]]],

        """
        return List.of(
            self._value,
            *[child.to_sexp() for child in self._subtrees]
        )

    def as_str(self, levels=None):
        "Returns a simple string representation of this tree"
        if levels is None:
            levels = []
        indent = ''.join('\u2502  ' if level == 0 else '   ' for level in levels)
        lead_a = '\u251c\u2500 '
        lead_r = '\u2514\u2500 '
        str_form = [f'{self._value}\n']
        n = len(self._subtrees)
        for index, child in enumerate(self._subtrees):
            if index < n - 1:
                str_form.append(f'{indent}{lead_a}{child.as_str([*levels, 0])}')
            else:
                str_form.append(f'{indent}{lead_r}{child.as_str([*levels, 1])}')
        return "".join(str_form)

    def __str__(self):
        return self.as_str().strip()

    @classmethod
    def make(cls, data: A, subtrees: list) -> RoseTree[A]:
        "Creates a Rose tree from data a list of subtrees."
        t = cls([data])
        t._subtrees = List(subtrees)
        return t

    @classmethod
    def unfold[B](cls, gen: Callable[[B], tuple[A, list[B]]], seed: B) -> RoseTree[A]:
        "Creates a rose tree by repeatedly unfolding a generating function from a starting seed."
        def unfold_sexp(s):
            a, seeds = gen(s)

            if len(seeds) == 0:
                return [a]

            return [a, *[unfold_sexp(seed) for seed in seeds]]

        return RoseTree(unfold_sexp(seed))

    def foldTree[B](self, f: Callable[[A, list[B]], B]) -> B:
        """ Folds the tree into a summary value, in inorder sequence.

        This has type Tree a -> (a -> [b] -> b) -> Tree b and is the
        formal dual of the RoseTree.unfold method.

        See also the fold, foldRight, and foldMap methods.

        """
        def go(t):
            return f(t._value, map(go, t._subtrees))
        return go(self)

    def foldM[M](self, f: Callable[[A], M], monoid: Monoid) -> M:
        """Reduces a tree to a monoidal value with a monoidal value for each node.

        We could use the default foldMap for traversables here, but this
        is illustrative and cleaner.

        """
        def rf(a: A, mvals: list[M]) -> M:
            return List(mvals).fold(monoid.mcombine, f(a))

        return self.foldTree(rf)

    def fold[B](self, f: Callable[[B, A], B], initial: B) -> B:
        """Left fold of the tree into a summary value, in preorder sequence.

        """
        def go(init, t):
            return t.fold(go, f(init, t._value))

        return go(initial, self)

    def foldRight[B](self, f: Callable[[A, B], B], initial: B) -> B:
        """Right fold of the tree into a single value, in postorder sequence.

        This has type Tree a -> (a -> b -> b) -> Tree b.

        """
        def go(t, init):
            return f(t._value, t._subtrees.foldRight(go, init))

        return go(self, initial)

    def map[B](self, g: Callable[[A], B]):
        "Functor instance that maps a function over this rose tree."
        tree: RoseTree = RoseTree([g(self._value)])  # Note: mypy forces annotation here??
        tree._subtrees = List(map(g, child) for child in self._subtrees)
        return tree

    def imap[I, B](self, g: Callable[[I, A], B]):
        "Maps an indexed function over this tree, returning a new tree."
        def go(index, tree):
            t = RoseTree([g(index, tree._value)])
            t._subtrees = imap(lambda j, s: go(index + List.of(j), s), tree._subtrees)
            return t
        return go(List(), self)

    @classmethod
    def pure(cls, a):
        return RoseTree([a])

    def map2[B, C](self, g: Callable[[A, B], C], fb: RoseTree[B]) -> RoseTree[C]:
        new_tree: RoseTree = RoseTree([g(self._value, fb._value)])

        h2 = lambda sub_b: map(partial(g, self._value), sub_b)
        h1 = lambda sub_a: map2(g, sub_a, fb)

        ab_cs = map(h2, fb._subtrees)
        ab_cs.extend(map(h1, self._subtrees))
        new_tree._subtrees = ab_cs

        return new_tree

    def traverse(self, f: type[Applicative], g: Callable[[A], Applicative]) -> Applicative:
        "Applies an effectful function (a -> f b) to each node, collecting results as a same-shaped tree."
        def go(t):
            return map2(RoseTree.make, g(t._value), traverse(go, t._subtrees, f))

        return go(self)

    def itraverse[I](self, f: type[Applicative], g: Callable[[I, A], Applicative]) -> Applicative:  # g : a -> f b
        "Like traverse, but the effectful function (g : i -> a -> f b) takes a node index and value."
        def go(index, t):
            return map2(RoseTree.make, g(index, t._value),
                        itraverse(lambda i, s: go(index + List.of(i), s), t._subtrees, f))

        return go(List(), self)


#
# Binary Trees in varied forms
#
# All the Binary Tree variants inherit from this abstract base class
#
# The empty BinaryTree class can be accessed through BinaryTree.Empty,
# e.g., for pattern matching.  Tip is the singleton value of the
# empty tree. For types, one needs to use BinaryTreeType[A] to comprise
# both non-empty and empty binary trees. This is annoying but seems
# like the best of several bad options.
#
# The concrete Binary Trees implement the IndexedFunctor protocol.
#

class AbstractBinaryTree(ABC, metaclass=ContingentSingletonFromABC):
    "Abstract base class for all binary tree variants."

    @abstractmethod
    def to_sexp(self):
        "Converts a binary tree to an s-expression format of nested lists."
        ...

    @abstractmethod
    def as_str(self, levels=None):
        "Returns a pleasant, human-readable string representation of the tree."
        ...

class EmptyBinaryTree[A](AbstractBinaryTree, metaclass=ContingentSingletonFromABC):
    """A look-alike representing an empty Binary Tree, for any value type.

    This is an immutable object with no data and attempting to
    change its attributes will lead to an error. It is also a singleton
    class, so all instances will be shared.

    Fur users, this common instance will be called Tip. It can be
    pattern matched with BinaryTree.Empty() in a match statement.
    For most type signatures, use AbstractBinaryTree, but if this is
    needed in particular (hard to see why it would be), then it will
    have to be imported explicitly.

    """
    __match_args__ = ()

    IS_SINGLETON = FinalAttribute(True)

    @classmethod
    def unfold[B](
            cls,
            gen: Callable[[B], tuple[A, B | EmptyBinaryTree[A] | None, B | EmptyBinaryTree[A] | None]],
            seed: B
    ) -> BinaryTree[A]:
        "Creates a binary tree from an arbitrary seed and a function that takes a seed."
        return BinaryTree.unfold(gen, seed)

    def __bool__(self):
        return False

    def to_sexp(self):
        return self  # [] ??

    def __str__(self):
        return 'Tip'

    def __repr__(self):
        return str(self)

    def __setattr__(self, name, value):
        raise AttributeError('An empty binary tree cannot be modified.')

    def as_str(self, _levels=None):
        return str(self)

    def map[B](self, _g: Callable[[A], B]):
        """Mapping on an empty tree just gives an empty tree."""
        return self

    def imap[I, B](self, _g: Callable[[I, A], B]):
        """Mapping on an empty tree just gives an empty tree."""
        return self

    def traverse(self, f: type[Applicative], _g: Callable[[A], Applicative]) -> Applicative:  # g : a -> f b
        "Applies an effectful function (a -> f b) to each node, collecting results as a same-shaped tree."
        return f.pure(self)

Tip: EmptyBinaryTree = EmptyBinaryTree()

#
# Tipped Binary Trees, which we take as a default form
#
# data BinaryTree a = Tip | Node (BinaryTree a) a (BinaryTree a)
#

class BinaryTree[A](AbstractBinaryTree, metaclass=ContingentSingletonFromABC):
    """Default (Tipped) Binary Trees with data in nodes but not at leaves.

    This type is described by

        data BinaryTree a = Tip | Node (BinaryTree a) a (BinaryTree a)

    """
    __match_args__ = ('_left', '_value', '_right')

    def __init__(self, sexp: list | tuple):
        "Creates a Binary Tree from s-expression input. See `to_sexp`."
        if len(sexp) == 0:  # Handle empty case elsewhere
            raise ValueError('Binary tree requires a nonempty root node, use binary_tree() for this case.')

        val, left, right, *_ = sexp
        self._value = val
        self._left: BinaryTree[A] | EmptyBinaryTree[A] = BinaryTree(List(left)) if left else Tip
        self._right: BinaryTree[A] | EmptyBinaryTree[A] = BinaryTree(List(right)) if right else Tip

        super().__init__()

    Empty = EmptyBinaryTree  # Accessible to allow matching; construction just returns Tip.

    @classmethod
    def make(
            cls,
            data: A,
            left: BinaryTree[A] | EmptyBinaryTree[A],
            right: BinaryTree[A] | EmptyBinaryTree[A]
    ) -> BinaryTree[A]:
        "Creates and returns a binary tree with specified data and children."
        t = cls([data, Tip, Tip])
        t._left = left
        t._right = right
        return t

    @classmethod
    def is_leaf(cls, tree: BinaryTree[A]) -> bool:
        "Is the root of this tree a leaf node (not Tip)?"
        return tree != Tip and tree._left == Tip and tree._right == Tip

    def to_sexp(self):
        """Converts a binary tree to s-expression format.

        A tree is represented by a list or tuple of the form
            [data, left, right] or (data, left, right),
        where left and right are either Tip for empty subtrees
        or lists/tuples in s-expression format for non-empty
        subtrees.

        Example: [1,
                  [2, [4, Tip, Tip], [5, Tip, Tip]],
                  [3, [6, Tip, Tip], Tip]]

        When created from input, as in the BinaryTree constructor,
        any falsy value can stand in for Tip.

        This returns the s-expression as a list rather than a tuple
        so that the result can be modified.

        """
        return SExp([
            self._value,
            self._left and self._left.to_sexp(),
            self._right and self._right.to_sexp()
        ])

    def as_str(self, levels=None):
        "Returns a simple string representation of this tree"
        if levels is None:
            levels = []
        lead_r = '\u251c\u2500 '
        lead_l = '\u2514\u2500 '
        root = f'{self._value}\n'
        if self._left or self._right:
            indent = ''.join('\u2502  ' if level == 1 else '   ' for level in levels)
            left = f'{indent}{lead_l}{self._left.as_str([*levels, 0]) if self._left else "\u25a1\n"}'
            right = f'{indent}{lead_r}{self._right.as_str([*levels, 1]) if self._right else "\u25a1\n"}'
        else:
            left = right = ''

        # Put the left subtrees on the bottom so the tree is rotationally consistent
        return root + right + left

    def __str__(self):
        return self.as_str().strip()

    @property
    def contents(self) -> tuple[BinaryTree[A] | EmptyBinaryTree[A], A, BinaryTree[A] | EmptyBinaryTree[A]]:
        """Extract the raw contents from the root node of this tree.

        Returns a tuple of subtrees for a branch node, where the
        subtrees may equal Tip.

        """
        # We know self != Tip here
        return (self._left, self._value, self._right)

    @classmethod
    def unfold[B](
            cls,
            gen: Callable[[B], tuple[A, B | EmptyBinaryTree[A] | None, B | EmptyBinaryTree[A] | None]],
            seed: B
    ) -> BinaryTree[A]:
        "Creates a binary tree by repeatedly unfolding a generating function from a starting seed."
        def unfold_sexp(s):
            if s is None or s is Tip:
                return Tip

            a, seed_l, seed_r = gen(s)
            return [a, unfold_sexp(seed_l), unfold_sexp(seed_r)]

        return BinaryTree(unfold_sexp(seed))

    def map[B](self, g: Callable[[A], B]) -> BinaryTree[B]:
        "Maps a function over this binary tree, returning a new tree."
        tree: BinaryTree[B] = BinaryTree([g(self._value), Tip, Tip])
        tree._left = map(g, self._left) if self._left else Tip        # type: ignore  # _left: BinaryTree[A] if not Tip
        tree._right = map(g, self._right) if self._right else Tip     # type: ignore  # _right: BinaryTree[A] if not Tip
        return tree

    def imap[I, B](self, g: Callable[[I, A], B]):
        "Maps an indexed function over this binary tree, returning a new tree."
        def go(index, tree):
            t = BinaryTree([g(index, tree._value), Tip, Tip])
            t._left = go(index + List.of(0), tree._left) if tree._left else Tip
            t._right = go(index + List.of(1), tree._right) if tree._right else Tip
            return t
        return go(List(), self)

    @staticmethod
    def _bt_traverse(effect, node_f, subtree_f, t):
        fl = subtree_f(t._left) if t._left else effect.pure(Tip)
        fa = node_f(t._value)
        fr = subtree_f(t._right) if t._right else effect.pure(Tip)
        return ap(ap(BinaryTree.make, fa, fl), fr)

    def traverse(self, f: type[Applicative], g: Callable[[A], Applicative]) -> Applicative:
        "Applies an effectful function (a -> f b) to each node, collecting results as a same-shaped tree."
        def inorder(tree):
            return self._bt_traverse(f, g, inorder, tree)
        return inorder(self)

type BinaryTreeType[A] = BinaryTree[A] | EmptyBinaryTree[A]

def is_binary_tree(t) -> TypeGuard[BinaryTreeType]:   # Duck typing here for type inference
    "Tests if object is a Binary Tree."
    return t == Tip or isinstance(t, AbstractBinaryTree)

def binary_tree(spec=None, left=Tip, right=Tip, *, seed=None, sexp=None):
    """Smart binary tree constructor.

    Accepts a variety of argument configurations:

    + binary_tree() -- gives the empty BinaryTree
    + binary_tree(bt) -- for bt : BinaryTree, returns bt as is
    + binary_tree(sexp=[...]) -- builds tree from an S-expression
    + binary_tree(SExp([...])) -- builds tree from an S-expression
    + binary_tree(f, seed=x) -- unfolds tree with f and starting seed x
    + binary_tree(data, left, right) -- returns BinaryTree data left right.

    A tree can be specified from an S-expression using either the
    sexp= keyword (alone) or by wrapping the S-expression for the
    tree in SExp(). This extra work is needed to distinguish from
    trees with tuple or list data types. Note that
    BinaryTree.to_sexp returns a SExp object and so can be used
    directly. Also note that BinaryTree accepts an sexp without
    wrapper.

    """
    if not spec and not sexp:
        return EmptyBinaryTree()

    if is_binary_tree(spec):
        return spec   # ATTN: deep copy?

    if callable(spec):
        return BinaryTree.unfold(spec, seed)

    if not spec and sexp:
        return BinaryTree(sexp)

    if isinstance(spec, SExp):
        return BinaryTree(spec)

    if isinstance(spec, str):  # Named trees, params in left, right, seed
        if 'complete'.startswith(spec.lower()):
            return complete_btree(left if left else 0)
        raise KeyError(f'Unrecognized named binary tree {spec}')

    return BinaryTree.make(spec, left, right)

def complete_btree(depth: int) -> BinaryTree[int]:
    "Returns a complete binary tree of given depth with integer data."
    def generate(k):
        if k < 2 ** depth - 1:
            return (k, 2 * k + 1, 2 * k + 2)
        return (k, Tip, Tip)

    return BinaryTree.unfold(generate, 0)


#
# Leafy Binary Trees - binary trees with values only in the leaves
#
# data LeafyBinaryTree a = Leaf a | Branch (LeafyBinaryTree a) (LeafyBinaryTree a)
#

class LeafyBinaryTree[A](AbstractBinaryTree):
    """Binary trees with values only in the leaves.

    These have type
        data LeafyBinaryTree a = Leaf a | Branch (LeafyBinaryTree a) (LeafyBinaryTree a)

    """
    def __init__(self, sexp: SExp | A):
        """Creates a Leafy Binary Tree from s-expression input. See `to_sexp`.

        Currently requires all lists to be wrapped in SExp to distinguish
        branches from leaves. This will need to be changed.

        """
        # Store node as Either[A, (LeafyBinaryTree[A], LeafyBinaryTree[A])]
        # with Left for Leaf and Right for Branch
        if isinstance(sexp, SExp):
            if len(sexp) == 0:       # type: ignore
                raise ValueError('Leafy Binary tree cannot be empty.')
            if len(sexp) != 2:       # type: ignore
                raise ValueError('Leafy Binary tree should have two children per branch node.')
            left, right = sexp       # type: ignore
            self._node: Either[A, tuple[LeafyBinaryTree[A], LeafyBinaryTree[A]]] = \
                Right((LeafyBinaryTree(left), LeafyBinaryTree(right)))
        else:  # Leaf node
            self._node = Left(sexp)  # type: ignore

        super().__init__()

    @classmethod
    def make(cls, node: Either[A, tuple[LeafyBinaryTree[A], LeafyBinaryTree[A]]]) -> LeafyBinaryTree[A]:
        "Creates and returns a leafy binary tree with specified data or subtrees."
        # ATTN: This is a hack! It fits the semantics of the function and causes no harm.
        t = cls(None)  # type: ignore
        t._node = node
        return t

    @classmethod
    def leaf(cls, data: A) -> LeafyBinaryTree[A]:
        "Creates a leafy binary tree singleton from a data value."
        return cls(data)

    @classmethod
    def branch(cls, left: LeafyBinaryTree[A], right: LeafyBinaryTree[A]) -> LeafyBinaryTree[A]:
        "Creates a leafy binary tree branch node from two subtrees."
        return cls(SExp([left.to_sexp(), right.to_sexp()]))

    def to_sexp(self):
        """Converts a leaf binary tree to s-expression format.

        A tree is represented by a list or tuple of the form
            leaf or (left, right) or [left, right],
        where left and right are either also leafy binary trees
        in s-expression format.

        ATTN

        This returns the s-expression as a list rather than a tuple
        so that the result can be modified.

        """
        match self.contents:
            case Left(leaf):
                return leaf

            case Right((left, right)):
                return SExp([left.to_sexp(), right.to_sexp()])

            case _:
                raise ValueError('Ill-formed LeafyBinaryTree: node of the wrong type')

    def as_str(self, levels=None):
        "Returns a simple string representation of this tree"
        match self.contents:
            case Left(leaf):
                return str(leaf) + '\n'

            case Right((left, right)):
                if levels is None:
                    levels = []
                indent = ''.join('\u2502  ' if level == 0 else '   ' for level in levels)
                lead_r = '\u251c\u2500 '
                lead_l = '\u2514\u2500 '
                root = '\u2022\n'

                left_s = f'{indent}{lead_l}{left.as_str([*levels, 0])}'
                right_s = f'{indent}{lead_r}{right.as_str([*levels, 1])}'

                # Put the left subtrees on the bottom so the tree is rotationally consistent
                return root + right_s + left_s

            case _:
                raise ValueError('Ill-formed LeafyBinaryTree: node of the wrong type')

    def __str__(self):
        return self.as_str().strip()

    @property
    def contents(self) -> Either[A, tuple[LeafyBinaryTree[A], LeafyBinaryTree[A]]]:
        "Returns the contents of the root node for processing."
        return self._node

    @classmethod
    def unfold[B](cls, gen: Callable[[B], Either[A, tuple[B, B]]], seed: B) -> LeafyBinaryTree[A]:
        "Creates a leafy binary tree by repeatedly unfolding a generating function from a starting seed."
        def unfold_sexp(s):
            match gen(s):
                case Left(a):
                    return a

                case Right((bl, br)):
                    return SExp([unfold_sexp(bl), unfold_sexp(br)])  # Distinguish Branch from Leaf

        return cls(unfold_sexp(seed))

    def map[B](self, g: Callable[[A], B]) -> LeafyBinaryTree[B]:
        "Maps a function over this binary tree, returning a new tree."
        match self.contents:
            case Left(leaf):
                return cast(LeafyBinaryTree[B], self.leaf(g(leaf)))  # type: ignore

            case Right((left, right)):
                return self.branch(map(g, left), map(g, right))      # type: ignore

            case _:
                raise ValueError('Ill-formed LeafyBinaryTree: node of the wrong type')

    def imap[I, B](self, g: Callable[[I, A], B]):
        "Maps an indexed function over this binary tree, returning a new tree."
        def go(index, tree):
            match tree.contents:
                case Left(leaf):
                    return self.leaf(g(index, leaf))

                case Right((left, right)):
                    return self.branch(go(index + List.of(0), left), go(index + List.of(1), right))

                case _:
                    raise ValueError('Ill-formed LeafyBinaryTree: node of the wrong type')
        return go(List(), self)

    @staticmethod
    def _lbt_traverse(_effect, node_f, subtree_f, t):
        match t.contents:
            case Left(leaf):
                return map(LeafyBinaryTree.leaf, node_f(leaf))

            case Right((left, right)):
                fl = subtree_f(left)
                fr = subtree_f(right)
                return ap(LeafyBinaryTree.branch, fl, fr)

            case _:
                raise ValueError('Ill-formed LeafyBinaryTree: node of the wrong type')

    def traverse(self, f: type[Applicative], g: Callable[[A], Applicative]) -> Applicative:
        "Applies an effectful function (a -> f b) to each node, collecting results as a same-shaped tree."
        def inorder(tree):
            return self._lbt_traverse(f, g, inorder, tree)
        return inorder(self)

#
# Heterogeneous Binary Trees - data at the leaves and branches of different types
#
#  data HetBinaryTree b a = Leaf a
#                         | Unary b (HetBinaryTree b a)
#                         | Binary (HetBinaryTree b a) b (HetBinaryTree b a)
#

# ATTN

#
# Tries (aka Prefix Trees)
#
#  data Trie = record Trie whee
#      key        : Maybe a
#      children   : Map k (Trie k m a)
#      annotation : m   -- an associated value or monoidal annotation
#

# ATTN

#
# Tulip Trees - like a Rose tree but with different types on leaf and branches
#
#   data TulipTree b l = Leaf l | Branch b (NonEmptyList HTree b l)

# ATTN


#
# Checking Protocol Compliance
#

if TYPE_CHECKING:
    _bt_check: type[IndexedFunctor] = BinaryTree
    _et_check: type[IndexedFunctor] = EmptyBinaryTree
