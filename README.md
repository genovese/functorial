# functorial

[![PyPI - Version](https://img.shields.io/pypi/v/functorial.svg?cacheSeconds=300)](https://pypi.org/project/functorial)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/functorial.svg?cacheSeconds=300)](https://pypi.org/project/functorial)

-----

This package started as a small collection of modules for teaching
functional programming (FP) concepts to to Ph.D. students in Statistics
whose primary experience has been with Python and R, in a highly imperative style.
The context for this is a course to help prepare the students for building
good software in their research and industrial work.
Among other goals, I hope to expose them to different ways of thinking about problems
and different approaches to software design.
In the FP section of the course,
I hope to get them to internalize (at least a bit) 
the essence of "functional thinking".

Over time, this has developed into a more complete set of FP tools
to illustrate a wider range of examples and concepts.
These include:

+ Sum types built with Python metaclasses;
+ Standard typeclasses, including Functor, Applicative, Monad,
  Foldable, Traversable, Alternative, and many others.
+ Higher-kinded data types, including `Maybe`, `Either`, `List`,
  `Dict`, `NTuple`, a variety of trees, `IO`, `Reader`, `Writer`,
  and `State`;
+ Monadic do-notation, using Python decorators and generators;
+ Profunctor-based optics, with lenses, prisms, traversals, folds, isos, and their indexed variants;
+ Pure and effectful streams, folds, and transducers; and
+ Monadic parser combinators.

The package also includes a number of example modules that use
`functorial` to tackle a range of problems and applications.

The terminology in the package is strongly Haskell-influenced,
but I make changes freely for preference and clarity.
For example, I use `Some` and `Nothing` for the data constructors
of `Maybe` types, and I use *trait* in lieu of typeclasses.
As part of the course, I use several idealized "thinking languages"
for considering problems, solutions, and ideas. One of these (TL1
in the docs) is an idealized hybrid of Haskell, Idris, Unison,
and Lean. Comments in this form are often included in the documentation;
FP enthusiasts should be able to understand them easily.

The objects here are designed to fit with Python as much as possible.
For instance, `List` inherits from `list` but has more capabilities,
and these types all support pattern matching.
SImilarly, decorators are used for do-notation, making it clear and easy to use.
And the `Resource` monad (analogous to `ResourceT IO` for Haskellers)
works well on its own and but also integrates nicely with context managers.

A pure functional style necessarily involves some compromises in Python.
For instance, do-notation and traversals must be annotated with the
appropriate monad or applicative type,
and `IO` actions are wrapped in functions.
Educational value rather than performance is the main design criterion,
but I do weight performance and safety as much as possible.
For instance, `Resource` is thread-safe,
and some variants of streams are optimized
for "fusion" to provide good performance.
While this package is designed for educational rather than production use,
I have nonetheless found many these tools useful in several other projects.


It's been actually quite illuminating and fun to implement these ideas
in Python, and I've benefited from studying many terrific packages in the Haskell
ecosystem, notably including
[optics](https://hackage.haskell.org/package/optics-0.4.2.1/docs/Optics.html),
[streamly](https://streamly.composewell.com/),
[resourcet](https://hackage.haskell.org/package/resourcet),
and [tree-traversals](https://hackage.haskell.org/package/tree-traversals).

This is an evolving, educational project. Some of the API is still
settling, some of the dev components have not yet been merged into
main, and not everything on the roadmap is implemented yet. Feedback
and issues are welcome.

## Installation

`functorial` is available on [PyPi](https://pypi.org/).
To install, you can do
```bash
pip install functorial
```
or
```bash
uv add functorial
```
in a `uv` project. This requires Python 3.12+,
and later versions are recommended.

## Quick taste

### Functors and such

```python
from functorial.maybe         import Maybe, Some, Nothing

def a_partial_func(x: int) -> Maybe[float]:
    if x <= 0:
        return Nothing()
    return Some(math.log(x))

match a_partial_func(10):
    case Some(y):
        ...
    case Nothing():
        ...
    case invalid:
        raise SumTypeInvalidCaseException(invalid, Maybe)

Some(3).map(lambda x: x + 1)     # Some(4)
Nothing().map(lambda x: x + 1)   # Nothing()

sequence(List.of(Some(3), Some(2), Some(1)), Maybe)
    # => Some([3, 2, 1])

sequence(List.of(Some(3), Nothing(), Some(1)), Maybe)
    # => Nothing()

t1 = complete_btree(3)
label = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
         "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
         "W", "X", "Y", "Z", "AA", "BB", "CC", "DD", "EE", "FF",]
t1a = t1.map(lambda k: label[k])
print(t1a)
# A
# ├─ C
# │  ├─ G
# │  │  ├─ O
# │  │  └─ N
# │  └─ F
# │     ├─ M
# │     └─ L
# └─ B
#    ├─ E
#    │  ├─ K
#    │  └─ J
#    └─ D
#       ├─ I
#       └─ H

t2 = RoseTree([1, [2, [3], [4], [5]], [6, [7, [8, [9], [10]]]]])
t3 = t2.map(lambda k: k ** 2)
print(t3)
# 1
# ├─ 4
# │  ├─ 9
# │  ├─ 16
# │  └─ 25
# └─ 36
#    └─ 49
#       └─ 64
#          ├─ 81
#          └─ 100
```

### do-notation and effects

```python
@do(Either)
def summed3a():
    x = yield Right(4)
    y = yield Right(10)
    z = yield Right(6)
    return x + y + z

@do(Either)
def summed3b():
    x = yield Right(4)
    y = yield Left(9)    # short circuits
    z = yield Right(10)
    return x + y + z

# sumed3a == Right(20)
# sumed3b == Left(9)

@effect(Either)         # effect makes a function wrapping do
def sum3(mx, my, mz):
    x = yield mx
    y = yield my
    z = yield mz
    return x + y + z

# sum3(Right(4), Left(5), Right(23))  == Left(5)

@effect(List)
def pairs(xs, ys):
    x = yield xs
    y = yield ys
    return (x, y)

# pairs([1, 2], [7, 8, 9])
# == [(1, 7), (1, 8), (1, 9), (2, 7), (2, 8), (2, 9)]
```

### Optics

```python
# Compose lenses with @, act on them with view/over
fst = lens(lambda s: s[0], lambda s, b: (b, s[1]))
snd = lens(lambda s: s[1], lambda s, b: (s[0], b))

view(fst)((1, 2))                     # 1
over(fst @ snd, str)(((1, 2), 3))     # ((1, '2'), 3)
```

## Documentation

*Coming Soon*

## Development

This is a `uv`-based project. With the repo cloned,
you can do
```bash
uv sync
uv run pytest
```
One of the default groups is `elpy` that installs tools
in the venv for use with emacs/elpy.

## License

MIT — see [LICENSE](LICENSE).
