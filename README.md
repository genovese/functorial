# functorial

[![PyPI - Version](https://img.shields.io/pypi/v/functorial.svg?cacheSeconds=300)](https://pypi.org/project/functorial)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/functorial.svg?cacheSeconds=300)](https://pypi.org/project/functorial)

-----


Functional-programming concepts for Python: typeclasses (Functor, Applicative,
Monad, Foldable, Traversable, Alternative, ...), standard FP data types
(`Maybe`, `Either`, `List`, `Dict`, `NTuple`, trees, `IO`, `Reader`, `Writer`,
`State`), and a profunctor-based optics library (lenses, prisms, traversals,
folds, isos, and their indexed variants).

This is an evolving, educational project — the API is still settling and not
everything on the roadmap is implemented yet. Feedback and issues are welcome.

## Installation

```bash
pip install functorial
# or
uv add functorial
```

Requires Python 3.12+.

## Quick taste

```python
from functorial.maybe import Some, Nothing
from functorial.optics.lens import lens
from functorial.optics.getter import view
from functorial.optics.setter import over

# Maybe: a Functor/Applicative/Monad/Traversable
Some(3).map(lambda x: x + 1)     # Some(4)
Nothing().map(lambda x: x + 1)   # Nothing()

# Optics: compose lenses with @, act on them with view/over
fst = lens(lambda s: s[0], lambda s, b: (b, s[1]))
snd = lens(lambda s: s[1], lambda s, b: (s[0], b))

view(fst)((1, 2))                     # 1
over(fst @ snd, str)(((1, 2), 3))     # ((1, '2'), 3)
```

## Development

```bash
uv sync
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
