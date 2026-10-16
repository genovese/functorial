"""Tests specific to the IO monad."""

from __future__ import annotations

import io as pio

from functorial.functor     import map         # pylint: disable=redefined-builtin
from functorial.io          import IO, as_io
from functorial.list        import List
from functorial.trees       import RoseTree
from functorial.traversable import sequence, traverse

def test_io():
    x = RoseTree([1, [2, [3], [4], [5]], [6, [7, [8, [9], [10]]]]])
    y = map(lambda xi: xi + 1000, x)
    out = pio.StringIO()

    def do_it(xi):
        return IO(lambda: as_io(print("(" + str(xi) + ")", file=out), xi + 1000))

    iox = traverse(do_it, x, IO)
    x_prime = IO.unsafe_run(iox)
    out_prime = "\n".join(f'({i})' for i in range(1, 11)) + "\n"

    assert x_prime == y
    assert out.getvalue() == out_prime


    out2 = pio.StringIO()
    ell = List.of(
        IO(lambda: as_io(print("foo", file=out2), 10)),
        IO(lambda: as_io(print("bar", file=out2), 20)),
        IO(lambda: as_io(print("zap", file=out2), print("ok", file=out2), 30)),
        IO(lambda: as_io(print("wow", file=out2), print("***", file=out2), 40))
    )

    z_prime = IO.unsafe_run(sequence(ell, effect=IO))
    assert z_prime == [10, 20, 30, 40]
    assert out2.getvalue() == "foo\nbar\nzap\nok\nwow\n***\n"
