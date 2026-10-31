"""Tests specific to the IO monad."""

from __future__ import annotations

import io as pio

import pytest

from functorial.functor     import map         # pylint: disable=redefined-builtin
from functorial.io          import IO, as_io
from functorial.list        import List
from functorial.monad       import do
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


def test_bind_runs_both_actions_only_when_run():
    log: list[str] = []
    a = IO(lambda: as_io(log.append("a"), 1))
    b = a.bind(lambda x: IO(lambda: as_io(log.append("b"), x + 1)))

    assert log == []
    assert IO.unsafe_run(b) == 2
    assert log == ["a", "b"]

    # Running again re-executes the effects
    assert IO.unsafe_run(b) == 2
    assert log == ["a", "b", "a", "b"]


def test_join_defers_outer_effects():
    log: list[str] = []
    nested = IO(lambda: as_io(log.append("outer"),
                              IO(lambda: as_io(log.append("inner"), 5))))
    flat = nested.join()

    assert isinstance(flat, IO)
    assert log == []
    assert IO.unsafe_run(flat) == 5
    assert log == ["outer", "inner"]

def test_do_body_can_handle_action_exceptions():
    log = []

    def boom():
        raise RuntimeError('boom')

    @do(IO)
    def program():
        try:
            yield IO(boom)
        except RuntimeError:
            log.append('handled')
        finally:
            log.append('finally')
        return 'ok'

    assert IO.unsafe_run(program) == 'ok'
    assert log == ['handled', 'finally']

def test_do_finally_runs_promptly_on_unhandled_exception():
    log = []

    def boom():
        raise RuntimeError('boom')

    @do(IO)
    def program():
        try:
            yield IO(boom)
        finally:
            log.append('finally')

    caught = None
    try:
        IO.unsafe_run(program)
    except RuntimeError as e:
        caught = e    # holding the exception must not delay the finally
        assert log == ['finally']
    assert caught is not None

def test_stop_iteration_from_action_is_not_a_return():
    def stop():
        raise StopIteration

    @do(IO)
    def program():
        yield IO(stop)
        return 'unreachable'

    with pytest.raises(RuntimeError):
        IO.unsafe_run(program)
