"""Tests specific to the Resource monad."""

from __future__ import annotations

import io as pio
import os
import tempfile
import threading
import time

from contextlib import contextmanager
from operator   import itemgetter

import pytest

from functorial.io          import IO, as_io
from functorial.monad       import do
from functorial.resource    import (Resource, ResourceRegistry, acquire, acquire_cm, acquire_with, open_file,
                                    release, lift_io, temp_file, attach, detach, AfterError, AtCleanup, EarlyRelease)

def close_file_in_io(fp) -> IO:
    return IO(lambda: fp.close())

def open_file_in_io(name, mode) -> IO:
    return IO(lambda: open(name, mode))

fst = itemgetter(0)
snd = itemgetter(1)


def test_resource(tmp_path):

    rtext = tmp_path / "test-resource-text"
    rnuml = tmp_path / "test-resource-numl"

    @do(Resource)
    def make_test_file():
        token, f = yield acquire(open_file_in_io(rtext, "w"), close_file_in_io)
        for i in range(16):
            print(f'Line {i + 1}', file=f)
        print("Done", file=f)

    IO.unsafe_run(make_test_file.run())

    out1 = pio.StringIO()
    x1 = acquire(open_file_in_io(rtext, "r"), close_file_in_io)
    x2 = x1.bind(lambda t_fp: lift_io(IO(lambda: as_io(print(snd(t_fp).read(), end='', file=out1), 32))))
    out1ret = IO.unsafe_run(x2.run())
    assert out1ret == 32
    assert out1.getvalue() == "\n".join([f'Line {i + 1}' for i in range(16)]) + '\nDone\n'
    out1.close()

    out2 = pio.StringIO()

    @do(Resource)
    def example2():
        token, f = yield acquire(open_file_in_io(rnuml, "w"), close_file_in_io)
        print('4', file=f)
        yield release(token)

        token1, f1 = yield acquire(IO(lambda: as_io(print("open first", file=out2), open(rnuml))),
                                   lambda f: IO(lambda: as_io(print("close first", file=out2), f.close(), ())))
        print("start", file=out2)
        n_lines = int(f1.read())
        print(n_lines, "lines", file=out2)
        yield release(token1)

        token2, f2 = yield acquire(IO(lambda: as_io(print("open secnd", file=out2), open(rtext))),
                                   lambda f: IO(lambda: as_io(print("close secnd", file=out2), f.close(), ())))
        n = 0
        for line in f2:
            if n >= n_lines:
                break
            print(line, end='', file=out2)
            n += 1
        print("done", file=out2)

        return ()

    tmp2 = IO.unsafe_run(example2.run())
    assert tmp2 == ()

    with open(rnuml) as f:
        n_check = int(f.read())
    assert n_check == 4

    expected2 = """open first
start
4 lines
close first
open secnd
Line 1
Line 2
Line 3
Line 4
done
close secnd
"""
    assert out2.getvalue() == expected2
    out2.close()

    out3 = pio.StringIO()
    x3 = acquire_cm(open_file_in_io(rtext, "r"))  # No close needed
    x4 = x3.bind(lambda t_fp: lift_io(IO(lambda: as_io(print(snd(t_fp).read(), end='', file=out3), 32))))
    out3ret = IO.unsafe_run(x4.run())
    assert out3ret == 32
    assert out3.getvalue() == "\n".join([f'Line {i + 1}' for i in range(16)]) + '\nDone\n'
    out3.close()


#
# Helpers: resources that log their lifecycle
#

def logged(log, name, fail_on_free=False):
    """Returns a Resource acquiring a named handle, logging open and close."""
    def free(handle):
        def close():
            log.append(f'close {handle}')
            if fail_on_free:
                raise ValueError(f'cannot close {handle}')
            return ()
        return IO(close)

    return acquire(IO(lambda: as_io(log.append(f'open {name}'), name)), free)

def run(resource):
    return IO.unsafe_run(resource.run())


#
# do-notation
#

def test_do_and_bind_agree():
    log = []

    via_bind = logged(log, 'h').bind(lambda token_h: Resource.pure(snd(token_h)))

    @do(Resource)
    def via_do():
        _, h = yield logged(log, 'h')
        return h

    assert run(via_bind) == run(via_do) == 'h'
    assert log == ['open h', 'close h'] * 2

def test_do_body_runs_only_when_run():
    log = []

    @do(Resource)
    def program():
        log.append('body')
        yield logged(log, 'a')
        log.append('after a')

    assert not log
    resource = program.run()
    assert not log

    IO.unsafe_run(resource)
    IO.unsafe_run(resource)
    assert log == ['body', 'open a', 'after a', 'close a'] * 2

def test_do_body_is_ordered_with_acquisitions():
    log = []

    @do(Resource)
    def program():
        log.append('start')
        _, a = yield logged(log, 'a')
        log.append(f'using {a}')
        _, b = yield logged(log, 'b')
        log.append(f'using {b}')
        return a + b

    assert run(program) == 'ab'
    assert log == ['start', 'open a', 'using a', 'open b', 'using b', 'close b', 'close a']

def test_yielded_io_is_lifted():
    @do(Resource)
    def program():
        x = yield IO.pure(3)
        y = yield lift_io(IO.pure(4))
        return x + y

    assert run(program) == 7

@pytest.mark.parametrize('returned, expected', [
    (5,                      5),
    (IO.pure(6),             6),
    (Resource.pure(7),       7),
    (lift_io(IO.pure(8)),     8),
])
def test_do_return_value_is_run(returned, expected):
    @do(Resource)
    def program():
        yield IO.pure(())
        return returned

    @do(Resource)
    def non_generator():
        return returned

    assert run(program) == expected
    assert run(non_generator) == expected

def test_nested_do_shares_registry():
    log = []

    @do(Resource)
    def inner():
        _, b = yield logged(log, 'b')
        log.append('inner done')
        return b

    @do(Resource)
    def outer():
        yield logged(log, 'a')
        b = yield inner
        log.append(f'outer got {b}')
        return b

    assert run(outer) == 'b'
    # Resources acquired in inner live until the end of the whole region
    assert log == ['open a', 'open b', 'inner done', 'outer got b', 'close b', 'close a']


#
# Release
#

def test_release_order_is_lifo():
    log = []

    @do(Resource)
    def program():
        for name in 'abc':
            yield logged(log, name)

    run(program)
    assert log == ['open a', 'open b', 'open c', 'close c', 'close b', 'close a']

def test_early_release_frees_immediately_and_only_once():
    log = []

    @do(Resource)
    def program():
        token_a, _ = yield logged(log, 'a')
        yield logged(log, 'b')
        yield release(token_a)
        log.append('released a')
        yield release(token_a)       # second release is a no-op

    run(program)
    assert log == ['open a', 'open b', 'close a', 'released a', 'close b']

def test_release_after_region_is_a_noop():
    log = []
    tokens = []

    @do(Resource)
    def program():
        token, _ = yield logged(log, 'a')
        tokens.append(token)

    run(program)
    IO.unsafe_run(release(tokens[0]))
    assert log == ['open a', 'close a']

def test_release_works_in_io_do():
    log = []

    @do(Resource)
    def program():
        token, _ = yield logged(log, 'a')

        @do(IO)
        def finish():
            log.append('in io')
            yield release(token)
            log.append('io done')

        yield finish
        log.append('resource done')

    run(program)
    assert log == ['open a', 'in io', 'close a', 'io done', 'resource done']


#
# Exceptions
#

def test_exception_releases_resources_promptly():
    log = []

    @do(Resource)
    def failing():
        yield logged(log, 'a')
        yield logged(log, 'b')
        raise RuntimeError('boom')

    caught = None
    try:
        run(failing)
    except RuntimeError as e:
        caught = e    # holding the exception must not delay cleanup
        assert log == ['open a', 'open b', 'close b', 'close a']
    assert caught is not None

def test_failing_action_releases_resources_promptly():
    log = []

    def boom():
        raise RuntimeError('boom')

    @do(Resource)
    def failing():
        yield logged(log, 'a')
        yield IO(boom)
        log.append('unreachable')

    caught = None
    try:
        run(failing)
    except RuntimeError as e:
        caught = e
        assert log == ['open a', 'close a']
    assert caught is not None

def test_do_body_can_handle_action_exceptions():
    log = []

    def boom():
        raise RuntimeError('boom')

    @do(Resource)
    def program():
        yield logged(log, 'a')
        try:
            yield IO(boom)
        except RuntimeError:
            log.append('handled')
        finally:
            log.append('finally')
        return 'ok'

    assert run(program) == 'ok'
    assert log == ['open a', 'handled', 'finally', 'close a']

def test_failed_allocation_registers_nothing():
    log = []

    def boom():
        raise OSError('no such thing')

    @do(Resource)
    def program():
        yield logged(log, 'a')
        yield acquire(IO(boom), lambda _: IO(lambda: as_io(log.append('wrong close'), ())))

    with pytest.raises(OSError):
        run(program)
    assert log == ['open a', 'close a']

def test_single_cleanup_error_is_raised_and_others_still_freed():
    log = []

    @do(Resource)
    def program():
        yield logged(log, 'a')
        yield logged(log, 'b', fail_on_free=True)
        yield logged(log, 'c')

    with pytest.raises(ValueError, match='cannot close b'):
        run(program)
    assert log == ['open a', 'open b', 'open c', 'close c', 'close b', 'close a']

def test_multiple_cleanup_errors_are_grouped():
    log = []

    @do(Resource)
    def program():
        yield logged(log, 'a', fail_on_free=True)
        yield logged(log, 'b')
        yield logged(log, 'c', fail_on_free=True)

    with pytest.raises(ExceptionGroup) as info:
        run(program)
    assert [str(e) for e in info.value.exceptions] == ['cannot close c', 'cannot close a']
    assert log == ['open a', 'open b', 'open c', 'close c', 'close b', 'close a']

def test_cleanup_error_chains_body_exception():
    log = []

    @do(Resource)
    def program():
        yield logged(log, 'a', fail_on_free=True)
        raise RuntimeError('boom')

    with pytest.raises(ValueError) as info:
        run(program)
    assert isinstance(info.value.__context__, RuntimeError)


#
# Release reasons
#

def reason_logger(log, name):
    """Returns a two-argument free action that logs why it was called."""
    def free(handle, why):
        return IO(lambda: as_io(log.append((handle, why)), ()))

    return acquire_with(IO.pure(name), free)

def test_release_reasons():
    log = []

    @do(Resource)
    def program():
        token, _ = yield reason_logger(log, 'early')
        yield reason_logger(log, 'late')
        yield release(token)

    run(program)
    assert log == [('early', EarlyRelease()), ('late', AtCleanup())]

def test_release_reason_carries_the_exception():
    log = []
    boom = RuntimeError('boom')

    @do(Resource)
    def program():
        yield reason_logger(log, 'a')
        raise boom

    with pytest.raises(RuntimeError):
        run(program)

    [(handle, why)] = log
    assert handle == 'a'
    assert isinstance(why, AfterError)
    match why:
        case AfterError(error):
            assert error is boom

def test_acquire_ignores_reason():
    log = []

    @do(Resource)
    def program():
        token, _ = yield logged(log, 'a')
        yield release(token)
        yield logged(log, 'b')

    run(program)
    assert log == ['open a', 'close a', 'open b', 'close b']


#
# The registry
#

def test_drain_closes_the_registry():
    registry = ResourceRegistry()
    token = registry.register(lambda why: IO.pure(()))
    assert token is not None

    remaining = list(registry.drain())
    assert len(remaining) == 1
    assert registry.register(lambda why: IO.pure(())) is None
    assert registry.deregister(token) is None
    assert not list(registry.drain())

def test_acquire_into_closed_registry_frees_and_raises():
    log = []
    tokens = []

    @do(Resource)
    def program():
        token, _ = yield logged(log, 'a')
        tokens.append(token)

    run(program)
    registry = tokens[0].registry

    late = reason_logger(log, 'late')
    with pytest.raises(Exception, match='closed registry'):
        IO.unsafe_run(late._action(registry))   # pylint: disable=protected-access
    [*_, (handle, why)] = log
    assert handle == 'late'
    assert isinstance(why, AfterError)

def test_early_release_racing_cleanup_frees_exactly_once():
    n_resources = 200
    for _ in range(20):
        freed = []
        tokens = []
        barrier = threading.Barrier(2)

        def release_all():
            barrier.wait()
            for token in reversed(tokens):     # same order as cleanup, to maximize collisions
                IO.unsafe_run(release(token))

        def free(h):
            def close():
                time.sleep(0)   # yield the GIL, as real frees (e.g., closing files) do
                freed.append(h)
                return ()
            return IO(close)

        @do(Resource)
        def program():
            for i in range(n_resources):
                token, _ = yield acquire(IO.pure(i), free)
                tokens.append(token)
            barrier.wait()      # cleanup now races with the other thread's releases

        releaser = threading.Thread(target=release_all)
        releaser.start()
        run(program)
        releaser.join()

        assert sorted(freed) == list(range(n_resources))


#
# Misuse
#

def test_yielding_a_non_monadic_value_is_a_type_error():
    @do(Resource)
    def program():
        yield 3

    with pytest.raises(TypeError, match='Resource or IO'):
        run(program)


#
# Context managers
#

class Recorder:
    """A context manager that records its __enter__ and __exit__ calls."""
    def __init__(self, log, name, *, value=None, suppress=False, fail_enter=False, fail_exit=False):
        self.log = log
        self.name = name
        self.value = value if value is not None else name.upper()
        self.suppress = suppress
        self.fail_enter = fail_enter
        self.fail_exit = fail_exit

    def __enter__(self):
        if self.fail_enter:
            raise OSError(f'cannot enter {self.name}')
        self.log.append(f'enter {self.name}')
        return self.value

    def __exit__(self, exc_type, exc, tb):
        self.log.append((f'exit {self.name}', exc_type, exc, tb is not None))
        if self.fail_exit:
            raise ValueError(f'cannot exit {self.name}')
        return self.suppress

def test_acquire_cm_defers_construction_until_run():
    log = []
    built = []

    def make():
        built.append('built')
        return Recorder(log, 'a')

    resource = acquire_cm(make).run()
    assert not built and not log

    IO.unsafe_run(resource)
    assert built == ['built']
    assert log == ['enter a', ('exit a', None, None, False)]

def test_acquire_cm_yields_enter_value():
    @do(Resource)
    def program():
        _, value = yield acquire_cm(lambda: Recorder([], 'a', value=42))
        return value

    assert run(program) == 42

def test_acquire_cm_interleaves_with_acquire_in_lifo_order():
    log = []

    @do(Resource)
    def program():
        yield acquire_cm(lambda: Recorder(log, 'a'))
        yield logged(log, 'b')
        yield acquire_cm(lambda: Recorder(log, 'c'))

    run(program)
    assert log == ['enter a', 'open b', 'enter c',
                   ('exit c', None, None, False), 'close b', ('exit a', None, None, False)]

def test_acquire_cm_early_release_exits_once():
    log = []

    @do(Resource)
    def program():
        token, _ = yield acquire_cm(lambda: Recorder(log, 'a'))
        yield release(token)
        log.append('released')
        yield release(token)

    run(program)
    assert log == ['enter a', ('exit a', None, None, False), 'released']

def test_acquire_cm_exit_receives_the_exception():
    log = []
    boom = RuntimeError('boom')

    @do(Resource)
    def program():
        yield acquire_cm(lambda: Recorder(log, 'a'))
        raise boom

    with pytest.raises(RuntimeError):
        run(program)
    assert log == ['enter a', ('exit a', RuntimeError, boom, True)]

def test_acquire_cm_exit_cannot_suppress():
    log = []

    @do(Resource)
    def program():
        yield acquire_cm(lambda: Recorder(log, 'a', suppress=True))
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        run(program)

def test_acquire_cm_failed_enter_skips_exit():
    log = []

    @do(Resource)
    def program():
        yield logged(log, 'a')
        yield acquire_cm(lambda: Recorder(log, 'b', fail_enter=True))

    with pytest.raises(OSError, match='cannot enter b'):
        run(program)
    assert log == ['open a', 'close a']

def test_acquire_cm_failed_exit_is_a_cleanup_error():
    log = []

    @do(Resource)
    def program():
        yield acquire_cm(lambda: Recorder(log, 'a'))
        yield acquire_cm(lambda: Recorder(log, 'b', fail_exit=True))

    with pytest.raises(ValueError, match='cannot exit b'):
        run(program)
    assert [entry[0] for entry in log[2:]] == ['exit b', 'exit a']

def test_acquire_cm_generator_context_manager_sees_exception():
    log = []

    @contextmanager
    def guarded():
        try:
            yield 'g'
        except RuntimeError as e:
            log.append(f'saw {e}')
            raise
        finally:
            log.append('finally')

    @do(Resource)
    def program():
        yield acquire_cm(guarded)
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        run(program)
    assert log == ['saw boom', 'finally']

def test_acquire_cm_holds_lock_for_the_region():
    lock = threading.Lock()

    @do(Resource)
    def program():
        _, acquired = yield acquire_cm(lambda: lock)
        return acquired, lock.locked()

    assert run(program) == (True, True)
    assert not lock.locked()

def test_acquire_cm_looks_up_methods_on_the_type():
    log = []
    recorder = Recorder(log, 'a')
    recorder.__enter__ = lambda: 'instance attribute'    # ignored, as in a with statement

    @do(Resource)
    def program():
        _, value = yield acquire_cm(lambda: recorder)
        return value

    assert run(program) == 'A'


#
# File convenience functions
#

def test_open_file_text_round_trip(tmp_path):
    path = tmp_path / 'text.txt'

    @do(Resource)
    def write():
        _, f = yield open_file(path, 'w', -1, 'utf-16', newline='\r\n')
        f.write('héllo\nworld\n')
        return f

    @do(Resource)
    def read():
        _, f = yield open_file(path, 'r', -1, 'utf-16')
        return f.read(), f

    written = run(write)
    text, read_from = run(read)
    assert text == 'héllo\nworld\n'
    assert path.read_bytes() == 'héllo\r\nworld\r\n'.encode('utf-16')
    assert written.closed and read_from.closed

@pytest.mark.parametrize('mode, buffering, expected_type', [
    ('rb',  -1, pio.BufferedReader),
    ('rb',  0,  pio.FileIO),
    ('r+b', -1, pio.BufferedRandom),
    ('ab',  -1, pio.BufferedWriter),
    ('r',   -1, pio.TextIOWrapper),
])
def test_open_file_forwards_mode_and_buffering(tmp_path, mode, buffering, expected_type):
    path = tmp_path / 'data.bin'
    path.write_bytes(b'abc')

    @do(Resource)
    def program():
        _, f = yield open_file(path, mode, buffering)
        return f

    f = run(program)
    assert type(f) is expected_type
    assert f.closed

def test_open_file_binary_round_trip(tmp_path):
    path = tmp_path / 'data.bin'

    @do(Resource)
    def program():
        token, out = yield open_file(path, 'wb')
        out.write(b'\x00\x01\x02')
        yield release(token)            # flushes and closes before reading back
        _, back = yield open_file(path, 'rb')
        return back.read()

    assert run(program) == b'\x00\x01\x02'

def test_open_file_missing_file_registers_nothing(tmp_path):
    log = []

    @do(Resource)
    def program():
        yield logged(log, 'a')
        yield open_file(tmp_path / 'missing.txt')

    with pytest.raises(FileNotFoundError):
        run(program)
    assert log == ['open a', 'close a']

@pytest.mark.parametrize('kwds', [{}, {'named': True}, {'spooled': True}])
@pytest.mark.parametrize('mode, data', [('w+b', b'bytes'), ('w+', 'text')])
def test_temp_file_kinds_and_modes(kwds, mode, data):
    @do(Resource)
    def program():
        _, f = yield temp_file(mode, **kwds)
        f.write(data)
        f.seek(0)
        return f.read(), f

    contents, f = run(program)
    assert contents == data
    assert f.closed
    assert isinstance(f, tempfile.SpooledTemporaryFile) == bool(kwds.get('spooled'))

def test_temp_file_forwards_text_options():
    @do(Resource)
    def program():
        _, f = yield temp_file('w+', -1, 'utf-16', newline='\r\n')
        f.write('a\nb')
        f.flush()
        f.buffer.seek(0)
        return f.buffer.read()

    assert run(program) == 'a\r\nb'.encode('utf-16')

def test_named_temp_file_is_deleted_after_region():
    names = []

    @do(Resource)
    def program():
        _, f = yield temp_file('w+', named=True, prefix='fnc-', suffix='.txt')
        names.append(f.name)
        return os.path.exists(f.name)

    assert run(program)
    name = os.path.basename(names[0])
    assert name.startswith('fnc-') and name.endswith('.txt')
    assert not os.path.exists(names[0])

def test_named_temp_file_kept_without_delete(tmp_path):
    @do(Resource)
    def program():
        _, f = yield temp_file('w+', named=True, delete=False, dir=tmp_path)
        f.write('kept')
        return f.name

    name = run(program)
    assert os.path.dirname(name) == str(tmp_path)
    with open(name) as f:
        assert f.read() == 'kept'

def test_spooled_temp_file_rolls_over_at_max_size():
    @do(Resource)
    def program():
        _, f = yield temp_file('w+b', spooled=True, max_size=8)
        f.write(b'1234')
        before = f._rolled                  # pylint: disable=protected-access
        f.write(b'56789')
        return before, f._rolled            # pylint: disable=protected-access

    assert run(program) == (False, True)


#
# Detaching and attaching
#

def test_detached_resource_survives_the_region():
    log = []

    @do(Resource)
    def program():
        token, handle = yield logged(log, 'a')
        yield logged(log, 'b')
        free = yield detach(token)
        return handle, free

    handle, free = run(program)
    assert handle == 'a'
    assert log == ['open a', 'open b', 'close b']

    IO.unsafe_run(free(EarlyRelease()))
    assert log == ['open a', 'open b', 'close b', 'close a']

def test_detached_free_action_receives_its_reason():
    log = []

    @do(Resource)
    def program():
        token, _ = yield reason_logger(log, 'a')
        return (yield detach(token))

    free = run(program)
    assert not log
    IO.unsafe_run(free(AtCleanup()))
    assert log == [('a', AtCleanup())]

def test_detach_after_release_returns_none():
    log = []

    @do(Resource)
    def program():
        token, _ = yield logged(log, 'a')
        yield release(token)
        return (yield detach(token))

    assert run(program) is None
    assert log == ['open a', 'close a']

def test_detach_after_region_returns_none():
    log = []
    tokens = []

    @do(Resource)
    def program():
        token, _ = yield logged(log, 'a')
        tokens.append(token)

    run(program)
    assert IO.unsafe_run(detach(tokens[0])) is None
    assert log == ['open a', 'close a']

def test_detach_twice_returns_none_the_second_time():
    @do(Resource)
    def program():
        token, _ = yield logged([], 'a')
        first = yield detach(token)
        second = yield detach(token)
        return first, second

    first, second = run(program)
    assert first is not None and second is None
    IO.unsafe_run(first(EarlyRelease()))

def test_attach_moves_a_resource_between_regions():
    log = []

    @do(Resource)
    def first():
        token, handle = yield logged(log, 'a')
        free = yield detach(token)
        return handle, free

    handle, free = run(first)

    @do(Resource)
    def second():
        yield logged(log, 'b')
        yield attach(free)
        log.append(f'using {handle}')

    run(second)
    # Attached last, so released first
    assert log == ['open a', 'open b', 'using a', 'close a', 'close b']

def test_attached_resource_can_be_released_early():
    log = []

    @do(Resource)
    def program():
        token, _ = yield reason_logger(log, 'a')
        free = yield detach(token)
        new_token = yield attach(free)
        yield release(new_token)
        log.append('released')

    run(program)
    assert log == [('a', EarlyRelease()), 'released']

def test_attach_passes_exception_reason():
    log = []
    boom = RuntimeError('boom')

    @do(Resource)
    def program():
        token, _ = yield reason_logger(log, 'a')
        free = yield detach(token)
        yield attach(free)
        raise boom

    with pytest.raises(RuntimeError):
        run(program)
    [(handle, why)] = log
    assert handle == 'a' and why == AfterError(boom)

def test_attach_to_closed_registry_frees_and_raises():
    log = []
    tokens = []

    @do(Resource)
    def program():
        token, _ = yield logged(log, 'a')
        tokens.append(token)
        return (yield detach(token))

    free = run(program)
    registry = tokens[0].registry       # closed when the region ended

    with pytest.raises(RuntimeError, match='closed registry'):
        IO.unsafe_run(attach(free)._action(registry))   # pylint: disable=protected-access
    assert log == ['open a', 'close a']

def test_detach_for_hand_off_on_success():
    """Acquires several resources; keeps all on success, frees all on failure."""
    log = []

    def open_pair(fail):
        @do(Resource)
        def program():
            token_a, a = yield logged(log, 'a')
            token_b, b = yield logged(log, 'b')
            if fail:
                raise OSError('setup failed')
            free_a = yield detach(token_a)
            free_b = yield detach(token_b)
            return (a, b), (free_b, free_a)
        return program

    with pytest.raises(OSError):
        run(open_pair(fail=True))
    assert log == ['open a', 'open b', 'close b', 'close a']

    log.clear()
    pair, frees = run(open_pair(fail=False))
    assert pair == ('a', 'b')
    assert log == ['open a', 'open b']
    for free in frees:
        IO.unsafe_run(free(EarlyRelease()))
    assert log == ['open a', 'open b', 'close b', 'close a']
