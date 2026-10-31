"""Resource (in IO) Monad

This provides a computational context in which resources
can be allocated and released and in which they are guaranteed
to be released on exit from the context, either by ordinary
execution or after an exception.

This provides functionality similar to context managers
in Python (and in particular, contextlib.ExitStack)
but encapsulated in a computational object.

The management of the resources is thread-safe, and
the actions releasing those resources can determine
the reason for release (EarlyRelease, AtCleanup, AfterError)
to make context-sensitive decisions.

Note: Resource[A] is analogous to ResourceT IO a in Haskell

"""
# ruff: noqua: FTPT001, FBT002
from __future__      import annotations

import io as pio
import tempfile
import typing

from collections.abc import Callable
from contextlib      import AbstractContextManager
from dataclasses     import dataclass
from threading       import Lock
from typing          import Any, BinaryIO, Literal, overload, TYPE_CHECKING


from .applicative    import map2
from .functions      import const
from .functor        import map
from .io             import IO
from .monad          import Monad
from .sum_type       import SumType
from .unit           import Unit

__all__ = [
    'acquire_cm',
    'acquire_with',
    'acquire',
    'AfterError',
    'AtCleanup',
    'attach',
    'detach',
    'EarlyRelease',
    'lift_io',
    'open_file',
    'release',
    'ReleaseReason',
    'ReleaseToken',
    'Resource',
    'run_in_io',
    'temp_file',
]


#
# Reasons that a resource was released
#

class ReleaseReason(SumType, data=['EarlyRelease', 'AtCleanup', 'AfterError']):
    ...

class EarlyRelease(ReleaseReason, singleton=True):
    """Resource released explicitly by the user."""

class AtCleanup(ReleaseReason, singleton=True):
    """Resource released during normal cleanup."""

@dataclass
class AfterError(ReleaseReason):
    """Resource released in response to an exception."""
    error: BaseException


#
# Helpers
#

class ReleaseToken:
    """Uniquely identifies a resource in a specific registry."""
    __slots__ = ('_registry',)

    def __init__(self, registry):
        self._registry = registry

    @property
    def registry(self):
        return self._registry

class ResourceRegistry:
    """Store for resource-freeing actions for all allocated resources.

    When closed, no additional resources can be registered.

    """
    __slots__ = ('_entries', '_lock', '_closed')

    def __init__(self) -> None:
        self._entries: dict[ReleaseToken, Callable[[ReleaseReason], IO[Unit]]] = {}
        self._lock = Lock()
        self._closed = False

    def register(self, free: Callable[[ReleaseReason], IO[Unit]]) -> ReleaseToken | None:
        """Registers a free action for a resource and returns a unique token for releasing it."""
        with self._lock:
            if self._closed:
                # Here, we expect caller to free the new resource and raise
                return None
            token = ReleaseToken(self)
            self._entries[token] = free
            return token

    def deregister(self, token) -> Callable[[ReleaseReason], IO[Unit]] | None:
        """Deregisters a resource and returns its free action, or None if already released."""
        # This operation is atomic without a lock
        return self._entries.pop(token, None)

    def close(self):
        """Prevents further registrations in this registry."""
        with self._lock:
            self._closed = True

    def drain(self):
        """Closes the registry and returns an iterator over all remaining items.

        The items are deregistered as they are returned.

        """
        self.close()
        return self._remaining_all_deregistered()

    def _remaining_all_deregistered(self):
        # This is used after the registry is closed and so is guaranteed
        # to terminate.
        while True:
            try:
                yield self._entries.popitem()[1]
            except KeyError:
                return


#
# Resource in IO Monad
#

class Resource[A](Monad):
    """The Resource Monad

    This provides an effectful computational context (a monad) in
    which a pool of resources is managed. Resources can be allocated,
    used, and released flexibly, and all resources are guaranteed to
    be released on exit from the context, whether through ordinary
    execution or via an exception. Resource management is thread safe.
    Releasing a resource is guaranteed to only happen once,
    and the release operation is idempotent.

    A resource object can be run to produce an IO action or
    run in IO (see `run_in_io`) to produce a value and its
    side effects in an IO context.

    Key operations:

      + acquire_with, acquire, acquire_cm :: allocate a resource
          via an IO action, registering a cleanup action
      + release :: releases a resource early
      + .run() :: produces an IO action from a Resource object
      + .unsafe_run() :: runs the returned IO action immediately
      + detach :: take responsibility for freeing a particular resource
      + attach :: register an existing resource
      + run_in_io :: run the resulting IO action in an IO context
      + lift_io :: wraps an IO action in a Resource object

    In addition, as a monad, Resource[A] inherits all the
    Functor, Applicative, and Monad methods.

    Example: The following produces an effectful function
    that combines several files, processing the lines
    of the middle file, into a new output file. This uses
    a convenience function in this module, open_file, to
    represent an open file resource for several files.
    It reads from the files, releasing them when finished
    and letting two release on cleanup. Note that the
    acquire step gives a unique token and a resource;
    the token is used to release that resource, which
    is guaranteed to only happen once.

    @effect(Resource)
    def build_report(
        header_file: Path | str,
        body_file: Path | str,
        footer_file: Path | str,
        output_file: Path | str,
        process_line: Callable[[str], str] = identity
    ):
        out_token, output = yield open_file(output_file, "w")
        hdr_token, header = yield open_file(header_file, "r")

        line_count = 0
        for line in header:
            output.write(line)
            line_count += 1
        yield release(hdr_token)

        bdy_token, body = yield open_file(body_file, "r")
        for line in body:
            output.write(process_line(line))
            line_count += 1
        yield release(bdy_token)

        _, footer = yield open_file(footer_file, "r")
        for line in footer:
            output.write(line)
            line_count += 1

        return line_count

    We can use this as

    build_report("/docs/hdr", "/docs/body", "/docs/ftr", "/doc/report", lambda s: "| " + s)

    creates the report file in "/docs/report".

    This maintains a pool of registered resources that are allocated
    within the monad's region. The resources can be used and will
    automatically clean up on exit of the region or if there is
    exception. The resources can also be released early by the user.
    Resources can be interleaved and freed.


    This plays a similar role as contextlib.ExitStack in Python,
    and Resource[A] is analogous to  ResourceT IO a in Haskell.

    """
    def __init__(self, action: Callable[[ResourceRegistry], IO[A]]):
        self._action = action

    def map[B](self, g: Callable[[A], B]) -> Resource[B]:
        def _mapped(r: ResourceRegistry):
            return map(g, self._action(r))

        return Resource[B](_mapped)

    @classmethod
    def pure(cls, a: A) -> Resource[A]:
        return cls(const(IO.pure(a)))

    def map2[B, C](self, g: Callable[[A, B], C], fb: Resource[B]) -> Resource[C]:
        def _mapped2(r: ResourceRegistry):
            ioa = self._action(r)
            iob = fb._action(r)
            return map2(g, ioa, iob)

        return Resource[C](_mapped2)

    def bind[B](self, a_to_r_b: Callable[[A], Resource[B]]) -> Resource[B]:
        """Implements the Monad trait."""
        return Resource(lambda r: self._action(r).bind(lambda a: a_to_r_b(a)._action(r)))

    @classmethod
    def __do__(cls, make_generator, is_generator) -> Resource[A]:
        """Implements do-notation for Resource"""

        # IO instances that  are yielded or returned in do are handled.
        def ensure_io(obj, r: ResourceRegistry) -> IO:
            if isinstance(obj, IO):
                return obj
            if isinstance(obj, Resource):
                return obj._action(r)
            mesg = f'Only a Resource or IO should be yielded/returned in @do(Resource), got {type(obj).__name__}'
            raise TypeError(mesg)

        def finalize(result, r: ResourceRegistry):
            if isinstance(result, (IO, Resource)):
                return IO.unsafe_run(ensure_io(result, r))
            return result

        def action(r: ResourceRegistry) -> IO[A]:
            def code():
                if not is_generator:
                    return finalize(make_generator(), r)

                generator = make_generator()
                try:
                    x = generator.send(None)
                    while True:
                        try:
                            value = IO.unsafe_run(ensure_io(x, r))
                        except BaseException as e:
                            x = generator.throw(e)
                        else:
                            x = generator.send(value)
                except StopIteration as finished:
                    return finalize(finished.value, r)

            return IO(code)

        return cls(action)

    def _close_and_release_all(self, r: ResourceRegistry, why: ReleaseReason):
        errors = []
        # We release resources in reverse insertion order
        # Gather errors raised during cleanup for the end
        remaining = r.drain()
        for free in remaining:
            try:
                IO.unsafe_run(free(why))
            except Exception as e:
                errors.append(e)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup('Errors encountered when releasing resources', errors)

    def run(self) -> IO[A]:
        def run_resource():
            r = ResourceRegistry()

            try:
                a = IO.unsafe_run(self._action(r))
            except BaseException as e:
                self._close_and_release_all(r, AfterError(e))
                raise

            self._close_and_release_all(r, AtCleanup())
            return a

        return IO(run_resource)

    @classmethod
    def unsafe_run(cls, resource: Resource[A]) -> A:
        return IO.unsafe_run(resource.run())

    @classmethod
    def acquire_with[B](cls, allocate: IO[B], free: Callable[[B, ReleaseReason], IO[Unit]]) -> Resource[tuple[ReleaseToken, B]]:
        def _acquire(r: ResourceRegistry) -> IO[tuple[ReleaseToken, B]]:
            def register() -> tuple[ReleaseToken, B]:
                resource = IO.unsafe_run(allocate)
                token = r.register(lambda why: free(resource, why))

                if token is None:
                    e = KeyError('Attempt to register a resource in a closed registry')
                    IO.unsafe_run(free(resource, AfterError(e)))
                    raise e

                return (token, resource)

            return IO(register)

        return Resource(_acquire)

    @classmethod
    def acquire[B](cls, allocate: IO[B], free: Callable[[B], IO[Unit]]) -> Resource[tuple[ReleaseToken, B]]:
        return cls.acquire_with(allocate, lambda b, _why: free(b))

    @classmethod
    def release(cls, token: ReleaseToken) -> IO[Unit]:
        def release_resource():
            free = token.registry.deregister(token)
            if free is not None:    # Do nothing if already released
                IO.unsafe_run(free(EarlyRelease()))
            return ()

        return IO(release_resource)

    @classmethod
    def detach(cls, token: ReleaseToken) -> IO[Callable[[ReleaseReason], IO[Unit]] | None]:
        """Removes a resource from its context without executing the free action.

        Returns the resource's free action, or None if the resource was already released.

        Executing the free action is **now the caller's responsibility**.

        A use case for this is when deciding dynamically that an acquired
        resource should be preserved or passed to another Resource.
        See also `attach`.

        """
        return IO(lambda: token.registry.deregister(token))

    @classmethod
    def attach(cls, free: Callable[[ReleaseReason], IO[Unit]]) -> Resource[ReleaseToken]:
        """Registers the free action for an existing resource, returning a unique release token.

        This can be used when transferring responsibility for an existing
        resource to the current Resource context.

        Attempting to attach a resource to a closed registry
        will run the free action and raise an error.

        """
        def _attach(r: ResourceRegistry) -> IO[ReleaseToken]:
            def register() -> ReleaseToken:
                token = r.register(free)
                if token is None:
                    e = RuntimeError('Attempt to attach a resource to a closed registry')
                    IO.unsafe_run(free(AfterError(e)))
                    raise e
                return token
            return IO(register)
        return Resource(_attach)

def acquire_with[A](allocate: IO[A], free: Callable[[A, ReleaseReason], IO[Unit]]) -> Resource[tuple[ReleaseToken, A]]:
    """Allocates a resource and registers its freeing action.

    The freeing action takes the value of the resource and a reason for the release.
    Returns a unique token for the resource that can be used with `release` and the value,
    in a Resource context.

    See also `acquire`.

    """
    return Resource.acquire_with(allocate, free)

def acquire[A](allocate: IO[A], free: Callable[[A], IO[Unit]]) -> Resource[tuple[ReleaseToken, A]]:
    """Like `acquire_with` but freeing action only takes the resource, not a release reason."""
    return Resource.acquire(allocate, free)

def acquire_cm[B](make_cm: Callable[[], AbstractContextManager[B]]) -> Resource[tuple[ReleaseToken, B]]:
    """Acquires a context manager's value as a Resource.

    This takes a *function that constructs a context manager* rather than a context manager itself.
    The context manager is created and entered only when the Resource runs.

    The primary purpose of this function is to avoid duplicating
    cleanup code that is already handled from pre-defined context
    managers. As a common example, we do not need to supply an
    explicit close action for files and streams.

    Examples:

      @do(Resource)
      def action():
          token1, f = yield acquire_cm(lambda: open(path))
          token2, my_lock = yield acquire_cm(lambda: lock)

    If an exception causes release, __exit__ receives the exception.
    Otherwise, on release, __exit__ receives None's. Unlike a with
    statement, a true return from __exit__ does not suppress the
    exception.

    Returns a token identifying the resource and the resource, in a
    Resource context. See also `acquire_with`.

    """
    def enter() -> tuple[AbstractContextManager[B], B]:
        cm = make_cm()
        return (cm, type(cm).__enter__(cm))   # as in `with`, lookup value on the type

    def exit_(cm_value: tuple[AbstractContextManager[B], B], why: ReleaseReason) -> IO[Unit]:
        cm, _ = cm_value
        def code():
            match why:
                case AfterError(e):
                    type(cm).__exit__(cm, type(e), e, e.__traceback__)
                case _:
                    type(cm).__exit__(cm, None, None, None)
            return ()
        return IO(code)

    # We store the token and the pair (cm, value). The exit_ needs the cm only
    # but we return only the (token, value) pair to the caller.
    acquired = acquire_with(IO(enter), exit_)
    return acquired.map(lambda token_cmval: (token_cmval[0], token_cmval[1][1]))

def release(token: ReleaseToken) -> IO[Unit]:
    """Releases a resource as an IO action."""
    return Resource.release(token)

def detach(token: ReleaseToken) -> IO[Callable[[ReleaseReason], IO[Unit]] | None]:
    """Removes a resource from its context without executing the free action.

    Returns the resource's free action, or None if the resource was already released.
    Executing the free action is **now the caller's responsibility**.

    """
    return Resource.detach(token)

def attach(free: Callable[[ReleaseReason], IO[Unit]]) -> Resource[ReleaseToken]:
    """Registers the free action for an existing resource, returning a unique release token.

    This can be used when transferring responsibility for an existing
    resource to the current Resource context.

    Attempting to attach a resource to a closed registry
    will run the free action and raise an error.

    """
    return Resource.attach(free)

def lift_io[A](ia: IO[A]) -> Resource[A]:
    """Lifts an IO action into the Resource monad."""
    return Resource(const(ia))

def run_in_io[A](resource: Resource[A]) -> IO[A]:
    """Runs a resource and then its IO action immediately.

    Returns the pure value of the action in an IO context.
    Rerunning the resulting IO value will *not* re-run the effect;
    it will just produce the value.

    In contrast, resource.run(), produces the effectful IO action
    from this resource. Resource.unsafe_run() runs the resource
    and its IO action and returns the unwrapped value.

    This is analogous to the pseudo-Haskell

        main :: IO a
        main = do a <- resource.run; pure a

    where the resource's action is run immediately.

    """
    return IO.run(resource.run())


#
# Resource Utilities
#

if TYPE_CHECKING:
    from _typeshed import (FileDescriptorOrPath, OpenBinaryMode, OpenBinaryModeReading,
                           OpenBinaryModeUpdating, OpenBinaryModeWriting, OpenTextMode,
                           StrPath)

# Opening Files

@overload          # Text Mode
def open_file(
    file: FileDescriptorOrPath,
    mode: OpenTextMode = 'r',
    buffering: int = -1,
    encoding: str | None = None,
    *,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener: Callable | None = None
) -> Resource[tuple[ReleaseToken, pio.TextIOWrapper]]:
    ...

@overload          # Unbuffered binary
def open_file(
    file: FileDescriptorOrPath,
    mode: OpenBinaryMode,
    buffering: Literal[0],
    encoding: Literal[None] = None,
    *,
    errors: Literal[None] = None,
    newline: Literal[None] = None,
    closefd: bool = True,
    opener: Callable | None = None
) -> Resource[tuple[ReleaseToken, pio.FileIO]]:
    ...

@overload          # Buffered binary, mixed read/write
def open_file(
    file: FileDescriptorOrPath,
    mode: OpenBinaryModeUpdating,
    buffering: Literal[-1, 1] = -1,
    encoding: Literal[None] = None,
    *,
    errors: Literal[None] = None,
    newline: Literal[None] = None,
    closefd: bool = True,
    opener: Callable | None = None
) -> Resource[tuple[ReleaseToken, pio.BufferedRandom]]:
    ...

@overload          # Buffered binary, writing or appending
def open_file(
    file: FileDescriptorOrPath,
    mode: OpenBinaryModeWriting,
    buffering: Literal[-1, 1] = -1,
    encoding: Literal[None] = None,
    *,
    errors: Literal[None] = None,
    newline: Literal[None] = None,
    closefd: bool = True,
    opener: Callable | None = None
) -> Resource[tuple[ReleaseToken, pio.BufferedWriter]]:
    ...

@overload          # Buffered binary, reading
def open_file(
    file: FileDescriptorOrPath,
    mode: OpenBinaryModeReading,
    buffering: Literal[-1, 1] = -1,
    encoding: Literal[None] = None,
    *,
    errors: Literal[None] = None,
    newline: Literal[None] = None,
    closefd: bool = True,
    opener: Callable | None = None
) -> Resource[tuple[ReleaseToken, pio.BufferedReader]]:
    ...

@overload          # General buffered binary
def open_file(
    file: FileDescriptorOrPath,
    mode: OpenBinaryMode,
    buffering: int = -1,
    encoding: Literal[None] = None,
    *,
    errors: Literal[None] = None,
    newline: Literal[None] = None,
    closefd: bool = True,
    opener: Callable | None = None
) -> Resource[tuple[ReleaseToken, BinaryIO]]:
    ...

@overload          # Mode not known statically
def open_file(
        file: FileDescriptorOrPath,
        mode: str,
        buffering: int = -1,
        encoding: str | None = None,
        *,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Callable | None = None
) -> Resource[tuple[ReleaseToken, typing.IO[Any]]]:
    ...

def open_file(
    file,
    mode='r',
    buffering=-1,
    encoding=None,
    *,
    errors=None,
    newline=None,
    closefd=True,
    opener=None
) -> Resource[tuple[ReleaseToken, Any]]:
    return acquire_cm(lambda: open(file, mode=mode, buffering=buffering, encoding=encoding,
                                   errors=errors, newline=newline, closefd=closefd, opener=opener))


# Temporary Files

@overload
def temp_file(
        mode: OpenTextMode,
        buffering: int = -1,
        encoding: str | None = None,
        *,
        named: bool = False,
        spooled: bool = False,
        newline: str | None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: StrPath | None = None,
        delete: bool = True,
        max_size: int = 0,
        delete_on_close: bool = True,
        errors: str | None = None
) -> Resource[tuple[ReleaseToken, typing.IO[str]]]:
    ...
@overload
def temp_file(
        mode: OpenBinaryMode = 'w+b',
        buffering: int = -1,
        encoding: None = None,
        *,
        named: bool = False,
        spooled: bool = False,
        newline: None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: StrPath | None = None,
        delete: bool = True,
        max_size: int = 0,
        delete_on_close: bool = True,
        errors: None = None
) -> Resource[tuple[ReleaseToken, typing.IO[bytes]]]:
    ...

@overload
def temp_file(
        mode: str = 'w+b',
        buffering: int = -1,
        encoding: None = None,
        *,
        named: bool = False,
        spooled: bool = False,
        newline: None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: StrPath | None = None,
        delete: bool = True,
        max_size: int = 0,
        delete_on_close: bool = True,
        errors: None = None
) -> Resource[tuple[ReleaseToken, typing.IO[Any]]]:
    ...

def temp_file(
        mode='w+b',
        buffering=-1,
        encoding=None,
        *,
        named=False,
        spooled=False,
        newline=None,
        suffix=None,
        prefix=None,
        dir=None,             # noqa: A002
        delete=True,
        max_size=0,
        delete_on_close=True,
        errors=None
) -> Resource[tuple[ReleaseToken, Any]]:
    if named:
        def alloc():
            return tempfile.NamedTemporaryFile(mode=mode, buffering=buffering, encoding=encoding, newline=newline,
                                               suffix=suffix, prefix=prefix, dir=dir, delete=delete, errors=errors,
                                               delete_on_close=delete_on_close)
    elif spooled:
        def alloc():
            return tempfile.SpooledTemporaryFile(max_size=max_size, mode=mode, buffering=buffering, encoding=encoding,
                                                 newline=newline, suffix=suffix, prefix=prefix, dir=dir, errors=errors)
    else:
        def alloc():
            return tempfile.TemporaryFile(mode=mode, buffering=buffering, encoding=encoding,
                                          newline=newline, suffix=suffix, prefix=prefix, dir=dir, errors=errors)

    return acquire_cm(alloc)
