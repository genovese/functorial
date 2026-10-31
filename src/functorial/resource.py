"""Resource in IO Monad

Resource[A] is analogous to ResourceT IO a in Haskell

"""
from __future__      import annotations

from collections.abc import Callable
from contextlib      import AbstractContextManager
from dataclasses     import dataclass
from threading       import Lock

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
    'EarlyRelease',
    'lift_io',
    'release',
    'ReleaseReason',
    'Resource',
    'run_in_io',
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

    This maintains a pool of registered resources that are allocated
    within the monad's region. The resources can be used and will
    automatically clean up on exit of the region or if there is
    exception. The resources can also be released early by the user.
    Resources can be interleaved and freed.

    A resource object can be run to produce an IO action or
    run in IO (see `run_in_io`) to produce a value and its
    side effects in an IO context.

    This plays a similar role as contextlib.ExitStack in Python.

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

def lift_io[A](ia: IO[A]) -> Resource[A]:
    """Lifts an IO action into the Resource monad."""
    return Resource(const(ia))

def run_in_io[A](resource: Resource[A]) -> IO[A]:
    """Runs a resource fully in an IO context.

    Specifically, the resource is run and then
    the returned IO action is run as well.

    """
    return IO.run(resource.run())
