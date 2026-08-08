```python
"""
Connection pool for the auth-service's backing database.

Provides bounded, thread-safe access to a fixed number of database
connections. Callers that request a connection while the pool is at
capacity are queued and block until either a connection is released
back to the pool or their own request times out.
"""

import queue
import threading
import logging

logger = logging.getLogger(__name__)


class ConnectionPool:
    def __init__(self, max_connections=50, max_waiters=None):
        self.max_connections = max_connections
        self.active_connections = 0
        self.queue = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()

        # Bound the number of callers that may be waiting for a connection
        # at any given time. Without this cap, a burst of incoming requests
        # during a pool-exhaustion event has no upper limit on how many
        # threads/requests can pile up blocked on the queue, each holding
        # memory (request state, connection wrapper allocations, etc.).
        # That unbounded pile-up is what drives GC pressure and, ultimately,
        # heap exhaustion/OOM under sustained load. New callers beyond this
        # limit fail fast instead of queuing indefinitely.
        self._max_waiters = max_waiters if max_waiters is not None else max_connections * 2
        self._waiters_semaphore = threading.BoundedSemaphore(self._max_waiters)

        for _ in range(max_connections):
            self.queue.put(self._open_connection())

    def _open_connection(self):
        # Opens a connection to the backing database. Stubbed here since
        # the driver setup is environment-specific.
        raise NotImplementedError

    def get_connection(self, timeout=5):
        """
        Acquire a connection from the pool.

        If a connection is immediately available it is returned right
        away. Otherwise the caller blocks on the internal queue for up
        to `timeout` seconds waiting for one to be released.

        If too many callers are already waiting for a connection, the
        request fails immediately rather than joining an unbounded queue
        of waiters, which protects the process from unbounded memory
        growth during sustained pool exhaustion.
        """
        if not self._waiters_semaphore.acquire(blocking=False):
            logger.error(
                "connection pool waiter limit reached (max_waiters=%d); "
                "rejecting request immediately instead of queuing",
                self._max_waiters,
            )
            raise TimeoutError(
                "connection pool waiter limit reached; rejecting request immediately"
            )

        try:
            with self._lock:
                self.active_connections += 1
            try:
                return self.queue.get(block=True, timeout=timeout)
            except queue.Empty:
                with self._lock:
                    self.active_connections -= 1
                logger.error(
                    "timeout acquiring connection from pool after %dms (max_wait=%dms)",
                    timeout * 1000, timeout * 1000,
                )
                raise TimeoutError(
                    f"timeout acquiring connection from pool after {timeout * 1000}ms"
                )
        finally:
            self._waiters_semaphore.release()

    def release_connection(self, conn):
        with self._lock:
            self.active_connections -= 1
        self.queue.put(conn)

    def status(self):
        with self._lock:
            active = self.active
