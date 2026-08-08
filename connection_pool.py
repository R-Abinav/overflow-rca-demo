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


class PoolQueueFullError(Exception):
    """Raised when the backlog of callers waiting for a connection is full."""
    pass


class ConnectionPool:
    def __init__(self, max_connections=50, max_queue_depth=None):
        self.max_connections = max_connections
        self.active_connections = 0
        self.queue = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()

        # Bound the number of callers that may wait for a connection at
        # once. Without this, a surge of concurrent requests (e.g. during
        # a login flood) can pile up an unbounded number of blocked
        # waiters, each consuming memory/thread resources, eventually
        # exhausting the heap. Default to twice the pool size if not
        # explicitly configured.
        self.max_queue_depth = (
            max_queue_depth if max_queue_depth is not None else max_connections * 2
        )
        self._waiters = 0
        self._waiters_lock = threading.Lock()

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

        If the number of callers already waiting for a connection has
        reached `max_queue_depth`, the request is rejected immediately
        with PoolQueueFullError instead of being allowed to pile up
        indefinitely.
        """
        with self._waiters_lock:
            if self._waiters >= self.max_queue_depth:
                logger.error(
                    "rejecting connection request: queue_full (queue_depth=%d, max_queue_depth=%d)",
                    self._waiters, self.max_queue_depth,
                )
                raise PoolQueueFullError(
                    f"connection pool queue is full (queue_depth={self._waiters}, "
                    f"max_queue_depth={self.max_queue_depth})"
                )
            self._waiters += 1

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
            with self._waiters_lock:
                self._waiters -= 1

    def release_connection(self, conn):
        with self._lock:
            self.active_connections -= 1
        self.queue.put(conn)

    def status(self):
        with self._lock:
            active = self.active_connections
        with self._waiters_lock:
            queue_depth = self._waiters
        return {
            "pool_size": self.max_connections,
            "active": active,
            "idle": self.max_connections - active,
            "utilization": int((active / self.max_connections) * 100),
            "queue_depth": queue_depth,
        }
