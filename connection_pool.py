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


class PoolExhaustedError(Exception):
    """Raised when the pool has too many callers already waiting for a
    connection. This is a fail-fast signal so that callers do not pile up
    indefinitely (and allocate ever more resources) while the backing
    database is unavailable or overloaded."""
    pass


class ConnectionPool:
    def __init__(self, max_connections=50, max_queue_depth=None):
        self.max_connections = max_connections
        self.active_connections = 0
        self.queue = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()

        # Bound the number of callers allowed to wait for a connection at
        # once. Without this bound, a surge of traffic (e.g. a credential
        # brute-force or a legitimate spike) can cause an unbounded number
        # of threads to pile up waiting on the queue, each allocating
        # request/connection-wrapper objects that live until the wait
        # resolves. That unbounded allocation is what drives heap
        # exhaustion and OOM under sustained load. Reject new callers
        # immediately once the waiting count exceeds this threshold so
        # memory use stays bounded and failures happen fast instead of
        # cascading into a crash.
        self.max_queue_depth = (
            max_queue_depth if max_queue_depth is not None else max_connections * 2
        )
        self._waiting = 0

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
        request is rejected immediately rather than being queued, to
        avoid unbounded resource growth when the pool is exhausted.
        """
        with self._lock:
            if self._waiting >= self.max_queue_depth:
                logger.error(
                    "rejecting connection request: queue_depth=%d exceeds max_queue_depth=%d",
                    self._waiting, self.max_queue_depth,
                )
                raise PoolExhaustedError(
                    f"connection pool queue is full (queue_depth={self._waiting}, "
                    f"max_queue_depth={self.max_queue_depth})"
                )
            self._waiting += 1
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
            with self._lock:
                self._waiting -= 1

    def release_connection(self, conn):
        with self._lock:
            self.active_connections -= 1
        self.queue.put(conn)

    def status(self):
        with self._lock:
            active = self.active_connections
            waiting = self._waiting
        return {
            "pool_size": self.max_connections,
            "active": active,
            "idle": self.max_connections - active,
            "waiting": waiting,
            "utilization": int((active / self.max_connections) * 100),
        }
