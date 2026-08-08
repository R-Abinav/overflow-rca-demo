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
    def __init__(self, max_connections=50):
        self.max_connections = max_connections
        self.active_connections = 0
        self.queue = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()

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
        """
        try:
            conn = self.queue.get(block=True, timeout=timeout)
        except queue.Empty:
            logger.error(
                "timeout acquiring connection from pool after %dms (max_wait=%dms)",
                timeout * 1000, timeout * 1000,
            )
            raise TimeoutError(
                f"timeout acquiring connection from pool after {timeout * 1000}ms"
            )

        # Only count a connection as "active" once it has actually been
        # checked out of the pool. Counting threads that are merely
        # waiting on the queue as active inflates utilization and can
        # cause the pool to be falsely reported as EXHAUSTED, triggering
        # unnecessary retries/backlog growth downstream.
        with self._lock:
            self.active_connections += 1
        return conn

    def release_connection(self, conn):
        with self._lock:
            self.active_connections -= 1
        self.queue.put(conn)

    def status(self):
        with self._lock:
            active = self.active_connections
        return {
            "pool_size": self.max_connections,
            "active": active,
            "idle": self.max_connections - active,
            "utilization": int((active / self.max_connections) * 100),
        }
