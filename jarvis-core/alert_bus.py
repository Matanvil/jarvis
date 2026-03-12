"""Thread-safe alert bus for pushing notifications to connected Swift clients.

The asyncio event loop is set once by server.py at startup.
push() is safe to call from any thread (e.g. APScheduler background threads).
"""
import asyncio
import threading
from typing import List

_subscribers: List[asyncio.Queue] = []
_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def subscribe(q: asyncio.Queue) -> None:
    with _lock:
        _subscribers.append(q)


def unsubscribe(q: asyncio.Queue) -> None:
    with _lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def push(title: str, body: str) -> None:
    """Enqueue an alert for all connected Swift clients. Thread-safe."""
    event = {"type": "alert", "title": title, "body": body}
    with _lock:
        subs = list(_subscribers)
    if not subs or _loop is None:
        return
    for q in subs:
        _loop.call_soon_threadsafe(q.put_nowait, event)
