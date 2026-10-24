"""Bounded single-flight caching for optional provider requests."""
from collections import OrderedDict
from copy import deepcopy
import threading
import time


class RequestCache:
    def __init__(self, limit=128):
        self.limit = limit
        self.values = OrderedDict()
        self.pending = set()
        self.condition = threading.Condition()

    def get_or_call(self, key, ttl, call):
        with self.condition:
            while key in self.pending:
                self.condition.wait()
            cached = self.values.get(key)
            if cached and cached[0] > time.monotonic():
                self.values.move_to_end(key)
                return deepcopy(cached[1])
            self.values.pop(key, None)
            self.pending.add(key)
        try:
            value = call()
            with self.condition:
                # A short negative cache prevents a provider outage from
                # generating a queue of identical slow retries.
                lifetime = ttl if value is not None else min(ttl, 10.0)
                self.values[key] = (time.monotonic() + lifetime, deepcopy(value))
                while len(self.values) > self.limit:
                    self.values.popitem(last=False)
            return value
        finally:
            with self.condition:
                self.pending.discard(key)
                self.condition.notify_all()
