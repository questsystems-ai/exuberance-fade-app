# monitoring.py
# Simple alert logger + per-provider rate-limit guard.

from __future__ import annotations
import os, time
from collections import deque
from typing import Optional

class AlertManager:
    def __init__(self, run_dir: str, echo: bool = True):
        self.run_dir = run_dir
        self.echo = echo
        os.makedirs(run_dir, exist_ok=True)
        self.log_fp = os.path.join(run_dir, "alerts.log")

    def _write(self, level: str, msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n"
        with open(self.log_fp, "a", encoding="utf-8") as f:
            f.write(line)
        if self.echo:
            print(line.strip())

    def info(self, msg: str):  self._write("INFO", msg)
    def warn(self, msg: str):  self._write("WARN", msg)
    def error(self, msg: str): self._write("ERROR", msg)

class RateLimitGuard:
    """Track approximate API calls/min and alert near/over plan limits."""
    def __init__(self, name: str, calls_per_min_limit: Optional[int], alert: Optional[AlertManager] = None, soft_margin: float = 0.9):
        self.name = name
        self.limit = calls_per_min_limit  # None disables checks
        self.alert = alert
        self.soft_margin = soft_margin
        self.times = deque()  # timestamps of recent calls (last 60s)

    def record(self, n_calls: int = 1):
        now = time.time()
        for _ in range(n_calls):
            self.times.append(now)
        self._prune(now)
        if not self.limit:
            return
        per_min = len(self.times)
        if per_min >= int(self.limit * self.soft_margin):
            if self.alert:
                self.alert.warn(f"{self.name}: approaching rate limit ({per_min}/{self.limit} calls in ~60s).")
        if per_min > self.limit:
            if self.alert:
                self.alert.error(f"{self.name}: exceeded rate limit ({per_min}/{self.limit} calls in ~60s).")

    def _prune(self, now: float):
        while self.times and now - self.times[0] > 60:
            self.times.popleft()
