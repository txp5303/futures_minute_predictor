from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class ConfirmResult:
    ok: bool
    count: int
    reason: str = ""


class SignalConfirmer:
    """
    3选2确认器（窗口确认）
    """
    def __init__(self, n: int = 3, k: int = 2):
        self.n = int(n)
        self.k = int(k)
        if self.n <= 0:
            raise ValueError("n must be > 0")
        if self.k <= 0 or self.k > self.n:
            raise ValueError("k must be in [1, n]")
        self.buf: Deque[bool] = deque(maxlen=self.n)

    def reset(self):
        self.buf.clear()

    def push(self, is_candidate: bool) -> ConfirmResult:
        self.buf.append(bool(is_candidate))
        cnt_true = int(sum(1 for x in self.buf if x))

        if len(self.buf) < self.n:
            return ConfirmResult(False, cnt_true, f"warming_up {len(self.buf)}/{self.n}")

        if cnt_true >= self.k:
            self.reset()
            return ConfirmResult(True, cnt_true, f"confirmed {self.k}/{self.n}")

        return ConfirmResult(False, cnt_true, f"window {cnt_true}/{self.n} (<{self.k})")