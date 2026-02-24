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
    3选2确认器（窗口确认）：
    - 维护最近 n 次（分钟）的候选标记 True/False
    - 当窗口满 n 且 True 的数量 >= k 时，确认 ok=True
    - 触发后清空，避免刷屏
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
        """
        每分钟调用一次：
        - is_candidate=True 表示该分钟通过 Stage2（候选成立）
        - is_candidate=False 表示该分钟不成立（仍然计入窗口）
        """
        self.buf.append(bool(is_candidate))
        cnt_true = int(sum(1 for x in self.buf if x))

        if len(self.buf) < self.n:
            return ConfirmResult(False, cnt_true, f"warming_up {len(self.buf)}/{self.n}")

        # 窗口已满
        if cnt_true >= self.k:
            self.reset()
            return ConfirmResult(True, cnt_true, f"confirmed {self.k}/{self.n}")

        return ConfirmResult(False, cnt_true, f"window {cnt_true}/{self.n} (<{self.k})")