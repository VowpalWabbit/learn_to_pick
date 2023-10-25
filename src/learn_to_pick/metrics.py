from collections import deque
from typing import TYPE_CHECKING, Dict, List, Union

if TYPE_CHECKING:
    import pandas as pd


class MetricsTrackerAverage:
    def __init__(self, step: int):
        self.history: List[Dict[str, Union[int, float]]] = [{"step": 0, "score": 0}]
        self.step: int = step
        self.feedback_count: int = 0
        self.score_sum: float = 0
        self.decision_count: float = 0

    @property
    def score(self) -> float:
        return self.score_sum / self.decision_count if self.decision_count > 0 else 0

    def on_decision(self) -> None:
        self.decision_count += 1

    def on_feedback(self, score: float) -> None:
        self.score_sum += score or 0
        self.feedback_count += 1
        if self.step > 0 and self.feedback_count % self.step == 0:
            self.history.append({"step": self.feedback_count, "score": self.score})

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(self.history)


class MetricsTrackerRollingWindow:
    def __init__(self, window_size: int, step: int):
        self.history: List[Dict[str, Union[int, float]]] = [{"step": 0, "score": 0}]
        self.step: int = step
        self.feedback_count: int = 0
        self.window_size: int = window_size
        self.queue: deque = deque()
        self.sum: float = 0.0

    @property
    def score(self) -> float:
        return self.sum / len(self.queue) if len(self.queue) > 0 else 0

    def on_decision(self) -> None:
        pass

    def on_feedback(self, value: float) -> None:
        self.sum += value
        self.queue.append(value)
        self.feedback_count += 1

        if len(self.queue) > self.window_size:
            old_val = self.queue.popleft()
            self.sum -= old_val

        if self.step > 0 and self.feedback_count % self.step == 0:
            self.history.append(
                {"step": self.feedback_count, "score": self.sum / len(self.queue)}
            )

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(self.history)
