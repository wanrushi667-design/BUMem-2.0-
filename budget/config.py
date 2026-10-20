"""Fixed budget for one character's full-budget run."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class BudgetConfig:
    baseline_tokens: int
    ratio: float = 0.9
    batch_size: int = 10

    def __post_init__(self):
        if self.baseline_tokens < 0 or not 0 < self.ratio <= 1:
            raise ValueError("Expected nonnegative baseline_tokens and 0 < ratio <= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @property
    def budget_tokens(self) -> int:
        return int(Decimal(self.baseline_tokens) * Decimal(str(self.ratio)))
