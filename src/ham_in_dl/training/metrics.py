"""Metric helpers used during training."""

from dataclasses import dataclass, field

import torch
from sklearn.metrics import f1_score


@dataclass
class RunningClassificationMetrics:
    """Accumulate loss, predictions, and targets for one epoch."""

    total_loss: float = 0.0
    total_examples: int = 0
    predictions: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)

    def update(self, loss: float, logits: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = targets.size(0)
        self.total_loss += float(loss) * batch_size
        self.total_examples += batch_size
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        self.predictions.extend(int(x) for x in preds)
        self.targets.extend(int(x) for x in targets.detach().cpu().tolist())

    def compute(self) -> dict[str, float]:
        if self.total_examples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}
        correct = sum(p == t for p, t in zip(self.predictions, self.targets))
        return {
            "loss": self.total_loss / self.total_examples,
            "accuracy": correct / self.total_examples,
            "macro_f1": f1_score(
                self.targets,
                self.predictions,
                average="macro",
                zero_division=0,
            ),
        }


def update_running_metrics(
    running: RunningClassificationMetrics,
    loss: float,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Compatibility wrapper for updating a running metric accumulator."""
    running.update(loss, logits, targets)
