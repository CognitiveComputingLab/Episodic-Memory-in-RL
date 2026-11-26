from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrainingBatch:
    states: Tensor
    actions: Tensor
    returns: Tensor
    timesteps: Tensor
    mask: Tensor

    @classmethod
    def from_collated(cls, batch: Iterable[Tensor]) -> "TrainingBatch":
        values = tuple(batch)
        if len(values) != 5:
            raise ValueError(
                f"Training batch must contain 5 tensors, got {len(values)}"
            )
        states, actions, returns, timesteps, mask = values
        return cls(states, actions, returns, timesteps, mask)


@dataclass(frozen=True)
class StepSpec:
    start: int
    end: int
    detach_state: bool = False


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    primary_loss: float


class TrainingExtension(ABC):
    """Optional feature that augments a base training loop."""

    @abstractmethod
    def attach(self, trainer: "BaseTrainer") -> None:
        pass

    @abstractmethod
    def before_batch(self, trainer: "BaseTrainer") -> None:
        pass

    @abstractmethod
    def after_batch(self, trainer: "BaseTrainer", primary_loss: float) -> None:
        pass

    @abstractmethod
    def checkpoint_state(self) -> dict[str, object]:
        pass

    @abstractmethod
    def restore_checkpoint(self, state: dict[str, object]) -> None:
        pass


class OptimizationPolicy:
    """Owns the base pipeline's primary-loss backward pass and gradient clipping."""

    def __init__(self, optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def step(
        self,
        losses: Mapping[str, Tensor],
        parameters: Iterable[torch.nn.Parameter],
    ) -> float:
        if not losses:
            raise ValueError("At least one loss is required")
        primary = losses["primary"]
        if not torch.isfinite(primary):
            raise FloatingPointError("Primary loss is non-finite")

        self.optimizer.zero_grad()
        primary.backward()

        parameter_list = list(parameters)
        if not parameter_list:
            raise ValueError("Cannot clip/step with no model parameters")
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameter_list, self.grad_clip)
        self.optimizer.step()
        return float(primary.detach().item())

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def backward(self, loss: Tensor) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss is non-finite")
        loss.backward()

    def apply(self, parameters: Iterable[torch.nn.Parameter], reported_loss: Tensor) -> float:
        parameter_list = list(parameters)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameter_list, self.grad_clip)
        self.optimizer.step()
        return float(reported_loss.item())
