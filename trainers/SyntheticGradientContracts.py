from abc import ABC, abstractmethod
from torch import Tensor
from configs.SyntheticGradientConfig import SyntheticGradientConfig
from trainers.TrainingContracts import TrainingExtension


class SyntheticGradientPredictor(ABC):
    """Predicts a future parameter gradient from an explicit context."""

    @abstractmethod
    def forward(self, context: Tensor, state_descriptor: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        pass


class SyntheticGradientExtension(TrainingExtension):
    """Opt-in feature hook for synthetic-gradient implementations.

    BaseTrainer does not import or depend on this class. A trainer or experiment
    attaches an implementation explicitly and uses its returned auxiliary losses
    and checkpoint state.
    """

    def __init__(
        self,
        config: SyntheticGradientConfig,
        predictor: SyntheticGradientPredictor,
    ) -> None:
        config.validate()
        self.config = config
        self.predictor = predictor

    def attach(self, trainer: object) -> None:
        del trainer
        if not self.config.enabled:
            return

    def before_batch(self, trainer: object) -> None:
        del trainer
        if not self.config.enabled:
            return

    def after_batch(self, trainer: object, primary_loss: float) -> None:
        del trainer, primary_loss
        if not self.config.enabled:
            return
        raise NotImplementedError(
            "Synthetic-gradient forward/bootstrapping is not implemented"
        )

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "horizon": self.config.horizon,
        }

    def restore_checkpoint(self, state: dict[str, object]) -> None:
        if bool(state.get("enabled", False)) != self.config.enabled:
            raise ValueError("Synthetic-gradient checkpoint configuration changed")


class DisabledSyntheticGradientPredictor(SyntheticGradientPredictor):
    def forward(self, context: Tensor, state_descriptor: Tensor) -> Tensor:
        raise RuntimeError("Synthetic gradients are disabled")

    def loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        raise RuntimeError("Synthetic gradients are disabled")
