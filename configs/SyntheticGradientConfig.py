from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticGradientConfig:
    enabled: bool = False
    horizon: int = 1
    bootstrapping_discount: float = 1.0
    loss_weight: float = 1.0

    def validate(self) -> None:
        if self.enabled and self.horizon < 2:
            raise ValueError(
                "Synthetic-gradient training requires horizon >= 2"
            )
        if not 0.0 <= self.bootstrapping_discount <= 1.0:
            raise ValueError("bootstrapping_discount must be in [0, 1]")
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative")
