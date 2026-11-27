from trainers.TrainingContracts import TrainingExtension


class SyntheticGradientExtension(TrainingExtension):
    """Opt-in feature hook for synthetic-gradient implementations.

    BaseTrainer does not import or depend on this class. A trainer or experiment
    attaches an implementation explicitly and uses its returned auxiliary losses
    and checkpoint state.
    """


    def attach(self, trainer: object) -> None:
        return

    def before_batch(self, trainer: object) -> None:
        return

    def after_batch(self, trainer: object, primary_loss: float) -> None:
        return

    def checkpoint_state(self) -> dict[str, object]:
        return None

    def restore_checkpoint(self, state: dict[str, object]) -> None:
        return None

    def log_step(self, trainer):
        from trainers.EpisodeTrainer import EpisodeSyntheticGradientTrainer
        if isinstance(trainer, EpisodeSyntheticGradientTrainer):
            return {'train/sg_loss': trainer.sg_loss}
        raise RuntimeError("Synethics gradient extension used on a non-sg trainer")