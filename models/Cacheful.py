from abc import ABC, abstractmethod

class Cacheful(ABC):
    """Interface for modules that maintain intra-chunk information to speed up sampling, e.g. attention.
    Inputs shorter than the model's context window should trigger cache use.
    Inputs of the context window size should not trigger cache."""

    use_cache = False

    @abstractmethod
    def reset_cache(self, batch: int) -> None:
        pass

    @abstractmethod
    def wipe_cache(self) -> None:
        pass

    def enable_cache(self, use_cache: bool):
        self.use_cache = use_cache