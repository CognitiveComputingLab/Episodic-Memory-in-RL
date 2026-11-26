from abc import abstractmethod

from torch import Tensor


class TrackableState():
    @abstractmethod
    def track_state() -> list[Tensor]:
        pass