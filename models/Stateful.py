from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')

class Stateful(ABC, Generic[T]):

    @abstractmethod
    def reset_state(self, batch: int) -> None:
        pass

    @abstractmethod
    def reset_state_at(self, indices: list[int]) -> None:
        pass

    @abstractmethod
    def get_state(self) -> T:
        pass

    @abstractmethod
    def set_state(self, state: T) -> None:
        pass

    @abstractmethod
    def detach_state(self) -> None:
        pass