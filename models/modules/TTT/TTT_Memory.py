import torch
import torch.nn as nn
from torch import Tensor, vmap
from einops import rearrange

from tensordict import TensorDict
from torch.func import functional_call, grad

from typing import Optional, cast

from models import Stateful
from models.modules.TTT import MemoryModule

TTT_MemoryState = Optional[TensorDict]

class TTT_Memory(MemoryModule, Stateful[TTT_MemoryState]):

    def __init__(self, lr: float, memoryModule: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.memoryModule = memoryModule    # 2-layered MLP by default
        self.state: Optional[TensorDict] = None
        self.lr_global = lr
        

    def _read(self, params, q: Tensor):
        return functional_call(self.memoryModule, dict(params), (q,))

    def forward(self, q: Tensor):   # Read
        assert self.state is not None
        return vmap(self._read)(self.state, q)


    def _write(self, params, k: Tensor, v: Tensor, lr: Tensor):
        v_pred = functional_call(self.memoryModule, dict(params), (k,))
        return (lr*((v - v_pred) ** 2).mean(-1, keepdim=True)).mean()   # 2nd mean is where assoc scan?

    @torch.compiler.disable
    def write(self, x: Tensor, k: Tensor, v: Tensor, lr: Tensor):
        grads = vmap(grad(self._write))(self.state, k, v, lr)   # Predict v from k
        self.state = (self.state - self.lr_global * grads)      # Update memory


    def _fresh_state(self, n: int) -> TensorDict:
        params = cast(TensorDict, TensorDict.from_module(self.memoryModule))    # Create a fresh copy of the memory module
        return params.expand(n, *params.shape).clone().flatten_keys(".")        # Each batch needs a different copy

    def reset_state(self, batch: int) -> None:
        self.state = self._fresh_state(batch)

    def reset_state_at(self, indices: list[int]) -> None:
        assert self.state is not None
        fresh = self._fresh_state(len(indices))
        idx = torch.as_tensor(list(indices), dtype=torch.long, device=self.state.device)

        # Create a new state to avoid autograd issues - claude
        self.state = TensorDict(    
            {name: t.index_copy(0, idx, fresh[name]) for name, t in self.state.items()},
            batch_size=self.state.batch_size,
        )

    def get_state(self) -> TTT_MemoryState:
        return self.state

    def set_state(self, state: TTT_MemoryState):
        self.state = state

    def detach_state(self) -> None:
        if self.state is not None:
            self.state = self.state.detach()
