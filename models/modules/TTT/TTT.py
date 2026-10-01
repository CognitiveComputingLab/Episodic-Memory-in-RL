import torch
import torch.nn as nn
from torch import Tensor, vmap
from einops import rearrange

from tensordict import TensorDict
from torch.func import functional_call, grad

from typing import Optional, cast

from models import Stateful
from models.modules.TTT import TTT_Module

class TTT_Memory(TTT_Module, Stateful[Optional[TensorDict]]):

    def __init__(self, e_dim: int, memoryModule: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.memoryModule = memoryModule    # 2-layered MLP by default
        self.state: Optional[TensorDict] = None
        self.kv_proj = nn.Linear(e_dim, 2 * e_dim)
        self.q_proj = nn.Linear(e_dim, e_dim)
        self.lr_proj = nn.Sequential(nn.Linear(e_dim, 1), nn.Sigmoid())

    def _read(self, params, q: Tensor):
        return functional_call(self.memoryModule, dict(params), (q,))

    def forward(self, x: Tensor):   # Read
        assert self.state is not None
        q = self.q_proj(x)
        return vmap(self._read)(self.state, q)

    def _write(self, params, k: Tensor, v: Tensor, lr: Tensor):
        v_pred = functional_call(self.memoryModule, dict(params), (k,))
        return (lr*((v - v_pred) ** 2).mean(-1, keepdim=True)).mean()

    def write(self, x: Tensor):
        assert self.state is not None
        kv = self.kv_proj(x)    # Get kv from x
        k, v = rearrange(kv, "b ... (p e) -> p b ... e", p=2)
        lr = self.lr_proj(x)

        grads = vmap(grad(self._write))(self.state, k, v, lr)   # Predict v from k
        
        self.state = (self.state - grads)   # Update memory


    def _fresh_state(self, n: int) -> TensorDict:
        params = cast(TensorDict, TensorDict.from_module(self.memoryModule))    # Create a fresh copy of the memory module
        return params.expand(n, *params.shape).clone().flatten_keys(".")        # Each batch needs a different copy

    def reset_state(self, batch: int) -> None:
        self.state = self._fresh_state(batch)

    def reset_state_at(self, indices: list[int]) -> None:
        assert self.state is not None
        fresh = self._fresh_state(len(indices))
        idx = torch.as_tensor(list(indices), dtype=torch.long, device=self.q_proj.weight.device)

        # Create a new state to avoid autograd issues - claude
        self.state = TensorDict(    
            {name: t.index_copy(0, idx, fresh[name]) for name, t in self.state.items()},
            batch_size=self.state.batch_size,
        )

    def get_state(self) -> Optional[TensorDict]:
        return self.state

    def set_state(self, state: Optional[TensorDict]):
        self.state = state

    def detach_state(self) -> None:
        if self.state is not None:
            self.state = self.state.detach()
