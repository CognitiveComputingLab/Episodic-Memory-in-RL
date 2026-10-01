import torch
import torch.nn as nn
from torch import Tensor, vmap
from einops import rearrange

from tensordict import TensorDict
from torch.func import functional_call, grad

from typing import Optional, cast

from models import Stateful
from models.modules.TTT import TTT_Module

class TitansMemory(TTT_Module, Stateful[tuple[Optional[TensorDict],Optional[TensorDict]]]):

    def __init__(self, e_dim: int, memoryModule: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.memoryModule = memoryModule    # 2-layered MLP by default
        self.memory_state: Optional[TensorDict] = None
        self.momentum_state: Optional[TensorDict] = None

        self.kv_proj = nn.Linear(e_dim, 2 * e_dim)
        self.q_proj = nn.Linear(e_dim, e_dim)
        self.lr_proj = nn.Sequential(nn.Linear(e_dim, 1), nn.Sigmoid())
        self.m_retain_proj = nn.Sequential(nn.Linear(e_dim, 1), nn.Sigmoid())   # memoy retention gate
        self.s_retain_proj = nn.Sequential(nn.Linear(e_dim, 1), nn.Sigmoid())   # momentum retention gate
        

    def _read(self, params, q: Tensor):
        return functional_call(self.memoryModule, dict(params), (q,))

    def forward(self, x: Tensor):   # Read
        assert self.memory_state is not None
        q = self.q_proj(x)
        return vmap(self._read)(self.memory_state, q)   # Query the memoy

    def _write(self, params, k: Tensor, v: Tensor, lr: Tensor):
        v_pred = functional_call(self.memoryModule, dict(params), (k,))
        return (lr*((v - v_pred) ** 2).mean(-1, keepdim=True)).mean()

    def write(self, x: Tensor):
        assert self.memory_state is not None
        kv = self.kv_proj(x)    # Get kv from x
        k, v = rearrange(kv, "b ... (p e) -> p b ... e", p=2)
        lr = self.lr_proj(x)
        m_retain = self.m_retain_proj(x).mean(dim=1).squeeze(-1)        # Calculate a global decay by meaning from each token (!)
        s_retain = self.s_retain_proj(x).mean(dim=1).squeeze(-1)        # TO-DO: update these to calculate value based on chunk

        grads = vmap(grad(self._write))(self.memory_state, k, v, lr)            # Predict v from k
        self.momentum_state = s_retain * self.momentum_state - grads            # Update momentum
        self.memory_state = m_retain * self.memory_state + self.momentum_state  # Update memory


    def _fresh_state(self, n: int) -> tuple[TensorDict, TensorDict]:
        params = cast(TensorDict, TensorDict.from_module(self.memoryModule))    # Create a fresh copy of the memory module
        memory = params.expand(n, *params.shape).clone().flatten_keys(".")      # Each batch needs a different copy
        momentum = TensorDict.new_zeros(memory)
        return memory, momentum

    def reset_state(self, batch: int) -> None:
        self.memory_state, self.momentum_state = self._fresh_state(batch)

    def reset_state_at(self, indices: list[int]) -> None:
        assert self.memory_state is not None and self.momentum_state is not None
        fresh_mem, fresh_momem = self._fresh_state(len(indices))
        idx = torch.as_tensor(list(indices), dtype=torch.long, device=self.q_proj.weight.device)

        # Create a new state to avoid autograd issues - claude
        self.memory_state = TensorDict(    
            {name: t.index_copy(0, idx, fresh_mem[name]) for name, t in self.memory_state.items()},
            batch_size=self.memory_state.batch_size,
        )
        self.momentum_state = TensorDict(    
            {name: t.index_copy(0, idx, fresh_momem[name]) for name, t in self.momentum_state.items()},
            batch_size=self.momentum_state.batch_size,
        )

    def get_state(self) -> tuple[Optional[TensorDict],Optional[TensorDict]]:
        return self.memory_state, self.momentum_state

    def set_state(self, state: tuple[Optional[TensorDict],Optional[TensorDict]]):
        self.memory_state, self.momentum_state = state

    def detach_state(self) -> None:
        if self.memory_state is not None:
            assert self.momentum_state is not None
            self.memory_state = self.memory_state.detach()
            self.momentum_state = self.momentum_state.detach()
