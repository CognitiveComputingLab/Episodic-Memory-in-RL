import math

import torch
import torch.nn as nn
from torch import Tensor, vmap
from einops import rearrange

from tensordict import TensorDict
from torch.func import functional_call, grad

from typing import Optional, cast

from models import Stateful
from models.modules.TTT import MemoryModule

FwPKM_MemoryState = tuple[Optional[Tensor], Optional[Tensor]]

class FwPKM_Memory(MemoryModule, Stateful[FwPKM_MemoryState]):

    def __init__(self, dim_in: int, dim_mem: int, lr: float, N: int, top_k: int, key_books: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n = round(N ** (1 / key_books))              # entries per codebook
        assert self.n ** key_books == N and dim_mem % key_books == 0

        self.N, self.dim_mem = N, dim_mem
        self.top_k, self.key_books = top_k, key_books

        # initial values for the fast weights; copy these into your per-stream state
        self.keys = nn.Parameter(torch.randn(key_books, self.n, dim_mem // key_books))   # [c, n, d/c]
        self.values = nn.Parameter(torch.randn(N, dim_mem))                              # [N, dv]
        self.keys_state: Optional[Tensor]
        self.values_state: Optional[Tensor]

        self.q_proj = nn.Linear(dim_in, dim_mem)
        self.kv_proj = nn.Linear(dim_in, 2*dim_mem)
        self.g_proj = nn.Linear(dim_in, 1)
        self.lr_proj = nn.Sequential(nn.Linear(dim_in, 1), nn.Sigmoid())
        self.lr_global = lr

    @staticmethod
    def _combine(x: Tensor, fn) -> Tensor:
        # x: [..., c, k] -> [..., k**c]; codebook 0 is the slowest-varying index
        z = x[..., 0, :]
        for i in range(1, x.shape[-2]):
            z = fn(z[..., :, None], x[..., i, None, :]).flatten(-2)
        return z

    def _read(self, q: Tensor):
        # keys: [b, c, n, d/c], values: [b, N, dv], q: [b, t, dim_mem]
        assert self.keys_state is not None and self.values_state is not None

        b = q.shape[0]
        q = rearrange(q, 'b t (c d) -> b t c d', c=self.key_books)
        s = torch.einsum('btcd,bcnd->btcn', q, self.keys_state)            # per-codebook scores
        s, idx = s.topk(self.top_k, dim=-1)                     # [b, t, c, k]

        scores = self._combine(s, torch.add)                                # [b, t, k**c]
        rows = self._combine(idx, lambda hi, lo: hi * self.n + lo)          # [b, t, k**c]

        top, pos = scores.topk(self.top_k, dim=-1)              # [b, t, k]
        rows = rows.gather(-1, pos)                             # [b, t, k]
        w = top.softmax(dim=-1)

        batch = torch.arange(b, device=q.device)[:, None, None]             # [b, 1, 1]
        v_hat = torch.einsum('btk,btkd->btd', w, self.values_state[batch, rows])       # values[batch, rows]: [b, t, k, dv]
        return v_hat, rows, w

    def q_read(self, q: Tensor):
        return self._read(q)[0]

    def forward(self, x: Tensor):   # Read
        q = self.q_proj(x)
        return self.q_read(q)[0]


    def write_kv(self, k: Tensor, v: Tensor, lr: Tensor):
        assert self.keys_state is not None and self.values_state is not None

        b = k.shape[0]

        v_hat, rows, w = self._read(k)
        err = v - v_hat                                            # [b, t, dv]

        # N_read: how many times each row is hit in this chunk, per sample
        flat = rows.flatten(1)                                     # [b, t*k]
        with torch.no_grad():
            counts = torch.zeros(b, self.N, device=k.device, dtype=w.dtype)
            counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=w.dtype))
            n_read = counts.gather(1, flat).view_as(rows)          # [b, t, k]

        # per-(token, slot) update, normalized by how many tokens wrote to that row
        delta = (lr[..., None] * w[..., None] * err[:, :, None, :]) / n_read[..., None]   # [b, t, k, dv]

        batch = torch.arange(b, device=k.device)[:, None, None]
        # out-of-place scatter-add: duplicate rows accumulate, no autograd in-place issues
        self.values_state = self.values_state.index_put((batch, rows), delta, accumulate=True)

    def write(self, x: Tensor) -> None:
        kv = self.kv_proj(x)
        wk, v = rearrange(kv, 'b t (p e) -> p b t e', p=2)        # write key, value: [b, t, dim_mem]
        lr = self.lr_proj(x) * self.lr_global                      # [b, t, 1]
        self.write_kv(wk,v,lr)
        

    def _fresh_state(self, n: int) -> tuple[Tensor, Tensor]:
        keys = self.keys.expand(n, *self.keys.shape).clone()
        values = self.values.expand(n, *self.values.shape).clone()
        return keys, values

    def reset_state(self, batch: int) -> None:
        self.keys_state, self.values_state = self._fresh_state(batch)

    def reset_state_at(self, indices: list[int]) -> None:
        assert self.keys_state is not None and self.values_state is not None
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.keys_state.device)
        fresh_k, fresh_v = self._fresh_state(len(idx))
        # out-of-place, so tensors saved for backward are never modified
        self.keys_state = self.keys_state.index_copy(0, idx, fresh_k)
        self.values_state = self.values_state.index_copy(0, idx, fresh_v)

    def get_state(self) -> FwPKM_MemoryState:
        return (self.keys_state, self.values_state)

    def set_state(self, state: FwPKM_MemoryState):
        self.keys_state, self.values_state = state

    def detach_state(self) -> None:
        if self.keys_state is not None:
            self.keys_state.detach()
        if self.values_state is not None:
            self.values_state.detach()
