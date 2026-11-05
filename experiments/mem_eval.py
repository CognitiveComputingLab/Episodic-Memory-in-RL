"""
mem_eval.py -- run (and optionally meta-train W_0 for) a TTT / Titans memory, and save raw
results for mem_plot.py.  No plotting happens here.

Files written to --out-dir for run name NAME and seed S
-------------------------------------------------------
    NAME_S.npz / NAME_S.json         final result (after W_0 training, if any)
    NAME_0_S.npz / .json             --train-w0 only: the untrained (random) W_0
    NAME_<step>_S.npz / .json        --ckpt-every N: evaluation after N, 2N, ... outer steps
    NAME_<step>_S.pt                 --save-w0: the memory's initial weights at that point

The LAST number in a file name is always the seed. mem_plot.py averages files that differ
only in that number, so a checkpoint ('ttt_100_0') never collides with a seed ('ttt_0').
For the same seed, every model_type gets the same W_0 init, the same kv projection and the
same test sequences, so different models can be compared directly.

Examples
--------
    python mem_eval.py --name ttt --model-type TTT --seed 0
    python mem_eval.py --name titans --model-type Titans --mem-lr 0.1 --train-w0 --ckpt-every 25
    for s in 0 1 2; do python mem_eval.py --name ttt --seed $s --train-w0; done
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from models.modules.TTT import FwPKM_Memory, MemoryModule, TitansMemory, TTT_Memory


# --------------------------------------------------------------------------- #
# Config: plain, JSON-serialisable data only (it is stored next to every result)
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    # identity
    name: str = "TTT"              # must not end in _<number>
    out_dir: str = "results/TTT"
    seed: int = 0                  # becomes the trailing number of every file name

    # model
    model_type: str = "TTT"        # 'TTT' | 'Titans' | 'FwPKM'
    mem_lr: float = 1.0            # Titans: momentum multiplies the step by 1/(1-s_retain) -> use ~0.1
    s_retain: float = 0.5          # Titans: momentum retention
    m_retain: float = 1            # Titans: memory retention, applied per inner step
    dim_in: int = 32
    dim_mem: int = 64
    e_factor: int = 4

    # data
    kv_proj: str = "linear_nb"     # 'linear_nb' = frozen linear map, no bias
    kv_mode: str = "independent"   # 'shared': k and v from the same noise (v = A k is learnable)
                                   # 'independent': k and v from independent noise (only storable)
    b: int = 8                     # sequences per evaluation (each has its own memory)
    t: int = 256
    minibatch: int = 8
    inner_steps: int = 10          # gradient steps per chunk

    # W_0 meta-training
    train_w0: bool = True
    outer_steps: int = 100
    outer_lr: float = 1e-3
    outer_loss: str = "seen"       # 'all' | 'seen' (memorisation) | 'unseen' (generalisation)
    grad_clip: float = 1.0
    ckpt_every: int = 0            # evaluate + save every N outer steps (0 = only start and end)
    save_w0: bool = False          # also store the memory's initial weights next to each result

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DummyGate(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x: Tensor):
        return torch.ones([x.shape[0], x.shape[1], 1], device=x.device) * self.val


@dataclass
class Exp:
    """Everything that is not plain data: the model and the functions bound to it."""
    model: MemoryModule
    write: Callable[[Tensor, Tensor, Tensor], None]            # (x_chunk, k_chunk, v_chunk)
    sample_kv: Callable[[Tensor], tuple[Tensor, Tensor]]       # x -> (k, v)


def build_experiment(cfg: Cfg) -> Exp:
    # W_0 is drawn first, straight after seeding, so it is identical for every model_type.
    torch.manual_seed(cfg.seed)
    mem_model = nn.Sequential(
        nn.Linear(cfg.dim_mem, cfg.dim_mem * cfg.e_factor),
        nn.GELU(),
        nn.Linear(cfg.dim_mem * cfg.e_factor, cfg.dim_mem),
    )
    gate = DummyGate(1)

    if cfg.model_type == "TTT":
        model = TTT_Memory(cfg.dim_in, cfg.dim_mem, cfg.mem_lr, mem_model)

        def write(x: Tensor, k: Tensor, v: Tensor):
            model.write_kv(k, v, gate(x))

    elif cfg.model_type == "Titans":
        model = TitansMemory(cfg.dim_in, cfg.dim_mem, cfg.mem_lr, mem_model)

        def write(x: Tensor, k: Tensor, v: Tensor):
            model.write_kv(k, v, gate(x), cfg.s_retain, cfg.m_retain)

    elif cfg.model_type == 'FwPKM':
        model = FwPKM_Memory(cfg.dim_in, cfg.dim_mem, cfg.mem_lr, 256, 5, 2)
        
        def write(x: Tensor, k: Tensor, v: Tensor):
            model.write_kv(k, v, gate(x))
    else:
        raise ValueError(f"Unrecognised model type: {cfg.model_type}")

    model = model.to(cfg.device)
    return Exp(model=model, write=write, sample_kv=make_kv_fn(cfg))


def make_kv_fn(cfg: Cfg):
    """Frozen k/v projection that depends only on (seed, dims) -- not on model_type."""
    if cfg.kv_proj != "linear_nb":
        raise ValueError(f"Unrecognised kv proj: {cfg.kv_proj}")
    g = torch.Generator().manual_seed(1_000_003 + cfg.seed)
    bound = 1 / math.sqrt(cfg.dim_in)                     # same range as nn.Linear's default init
    w = ((torch.rand(2 * cfg.dim_mem, cfg.dim_in, generator=g) * 2 - 1) * bound).to(cfg.device)

    def sample_kv(x: Tensor):
        return rearrange(F.linear(x, w), "b ... (p e) -> p b ... e", p=2)

    return sample_kv


def make_gen(cfg: Cfg, seed: int) -> torch.Generator:
    g = torch.Generator(device=cfg.device)
    g.manual_seed(seed)
    return g


@torch.no_grad()
def sample(exp: Exp, cfg: Cfg, gen: torch.Generator):
    """Fresh noise 'hidden states' -> (x, k, v). The k/v projection is the same every call."""
    shape = [cfg.b, cfg.t, cfg.dim_in]
    x = torch.randn(shape, generator=gen, device=cfg.device)
    k, v = exp.sample_kv(x)
    if cfg.kv_mode == "independent":
        _, v = exp.sample_kv(torch.randn(shape, generator=gen, device=cfg.device))
    return x, k, v


# --------------------------------------------------------------------------- #
# Measuring recall
# --------------------------------------------------------------------------- #
def per_chunk_mse(v: Tensor, v_pred: Tensor, mb: int) -> Tensor:
    """MSE per sequence and chunk position. Shape [b, t // mb]."""
    err = ((v_pred - v) ** 2).mean(-1)                                 # [b, t]
    return rearrange(err, "b (n m) -> b n m", m=mb).mean(-1)


def recall(exp: Exp, cfg: Cfg, k: Tensor, v: Tensor) -> Tensor:
    return per_chunk_mse(v, exp.model.q_read(k), cfg.minibatch)


def rollout(exp: Exp, cfg: Cfg, x, k, v, held=None, differentiable=False):
    """Write x chunk by chunk, measuring recall after each chunk.

    Returns
        L [n+1, b, n] : L[s, i, c] = loss of sequence i on chunk c after s chunks were written
        H [n+1, b]    : loss of sequence i on a held-out sequence (never written) after s chunks
    """
    mb = cfg.minibatch
    exp.model.reset_state(cfg.b)               # state = W_0 copy per sequence

    L = [recall(exp, cfg, k, v)]
    H = [recall(exp, cfg, *held).mean(-1)] if held is not None else []
    for i in range(0, cfg.t, mb):
        sl = slice(i, i + mb)
        for _ in range(cfg.inner_steps):
            exp.write(x[:, sl], k[:, sl], v[:, sl])
        if not differentiable:
            exp.model.detach_state()           # training keeps the graph for dLoss/dW_0
        L.append(recall(exp, cfg, k, v))
        if held is not None:
            H.append(recall(exp, cfg, *held).mean(-1))
    return torch.stack(L), (torch.stack(H) if held is not None else None)


@torch.no_grad()
def evaluate(exp: Exp, cfg: Cfg) -> dict:
    """Evaluate the current W_0 on a fixed test sequence set.

    The data depends only on cfg.seed, so every checkpoint of a run -- and every model_type
    run with the same seed -- is evaluated on identical sequences. Training never sees them
    (it draws from a different generator).
    """
    gen = make_gen(cfg, 10_000 + cfg.seed)
    x, k, v = sample(exp, cfg, gen)
    _, kh, vh = sample(exp, cfg, gen)          # held-out sequence: same distribution, never written
    L, H = rollout(exp, cfg, x, k, v, held=(kh, vh))
    return {"L": L.cpu().numpy(), "H": H.cpu().numpy()}


# --------------------------------------------------------------------------- #
# W_0 meta-training (outer loop)
# --------------------------------------------------------------------------- #
def outer_objective(L: Tensor, mode: str) -> Tensor:
    L = L.mean(1)                                                      # average over sequences
    s = torch.arange(L.shape[0], device=L.device)[:, None]
    c = torch.arange(L.shape[1], device=L.device)[None, :]
    mask = {"all": torch.ones_like(L, dtype=torch.bool), "seen": c < s, "unseen": c >= s}[mode]
    return (L * mask).sum() / mask.sum()


def train_w0(exp: Exp, cfg: Cfg, on_checkpoint: Optional[Callable[[int, list], None]] = None) -> list[float]:
    """Learn the memory's initial weights W_0 by backprop through the whole inner loop.

    Each outer iteration draws NEW x; the k/v projection is fixed. Gradients flow
    loss -> reads -> state_n -> ... -> state_0 = W_0, so the inner loop is not detached.
    """
    params = list(exp.model.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.outer_lr)
    gen = make_gen(cfg, 20_000 + cfg.seed)
    history: list[float] = []
    for it in range(cfg.outer_steps):
        x, k, v = sample(exp, cfg, gen)
        with torch.enable_grad():
            L, _ = rollout(exp, cfg, x, k, v, differentiable=True)
            loss = outer_objective(L, cfg.outer_loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        if torch.isfinite(gnorm):
            opt.step()
        else:
            print(f"  iter {it}: non-finite meta-gradient, step skipped", flush=True)
        history.append(loss.item())
        if it % max(1, cfg.outer_steps // 20) == 0 or it == cfg.outer_steps - 1:
            print(f"  outer {it:4d}/{cfg.outer_steps}  loss {loss.item():.4g}  |g| {gnorm.item():.3g}", flush=True)

        done = it + 1
        if on_checkpoint and cfg.ckpt_every and done % cfg.ckpt_every == 0 and done < cfg.outer_steps:
            on_checkpoint(done, history)       # the last step is saved as the final result instead
    for p in params:
        p.requires_grad_(False)
    exp.model.detach_state()
    return history


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_result(exp: Exp, cfg: Cfg, step: int, history: list[float], final: bool) -> Path:
    """Evaluate the current W_0 and write NAME[_step]_SEED.npz + .json (+ .pt)."""
    res = evaluate(exp, cfg)
    stem = f"{cfg.name}_{cfg.seed}" if final else f"{cfg.name}_{step}_{cfg.seed}"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez(
        out / f"{stem}.npz",
        L=res["L"],                                  # [n+1, b, n] per-sequence loss matrix
        H=res["H"],                                  # [n+1, b]    per-sequence held-out loss
        minibatch=cfg.minibatch,
        t=cfg.t,
        step=step,                                   # outer updates applied when this was evaluated
        history=np.asarray(history, dtype=np.float32),   # outer loss per iteration so far
    )
    with open(out / f"{stem}.json", "w") as f:
        json.dump({"cfg": asdict(cfg), "step": step, "final": final}, f, indent=2)
    if cfg.save_w0:
        torch.save(exp.model.memoryModule.state_dict(), out / f"{stem}.pt")

    if not np.isfinite(res["L"]).all():
        print(f"  WARNING: non-finite losses in {stem} -- the memory diverged "
              f"(Titans: lower mem_lr or s_retain)", flush=True)
    seen = res["L"][-1].mean()
    print(f"[{stem}] step {step}: start {res['L'][0].mean():.4g} | "
          f"final seen {seen:.4g} | held-out {res['H'][-1].mean():.4g}", flush=True)
    return out / f"{stem}.npz"


def run(id: int = 0, cfg: Cfg | None = None) -> list[Path]:
    cfg = cfg or Cfg(seed=id)

    if re.search(r"_\d+$", cfg.name):
        raise ValueError(f"name '{cfg.name}' ends in _<number>, which would be read as a seed or step")
    exp = build_experiment(cfg)
    paths: list[Path] = []

    history: list[float] = []
    if cfg.train_w0:
        paths.append(save_result(exp, cfg, 0, [], final=False))          # untrained W_0
        print(f"Training W_0: {cfg.outer_steps} outer iterations (loss='{cfg.outer_loss}')")
        history = train_w0(exp, cfg, lambda step, h: paths.append(save_result(exp, cfg, step, h, final=False)))
    paths.append(save_result(exp, cfg, cfg.outer_steps if cfg.train_w0 else 0, history, final=True))
    return paths


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Cfg):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(Cfg, f.name)
        if isinstance(default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            ap.add_argument(flag, type=type(default), default=default)
    return Cfg(**vars(ap.parse_args()))


if __name__ == "__main__":
    run(0, parse_args())
