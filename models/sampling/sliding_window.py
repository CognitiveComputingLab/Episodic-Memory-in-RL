from gymnasium import Env
import torch

from models import BaseModel

@torch.no_grad()
def sliding_window_sample(
    model: BaseModel, 
    max_len: int,
    obs_dim,
    act_dim,
    is_discrete,
    env: Env, 
    target: int) -> float:
    
    model.eval()
    model.reset_states(1)

    device = next(model.parameters()).device

    obs, _ = env.reset()

    rtgs      = torch.zeros(1, max_len, 1,           device=device)
    states    = torch.zeros(1, max_len, obs_dim, device=device)
    actions   = (torch.zeros(1, max_len, 1 , device=device, dtype=torch.int32) if is_discrete
            else torch.zeros(1, max_len, act_dim, device=device, dtype=torch.float32))
    timesteps = torch.arange(max_len,                 device=device).unsqueeze(0)

    rtgs[0, 0, 0] = target
    states[0, 0]  = torch.as_tensor(obs, dtype=torch.float32, device=device)

    ep_return = 0.0
    for step in range(max_len):
        model.reset_caches(1)
        end = step + 1
        start = max(0, end - model.context_len)

        pred = model(
            rtgs     [:, start:end],
            states   [:, start:end],
            actions  [:, start:end-1],
            timesteps[:, start:end],
        )
        action = pred[0, -1]

        if is_discrete:
            action_np  = int(action.argmax())
            action_vec = torch.tensor(action_np, dtype=torch.int32)
        else:
            action_np  = action.cpu().numpy()
            action_vec = action

        obs, reward, terminated, truncated, _ = env.step(action_np)
        ep_return += float(reward)

        if step + 1 < max_len:
            rtgs  [0, step + 1, 0] = rtgs[0, step, 0] - torch.tensor(reward)
            states[0, step + 1]    = torch.as_tensor(obs, dtype=torch.float32, device=device)
        actions[0, step] = action_vec

        if terminated or truncated:
            break

    model.train()
    return ep_return