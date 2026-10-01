from gymnasium import Env
import torch

from models import BaseModel

@torch.no_grad()
def chunked_cacheful_sample(
    model: BaseModel, 
    max_len: int,
    obs_dim,
    act_dim,
    is_discrete,
    env: Env, 
    target: int) -> float:

    model.eval()
    model.reset_states(1)    # Reset memory
    model.reset_caches(1)    # Reset caches    
    model.enable_caches(True)

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

        end = step + 1
        pred = model(
            rtgs     [:, step:end],
            states   [:, step:end],
            actions  [:, (step-1 if step > 0 else 0):step],    # Do not pass in a dummy action to avoid this being saved in memory
            timesteps[:, step:end],
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

        # At the end of a chunk, do another inference with the final action in order to save it
        if end % model.context_len == 0:
            model(
                rtgs     [:, end:end],
                states   [:, end:end],
                actions  [:, step:end],
                timesteps[:, step:end],
            )
            model.reset_caches(1)    # reset cache at end of chunk

        if terminated or truncated:
            break

    model.wipe_caches()
    model.enable_caches(False)
    model.train()
    return ep_return