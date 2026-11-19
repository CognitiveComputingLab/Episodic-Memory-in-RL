from torch import Tensor
import torch.nn as nn 

from models.DTi.DTi import DecisionTitan
from models.TBD.TBD_Block import TBD_Block


class TBD(DecisionTitan):

    def __init__(
            self, 
            context_len: int, 
            embed_dim: int, 
            num_persistent: int,        # How many persistent tokens appended to the beginning of a chunk (unique for each layer)
            act_dim: int, 
            obs_dim: int, 
            max_action: float, 
            max_ep_len: int,
            is_discrete: bool, 
            time_embed: nn.Module, 
            rtg_embed: nn.Module, 
            state_embed: nn.Module, 
            action_embed: nn.Module, 
            action_unembed: nn.Module,
            blocks: nn.ModuleList,
            lr_T: float,
            lr_V: float,
            lr_R: float) -> None:
        super().__init__(context_len, embed_dim, num_persistent, act_dim, obs_dim, max_action, max_ep_len, is_discrete, time_embed, rtg_embed, state_embed, action_embed, action_unembed, blocks)
        self.lr_T = lr_T
        self.lr_V = lr_V
        self.lr_R = lr_R

    
    def loss(self, y: Tensor, pred_y: Tensor, mask: Tensor) -> Tensor:
        t_loss = super().loss(y, pred_y, mask)
        v_loss = sum([block.v_loss for block in self.blocks if isinstance(block, TBD_Block) and block.v_loss is not None])
        r_loss = sum([block.r_loss for block in self.blocks if isinstance(block, TBD_Block) and block.r_loss is not None])

        return self.lr_T * t_loss + self.lr_V * v_loss + self.lr_R * r_loss 
