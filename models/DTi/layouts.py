import torch.nn as nn

from models.DTi import MAL_Block
from models.DTi.SG_Block import SG_Block
from models.DTi.Config import register_layout
from models.TBD import TBD_Block
from models.modules.TTT import titans_mlp
from models.modules.Transformer import TransformerBlock

@register_layout("mal_blocks")
def mal_blocks(
        layers: int,
        heads: int,
        context_len: int,
        embed_dim: int,
        num_persistent: int,
        memory_update_interval: int,
        titan_layers: list[int]
):
    return nn.ModuleList([
        TransformerBlock(
            w_size=context_len * 3,
            e_dim=embed_dim,
            heads=heads
            ) if i not in titan_layers else
        MAL_Block(
            context_len=context_len * 3,
            e_dim=embed_dim,
            heads=heads,
            num_persistent=num_persistent,
            mini_batch_len=memory_update_interval,
            mem_module=titans_mlp(ff_mult=4)
            )
        for i in range(layers)
    ])


@register_layout("tbd_blocks")
def tbd_blocks(
        layers: int,
        heads: int,
        context_len: int,
        embed_dim: int,
        num_persistent: int,
        memory_update_interval: int,
        titan_layers: list[int]
):
    return nn.ModuleList([
        TransformerBlock(
            w_size=context_len * 3,
            e_dim=embed_dim,
            heads=heads
            ) if i not in titan_layers else
        TBD_Block(
            context_len=context_len * 3,
            e_dim=embed_dim,
            m_dim=embed_dim,
            heads=heads,
            num_persistent=num_persistent,
            mini_batch_len=memory_update_interval,
            mem_module=titans_mlp(ff_mult=4)
            )
        for i in range(layers)
    ])


@register_layout("sg_blocks")
def sg_blocks(
        layers: int,
        heads: int,
        context_len: int,
        embed_dim: int,
        num_persistent: int,
        memory_update_interval: int,
        titan_layers: list[int]
):
    """Construct Titans blocks with the synthetic-gradient block contract."""
    return nn.ModuleList([
        TransformerBlock(
            w_size=context_len * 3,
            e_dim=embed_dim,
            heads=heads
            ) if i not in titan_layers else
        SG_Block(
            context_len=context_len * 3,
            e_dim=embed_dim,
            heads=heads,
            num_persistent=num_persistent,
            mini_batch_len=memory_update_interval,
            mem_module=titans_mlp(ff_mult=4)
            )
        for i in range(layers)
    ])
