import torch.nn as nn

from models.DTi.Config import register_layout
from models.modules.TTT import MAL_Block, titans_mlp
from models.modules.Transformer import TransformerBlock

@register_layout("mal_blocks")
def mal_blocks(
        layers: int,
        heads: int,
        context_len: int,
        embed_dim: int,
        num_persistent: int,
        mini_batch_size: int,
        titan_layers: list[int]
):
    return nn.ModuleList([
        TransformerBlock(
            context_len * 3,
            embed_dim,
            heads
            ) if i not in titan_layers else
        MAL_Block(
            context_len * 3,
            embed_dim,
            heads,
            num_persistent=num_persistent,
            mini_batch_len=mini_batch_size,
            mem_module=titans_mlp(ff_mult=4)
            )
        for i in range(layers)
    ])