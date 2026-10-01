import torch

from models.ELMUR import ELMUR_Block

b,t,e = 4, 9, 16
x = torch.normal(0,1,(b,t,e))

block = ELMUR_Block(t,e,1,6,0.5)
block.reset_state(b)
y = block(x)
print(y)