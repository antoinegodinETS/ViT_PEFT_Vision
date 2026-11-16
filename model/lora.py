import math
import torch.nn as nn
from utils.misc import get_param


class LoRA(nn.Module):
    """LoRA (Hu et al., 2021) on the query and value projections only.

    W_q and W_v each get their own low-rank pair (A: dim -> r, B: r -> dim);
    W_k is left untouched. The update is scaled by alpha / r (alpha defaults
    to r, i.e. scale 1, which matches the original ViT_PEFT_Vision code) and
    by merge_factor. Dropout (default 0.1) is applied to the LoRA input.
    With dim=768, 12 blocks and r=8 this gives 294,912 trainable parameters.
    """

    def __init__(self, dim, num_heads, params):
        super().__init__()
        self.down_size = params.lora_bottleneck
        alpha = get_param(params, 'lora_alpha', None)
        if alpha is None:
            alpha = self.down_size
        self.scaling = float(alpha) / self.down_size
        self.lora_drop = nn.Dropout(p=get_param(params, 'lora_dropout', 0.1))
        self.num_heads = num_heads
        self.lora_a_q = nn.Linear(dim, self.down_size, bias=False)
        self.lora_b_q = nn.Linear(self.down_size, dim, bias=False)
        self.lora_a_v = nn.Linear(dim, self.down_size, bias=False)
        self.lora_b_v = nn.Linear(self.down_size, dim, bias=False)
        for a, b in ((self.lora_a_q, self.lora_b_q), (self.lora_a_v, self.lora_b_v)):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
        self.params = params

    def _delta(self, x, lora_a, lora_b, B, N, C):
        # (B, N, C) -> (B, num_heads, N, head_dim), the layout of q/k/v in AttentionPETL
        return lora_b(lora_a(x)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    def forward(self, x, q, k, v, B, N, C):
        x = self.lora_drop(x)
        scale = self.scaling * self.params.merge_factor
        q = q + scale * self._delta(x, self.lora_a_q, self.lora_b_q, B, N, C)
        v = v + scale * self._delta(x, self.lora_a_v, self.lora_b_v, B, N, C)
        return q, k, v
