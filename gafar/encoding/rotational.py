"""
Rotational Fourier Feature Positional Encoding

adapted from LightGlue
https://github.com/cvg/LightGlue/blob/main/lightglue/lightglue.py#L61

"""

import torch


class LearnableFourierPositionalEncoding(torch.nn.Module):
    def __init__(self, m: int, dim: int, f_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        f_dim = f_dim if f_dim is not None else dim
        self.gamma = gamma
        self.Wr = torch.nn.Linear(m, f_dim // 2, bias=False)
        torch.nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3).repeat_interleave(2, dim=-1)
        # R x B x H x M x D -> R x B x D x H x M
        return emb.permute(0, 1, 4, 2, 3).contiguous()
