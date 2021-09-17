import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Compute sinusoidal encoding
    """

    def __init__(self, d_model, max_len, device):
        """Constructor of sinusoid encoding class

        Args:
            d_model ([type]): [description]
            max_len ([type]): [description]
            device ([type]): [description]
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(start=0, end=max_len, device=device).float()
        pos = pos.unsqueeze(dim=1)
        # [0., 1., 2., 3.] ==> [[0.], [1.], [2.], [3.]]
        # 1D ==> 2D to represent words' positions

        _2i = torch.arange(start=0, end=d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding vector size = 50, then 'i' = [0,50])
        # 'step=2' means 'i' multiplied with 2 (i*2 = 2*i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 256, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb: [128, 30, 512]
