import math

from torch import nn


class ScaledDotProductAttention(nn.Module):
    """Compute scale dot-product attention
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, mask=None, eps=1e-12):
        # input has 4D tensor shape [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.view(batch_size, head, d_tensor, length)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (optional)
        if mask is not None:
            score = score.masked_fill(mask==0, -eps)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score