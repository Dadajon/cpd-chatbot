from torch import nn

from models.layers.layer_normalization import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """[summary]

        Args:
            d_model ([type]): [description]
            ffn_hidden ([type]): [description]
            n_head ([type]): [description]
            drop_prob ([type]): [description]
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x  # copy for residual (skip) connection
        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        # 2. add and norm
        x = self.norm1(x+_x)
        x = self.dropout1(x)

        # 3. pointwise feed forward network
        _x = x  # copy for residual (skip) connection
        x = self.ffn(x)

        # 4. add and norm
        x = self.norm2(x+_x)
        x=self.dropout2(x)

        return x