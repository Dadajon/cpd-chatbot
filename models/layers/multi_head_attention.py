from torch import nn

from models.layers.scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
    """

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        # attention is to visualize attention maps
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        
        return out

    def split(self, tensor):
        """Split tensor by number of heads
        Returns a tensor with shape [batch_size, length, d_tensor]

        Args:
            tensor (tensor): [batch_size, length, d_model]
        """
        batch_size, length, d_model = tensor.size()
        
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, self.n_head, length, d_tensor)
        # it is similar with group convolution (split by number of heads)
        
        return tensor

    def concat(self, tensor):
        """Inverse function of self.split(tensor: torch.Tensor)
        Returns a tensor with shape [batch_size, length, d_tensor]

        Args:
            tensor (tensos): [batch_size, head, length, d_tensor]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        
        tensor = tensor.view(batch_size, length, d_model)
        return tensor
