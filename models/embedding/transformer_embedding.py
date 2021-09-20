from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    transformer embedding = token embedding + positional encoding
    positional encoding can give positional information to a network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """Word embedding that contains positional information

        Args:
            vocab_size (int): size of vocabulary
            d_model (int): dimensions of model
            max_len (int): 
            drop_prob (float): the probability of dropping 
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)
