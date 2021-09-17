from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Use PyTorch Embedding module
    Here we will get a dense representation of words using a weighted matrix
    """


    def __init__(self, vocab_size, d_model):
        """Initialize the class for token embedding that included positional information

        Args:
            vocab_size (int): input vocabulary size
            d_model (int): model dimension
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)