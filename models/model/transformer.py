import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        
        self.encoder = Encoder(enc_voc_size=enc_voc_size,
                               max_len=max_len,
                               d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               drop_prob=drop_prob,
                               device=device)
        
        self.decoder = Decoder(dec_voc_size=dec_voc_size, 
                               max_len=max_len, 
                               d_model=d_model,
                               ffn_hidden=ffn_hidden, 
                               n_head=n_head, 
                               n_layers=n_layers, 
                               drop_prob=drop_prob, 
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        src_trg_maks = self.make_pad_mask(trg, src)
        trg_mask = self.make_pad_mask(trg, trg) * self.make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_maks)
        
        return output

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # [batch_size, 1, 1, len_k]
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # [batch_size, 1, len_q, len_k]
        k = k.repeat(1, 1, len_q, 1)
        
        # [batch_size, 1, len_q, 1]
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # [batch_size, 1, len_q, len_k]
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # [len_q, len_k]
        mask = torch.tril(torch.ones((len_q, len_k))).type(torch.BoolTensor).to(self.device)
        
        return mask
