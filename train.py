import math
import time

from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler

from data import *
from models.model.transformer import Transformer
from utils.blue import idx_to_word, get_blue
from utils.epoch_timer import epoch_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(src_pad_idx=src_pad_idx, 
                    trg_pad_idx=trg_pad_idx, 
                    trg_sos_idx=trg_sos_idx,
                    enc_voc_size=enc_voc_size, 
                    dec_voc_size=dec_voc_size,
                    d_model=d_model, 
                    n_head=n_head, 
                    max_len=max_len, 
                    ffn_hidden=ffn_hidden, 
                    n_layers=n_layers, 
                    drop_prob=drop_prob, 
                    device=device).to(device)

# count parameters
print(f"The model has {count_parameters(model):,} trainable parameters")

# initialize weights
model.apply(initialize_weights)

optimizer = Adam(params=model.parameters(),
                    lr=init_lr,
                    weight_decay=weight_decay,
                    eps=adam_eps)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                            verbose=True,
                                            factor=factor,
                                            patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print(f"step : {round((i/len(iterator))*100, 2)}%, loss : {loss.item()}")

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_blue = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_blue = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_blue(hypothesis=output_words.split(), reference=trg_words.split())
                    total_blue.append(bleu)
                except:
                    pass
            total_blue = sum(total_blue) / len(total_blue)
            batch_blue.append(total_blue)
    
    batch_blue = sum(batch_blue) / len(batch_blue)
    return epoch_loss / len(iterator), batch_blue


def run(total_epoch, best_loss):
    train_losses, test_losses, blues = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss, blue = evaluate(model, valid_iterator, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)
        
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        blue.append(blue)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('results/train_loss.txt', 'w')
        f.write(str(train_loss))
        f.close()

        f = open('results/blue.txt', 'w')
        f.write(str(blue))
        f.close()

        f = open('results/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f"Epoch: {step+1} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f)}")
        print(f"\tVal Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):7.3f)}")
        print(f"\tBLUE Score: {blue:.3f}")


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)

