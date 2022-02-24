import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x, temperature=10):
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()

        if bidirectional:
            hid_dim //= 2

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        if self.bidirectional:
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim)
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)

            cell = cell.reshape(self.n_layers, 2, -1, self.hid_dim)
            cell = cell.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)


        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, hidden, encoder_outputs):
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]
        # take the hidden state from the last layer (for multilayer model primarily)
        hidden = hidden[-1].unsqueeze(0) # unsqueeze to save number of dimensions
        
        # repeat hidden and concatenate it with encoder_outputs
        hidden = torch.cat(len(encoder_outputs) * [hidden])
        attn_data = torch.cat((hidden, encoder_outputs), dim=-1)
        # calculate energy
        energy = self.tanh((self.attn(attn_data)))
        # get attention, using softmax function defined above
        attention = softmax(self.v(energy))
            
        return attention
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        # Attention layer (object of Attention class, see above)
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim, n_layers, dropout=dropout) # use GRU
        
        self.out = nn.Linear(emb_dim + enc_hid_dim + dec_hid_dim, output_dim) # linear layer to get next word
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inp, hidden, encoder_outputs):
        #inp = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        
        inp = inp.unsqueeze(0) # because only one word, no words sequence 
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(inp))
        
        #embedded = [1, batch size, emb dim]
        
        # get weighted sum of encoder_outputs
        # attn = [sent_len, batch_size, 1]
        attn = self.attention(hidden, encoder_outputs).permute(1, 2, 0) # [batch_size, 1, sent_len]
        # encoder_outputs = [sent_len, batch_size, enc_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch_size, sent_len, enc_hid_dim]
        # w = [batch_size, 1, enc_hid_dim]
        w = torch.bmm(attn, encoder_outputs).permute(1, 0, 2) # [1, batch_size, enc_hid_dim]
        
        # concatenate weighted sum and embedded, break through the GRU
        outp, hidden = self.rnn(torch.cat((embedded, w), dim=-1), hidden)
        
        # get predictions
        pred_data = torch.cat((embedded, w, outp), dim=-1)
        prediction = self.out(pred_data)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim * (encoder.bidirectional + 1) == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        inp = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(inp, hidden, enc_states)
            outputs[t] = output[0]
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output[0].argmax(-1)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            inp = trg[t,:] if teacher_force else top1
        
        return outputs
