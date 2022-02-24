import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    def __init__(self, in_seq_len, out_seq_len, in_feat_len,  out_feat_len,static_feat_len, 
                hidden_size, num_layers, enc_dropout, dec_dropout, arch, device):
        super().__init__()
        self.encoder = RNNEncoder(rnn_num_layers=num_layers, 
                                  input_feature_len=in_feat_len,
                                  sequence_len=in_feat_len,
                                  hidden_size=hidden_size,
                                  rnn_dropout=enc_dropout,
                                  arch = arch,
                                  device=device)

        self.decoder_cell = DecoderCell(in_feat_len=in_feat_len ,
                                        out_feat_len=out_feat_len,
                                        hidden_size=hidden_size,
                                        dropout=dec_dropout, 
                                        num_layers=num_layers,
                                        arch=arch)
        self.in_feat_len = in_feat_len
        self.out_seq_len = out_seq_len
        self.device = device
        self.out_feat_len = out_feat_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, enc_input, dec_input):
        #forward pass of encoder
        enc_output, enc_hidden = self.encoder(enc_input)

        # hidden state dim: num_layers x batchsize x hidden size
        prev_hidden = enc_hidden

        # decoder input dim: batchsize x 1 x forecast features
        # give the last known datapoint of the forecast features as first input to decoder
     
        y_prev = enc_input[:, -1:,:self.out_feat_len]
        # output dim: batchsize x out_seq_len x forecast features
        dec_output = torch.zeros(enc_input.size(0), self.out_seq_len, self.out_feat_len, device=self.device).double()

        # autoregressive prediction loop of decoder
        for i in range(self.out_seq_len):
            step_decoder_input = torch.cat((y_prev, dec_input[:, [i]]), axis=2)
            rnn_output, prev_hidden = self.decoder_cell(step_decoder_input, prev_hidden)
            y_prev = rnn_output
            dec_output[:, i] = rnn_output.squeeze(1)
        return dec_output




class RNNEncoder(nn.Module):
    def __init__(self, rnn_num_layers, input_feature_len, sequence_len, hidden_size, device, rnn_dropout, arch):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.arch = arch
        if arch == "GRU":
            self.rnn_layer = nn.GRU(
                num_layers=rnn_num_layers,
                input_size=input_feature_len,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=rnn_dropout
            )
        elif arch == "LSTM":
            self.rnn_layer = nn.LSTM(
                num_layers=rnn_num_layers,
                input_size=input_feature_len,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=rnn_dropout
            )
            
        self.device = device

    def forward(self, input_seq):
        if self.arch == "GRU":
            if True: 
                h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size, device=self.device).double()
            else:
                h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).cuda().double()
        elif self.arch == "LSTM":
            h0 = (torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size, device=self.device).double(),
                  torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size, device=self.device).double())
        gru_out, hidden = self.rnn_layer(input_seq, h0)
        return gru_out, hidden


class DecoderCell(nn.Module):
    def __init__(self, in_feat_len, out_feat_len, hidden_size, dropout, num_layers, arch):
        super().__init__()

        if arch == "GRU":
            self.decoder_rnn_cell = nn.GRU(
                num_layers=num_layers,
                input_size=in_feat_len+out_feat_len, # lenght of encoder output+ decoder input
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout
            )
        elif arch == "LSTM":
            self.decoder_rnn_cell = nn.LSTM(
                num_layers=num_layers,
                input_size=in_feat_len, 
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout
            )
        self.out1 = nn.Linear(hidden_size , hidden_size)
        self.relu = torch.nn.ReLU()
        self.out2 = nn.Linear(hidden_size, out_feat_len,bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, prev_hidden):
        output, hidden = self.decoder_rnn_cell(y, prev_hidden)
        output = self.out1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.out2(output)
        return output, hidden
