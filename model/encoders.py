import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import Attn

class MetaNetRNN(nn.Module):
    # Meta Seq2Seq encoder
    #
    # Encodes query items in the context of the support set, which is stored in external memory.
    #
    #  Architecture
    #   1) RNN encoder for input symbols in query and support items (either shared or separate)
    #   2) RNN encoder for output symbols in the support items only
    #   3) Key-value memory for embedding query items with support context
    #   3) MLP to reduce the dimensionality of the context-sensitive embedding
    def __init__(self, embedding_dim, input_size, output_size, nlayers, dropout_p=0.1, bidirectional=True, tie_encoders=True):
        #
        # Input
        #  embedding_dim : number of hidden units in RNN encoder, and size of all embeddings
        #  input_size : number of input symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers in RNN encoder
        #  dropout : dropout applied to symbol embeddings and RNNs
        #  bidirectional : use bi-directional RNN encoders? (default=True)
        #  tie_encoders : use the same encoder for the support and query items? (default=True)
        #
        super(MetaNetRNN, self).__init__()
        self.nlayers = nlayers
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.bi_encoder = bidirectional
        self.attn = Attn()
        self.suppport_embedding = EncoderRNN(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
        if tie_encoders:
            self.query_embedding = self.suppport_embedding
        else:
            self.query_embedding = EncoderRNN(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
        self.output_embedding = EncoderRNN(output_size, embedding_dim, nlayers, dropout_p, bidirectional)
        self.hidden = nn.Linear(embedding_dim*2,embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, sample):
        #
        # Forward pass over an episode
        #
        # Input
        #   sample: episode dict wrapper for ns support and nq query examples (see 'build_sample' function in training code)
        #
        # Output
        #   context_last : [nq x embedding]; last step embedding for each query example
        #   embed_by_step: embedding at every step for each query [max_xq_length x nq x embedding_dim]
        #   attn_by_step : attention over support items at every step for each query [max_xq_length x nq x ns]
        #   seq_len : length of each query [nq list]
        #
        xs_padded = sample['xs_padded'] # support set input sequences; LongTensor (ns x max_xs_length)
        xs_lengths = sample['xs_lengths'] # ns list of lengths
        ys_padded = sample['ys_padded'] # support set output sequences; LongTensor (ns x max_ys_length)
        ys_lengths = sample['ys_lengths'] # ns list of lengths
        ns = xs_padded.size(0)
        xq_padded = sample['xq_padded'] # query set input sequences; LongTensor (nq x max_xq_length)
        xq_lengths = sample['xq_lengths'] # nq list of lengths
        nq = xq_padded.size(0)

        # Embed the input sequences for support and query set
        embed_xs,_ = self.suppport_embedding(xs_padded,xs_lengths) # ns x embedding_dim
        embed_xq,dict_embed_xq = self.query_embedding(xq_padded,xq_lengths) # nq x embedding_dim
        embed_xq_by_step = dict_embed_xq['embed_by_step'] # max_xq_length x nq x embedding_dim (embedding at each step)
        len_xq = dict_embed_xq['seq_len'] # len_xq is nq array with length of each sequence

        # Embed the output sequences for support set
        embed_ys,_ = self.output_embedding(ys_padded,ys_lengths) # ns x embedding_dim

        # Compute context based on key-value memory at each time step for queries
        max_xq_length = embed_xq_by_step.size(0) # for purpose of attention, this is the "batch_size"
        value_by_step, attn_by_step = self.attn(embed_xq_by_step, embed_xs.expand(max_xq_length,-1,-1), embed_ys.expand(max_xq_length, -1, -1))
            # value_by_step : max_xq_length x nq x embedding_dim
            # attn_by_step : max_xq_length x nq x ns
        concat_by_step = torch.cat((embed_xq_by_step,value_by_step),2) # max_xq_length x nq x embedding_dim*2
        context_by_step = self.tanh(self.hidden(concat_by_step)) # max_xq_length x nq x embedding_dim

        # Grab the last context for each query
        context_last = [context_by_step[len_xq[q]-1,q,:] for q in range(nq)] # list of 1D Tensors
        context_last = torch.stack(context_last, dim=0) # nq x embedding_dim
        return context_last, {'embed_by_step' : context_by_step, 'attn_by_step' : attn_by_step, 'seq_len' : len_xq}
            # context_last : nq x embedding
            # embed_by_step: embedding at every step for each query [max_xq_length x nq x embedding_dim]
            # attn_by_step : attention over support items at every step for each query [max_xq_length x nq x ns]
            # seq_len : length of each query [nq list]

class EncoderRNN(nn.Module):
    # LSTM encoder for a sequence of symbols
    #
    # The RNN hidden vector (not cell vector) at each step is captured,
    #   for transfer to an attention-based decoder.
    #
    # Does not assume that sequences are sorted by length
    def __init__(self, input_size, embedding_dim, nlayers, dropout_p, bidirectional):
        #
        # Input
        #  input_size : number of input symbols
        #  embedding_dim : number of hidden units in RNN encoder, and size of all embeddings
        #  nlayers : number of hidden layers
        #  dropout : dropout applied to symbol embeddings and RNNs
        #  bidirectional : use a bidirectional LSTM instead and sum of the resulting embeddings
        #
        super(EncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.bi = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=nlayers, dropout=dropout_p, bidirectional=bidirectional)

    def forward(self, z_padded, z_lengths):
        # Input
        #   z_padded : LongTensor (n x max_length); list of n padded input sequences
        #   z_lengths : Python list (length n) for length of each padded input sequence
        #
        # Output
        #   hidden is (n x embedding_size); last hidden state for each input sequence
        #   embed_by_step is (max_length x n x embedding_size); stepwise hidden states for each input sequence
        #   seq_len is tensor of length n; length of each input sequence
        z_embed = self.embedding(z_padded) # n x max_length x embedding_size
        z_embed = self.dropout(z_embed) # n x max_length x embedding_size

        # Sort the sequences by length in descending order
        n = len(z_lengths)
        max_length = max(z_lengths)
        z_lengths = torch.LongTensor(z_lengths)
        if z_embed.is_cuda: z_lengths = z_lengths.cuda()
        z_lengths, perm_idx = torch.sort(z_lengths, descending=True)
        z_embed = z_embed[perm_idx]

        # RNN embedding
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(z_embed, z_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
             # hidden is nlayers*num_directions x n x embedding_size
             # hidden and cell are unpacked, such that they stores the last hidden state for each unpadded sequence
        hidden_by_step, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output) # max_length x n x embedding_size*num_directions

        # If biLSTM, sum the outputs for each direction
        if self.bi:
            hidden_by_step = hidden_by_step.view(max_length, n, 2, self.embedding_dim)
            hidden_by_step = torch.sum(hidden_by_step, 2) # max_length x n x embedding_size
            hidden = hidden.view(self.nlayers, 2, n, self.embedding_dim)
            hidden = torch.sum(hidden, 1) # nlayers x n x embedding_size
        hidden = hidden[-1,:,:] # n x embedding_size (grab the last layer)

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden[unperm_idx,:] # n x embedding_size
        hidden_by_step = hidden_by_step[:,unperm_idx,:] # max_length x n x embedding_size
        seq_len = z_lengths[unperm_idx].tolist()

        return hidden, {"embed_by_step" : hidden_by_step, "seq_len" : seq_len}
                # hidden is (n x embedding_size); last hidden state for each input sequence
                # embed_by_step is (max_length x n x embedding_size); stepwise hidden states for each input sequence
                # seq_len is tensor of length n; length of each input sequence

class WrapperEncoderRNN(EncoderRNN):
    # Wrapper for RNN encoder to behave like MetaNetRNN encoder.
    #  This isn't really doing meta-learning, since it is ignoring the support set entirely.
    #  Instead, it allows us to train a standard sequence-to-sequence model, using the query set as the batch.
    def __init__(self, embedding_dim, input_size, output_size, nlayers, dropout_p=0.1, bidirectional=True, tie_encoders=True):
        super(WrapperEncoderRNN, self).__init__(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
    def forward(self, sample):
        hidden, mydict = super(WrapperEncoderRNN, self).forward(sample['xq_padded'],sample['xq_lengths'])
        mydict['attn_by_step'] = [] # not applicable
        return hidden, mydict
