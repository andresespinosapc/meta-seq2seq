import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import Attn

class AttnDecoderRNN(nn.Module):
    #
    # One-step batch LSTM decoder with Luong et al. attention
    #
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1):
        #
        # Input
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        #
        super(AttnDecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=nlayers, dropout=dropout_p)
        self.attn = Attn()
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        #
        # Run batch decoder forward for a single time step.
        #  Each decoder step considers all of the encoder_outputs through attention.
        #  Attention retrieval is based on decoder hidden state (not cell state)
        #
        # Input
        #  input: LongTensor of length batch_size (single step indices for batch)
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #  encoder_outputs: all encoder outputs for attention, max_input_length x batch_size x embedding_size
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #   attn_weights : attention weights, batch_size x max_input_length
        #
        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input) # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0) # S=1 x batch_size x hidden_size

        rnn_output, hidden = self.rnn(embedding, last_hidden)
            # rnn_output is S=1 x batch_size x hidden_size
            # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)

        context, attn_weights = self.attn(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1), encoder_outputs.transpose(0,1))
            # context : batch_size x 1 x hidden_size
            # attn_weights : batch_size x 1 x max_input_length

        # Concatenate the context vector and RNN hidden state, and map to an output
        rnn_output = rnn_output.squeeze(0) # batch_size x hidden_size
        context = context.squeeze(1) # batch_size x hidden_size
        attn_weights = attn_weights.squeeze(1) # batch_size x max_input_length
        concat_input = torch.cat((rnn_output, context), 1) # batch_size x hidden_size*2
        concat_output = self.tanh(self.concat(concat_input)) # batch_size x hidden_size
        output = self.out(concat_output) # batch_size x output_size
        return output, hidden, attn_weights
            # output : [unnormalized probabilities] batch_size x output_size
            # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)
            # attn_weights: tensor of size (batch_size x max_input_length)

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder.
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0) # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers,-1,-1).contiguous() # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)

class DecoderRNN(nn.Module):
    #
    # One-step simple batch LSTM decoder with no attention
    #
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1):
        # Input
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        super(DecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=nlayers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden):
        # Run batch decoder forward for a single time step.
        #
        # Input
        #  input: LongTensor of length batch_size
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #
        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input) # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0) # S=1 x batch_size x hidden_size
        rnn_output, hidden = self.rnn(embedding, last_hidden)
            # rnn_output is S=1 x batch_size x hidden_size
            # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)
        rnn_output = rnn_output.squeeze(0) # batch_size x hidden_size
        output = self.out(rnn_output) # batch_size x output_size
        return output, hidden
            # output : [unnormalized probabilities] batch_size x output_size
            # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder.
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0) # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers,-1,-1).contiguous() # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)
