import torch
import torch.nn as nn
import numpy as np

from .encoders import MetaNetRNN, EncoderRNN, WrapperEncoderRNN
from .decoders import AttnDecoderRNN, DecoderRNN

# TODO: currently duplicated in utils
SOS_token = "SOS" # start of sentence

class BaselineModel(nn.Module):
    def __init__(self, args, input_lang, output_lang):
        super(BaselineModel, self).__init__()
        self.args = args
        input_size = input_lang.n_symbols
        output_size = output_lang.n_symbols

        if args.disable_memory:
            self.encoder = WrapperEncoderRNN(args.emb_size, input_size, output_size, args.nlayers, args.dropout)
        else:
            self.encoder = MetaNetRNN(args.emb_size, input_size, output_size, args.nlayers, args.dropout)
        if not args.disable_attention:
            self.decoder = AttnDecoderRNN(args.emb_size, output_size, args.nlayers, args.dropout)
        else:
            self.decoder = DecoderRNN(args.emb_size, output_size, args.nlayers, args.dropout)

    def forward(self, sample, output_lang):
        if self.training:
            # Run words through encoder
            encoder_embedding, dict_encoder = self.encoder(sample)
            encoder_embedding_steps = dict_encoder['embed_by_step']

            # Prepare input and output variables
            nq = len(sample['yq']) # number of queries
            decoder_input = torch.tensor([output_lang.symbol2index[SOS_token]]*nq) # nq length tensor
            decoder_hidden = self.decoder.initHidden(encoder_embedding)
            target_batches = torch.transpose(sample['yq_padded'], 0, 1) # (max_length x nq tensor) ... batch targets with padding
            target_lengths = sample['yq_lengths']
            max_target_length = max(target_lengths)
            all_decoder_outputs = torch.zeros(max_target_length, nq, self.decoder.output_size)
            if self.args.use_cuda:
                decoder_input = decoder_input.cuda()
                target_batches = target_batches.cuda()
                all_decoder_outputs = all_decoder_outputs.cuda()

            # Run through decoder one time step at a time
            for t in range(max_target_length):
                if type(self.decoder) is AttnDecoderRNN:
                    decoder_output, decoder_hidden, attn_by_query = self.decoder(decoder_input, decoder_hidden, encoder_embedding_steps)
                elif type(self.decoder) is DecoderRNN:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                else:
                    assert False
                all_decoder_outputs[t] = decoder_output # max_len x nq x output_size
                decoder_input = target_batches[t]

            return all_decoder_outputs
        else:
            # Run words through encoder
            encoder_embedding, dict_encoder = self.encoder(sample)
            encoder_embedding_steps = dict_encoder['embed_by_step']
            memory_attn_steps = dict_encoder['attn_by_step']

            # Prepare input and output variables
            nq = len(sample['yq'])
            decoder_input = torch.tensor([output_lang.symbol2index[SOS_token]]*nq) # nq length tensor
            decoder_hidden = self.decoder.initHidden(encoder_embedding)

            # Store output words and attention states
            decoded_words = []

            # Run through decoder
            all_decoder_outputs = np.zeros((nq, self.args.max_length_eval), dtype=int)
            all_attn_by_time = [] # list (over time step) of batch_size x max_input_length tensors
            if self.args.use_cuda:
                decoder_input = decoder_input.cuda()
            for t in range(self.args.max_length_eval):
                if type(self.decoder) is AttnDecoderRNN:
                    decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_embedding_steps)
                    all_attn_by_time.append(attn_weights)
                elif type(self.decoder) is DecoderRNN:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                else:
                    assert False

                # Choose top symbol from output
                topv, topi = decoder_output.cpu().data.topk(1)
                decoder_input = topi.view(-1)
                if self.args.use_cuda:
                    decoder_input = decoder_input.cuda()
                all_decoder_outputs[:,t] = topi.numpy().flatten()

            return all_decoder_outputs, memory_attn_steps, all_attn_by_time
