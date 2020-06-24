import os
import math
import time
import argparse

# Special symbols
SOS_token = "SOS" # start of sentence
EOS_token = "EOS" # end of sentence
PAD_token = SOS_token # padding symbol

# Adjustable parameters
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for CometML logging')
    parser.add_argument('--evaluate_every', type=int, default=100, help='Episodes before evaluating model')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Episodes before saving model checkpoint')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generators')

    parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes for training')
    parser.add_argument('--max_try_novel', type=int, default=100, help='number of attempts to find a novel episode (not in tabu list) before throwing an error')
    parser.add_argument('--num_episodes_val', type=int, default=5, help='number of episodes to use as validation throughout learning')
    parser.add_argument('--clip', type=float, default=50.0, help='clip gradients with larger magnitude than this')
    parser.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
    parser.add_argument('--lr_decay_schedule', type=int, default=5000, help='decrease learning rate by 1e-1 after this many episodes')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers in the LSTM')
    parser.add_argument('--max_length_eval', type=int, default=50, help='maximum generated sequence length when evaluating the network')
    parser.add_argument('--emb_size', type=int, default=200, help='size of sequence embedding (also, nhidden for encoder and decoder LSTMs)')
    parser.add_argument('--dropout', type=float, default=0.5, help=' dropout applied to embeddings and LSTMs')
    parser.add_argument('--fn_out_model', type=str, default='', help='filename for saving the model')
    parser.add_argument('--dir_model', type=str, default='out_models', help='directory for saving model files')
    parser.add_argument('--episode_type', type=str, default='ME',
        choices=["ME", "scan_prim_permutation", "scan_prim_augmentation", "scan_around_right", "scan_length"],
        help='what type of episodes do we want')
    parser.add_argument('--disable_memory', action='store_true', help='Disable external memory, ignore support set, and use simple RNN encoder')
    parser.add_argument('--disable_attention', action='store_true', help='Disable the decoder attention')
    parser.add_argument('--disable_recon_loss', action='store_true', help='Disable reconstruction loss, where support items are included also as query items')
    parser.add_argument('--use_cuda', action='store_true', help='Enable cuda')

    return parser, parser.parse_args()

def asMinutes(s):
    # convert seconds to minutes
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    # prints time elapsed and estimated time remaining
    #
    # Input
    #  since : previous time
    #  percent : amount of training complete
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def extract(include,arr):
    # Create a new list only using the included (boolean) elements of arr
    #
    # Input
    #  include : [n len] boolean array
    #  arr [ n length array]
    assert len(include)==len(arr)
    return [a for idx, a in enumerate(arr) if include[idx]]

def display_input_output(input_patterns,output_patterns,target_patterns):
    # Verbose analysis of performance on query items
    #
    # Input
    #   input_patterns : list of input sequences (each in list form)
    #   output_patterns : list of output sequences, which are actual outputs (each in list form)
    #   target_patterns : list of targets
    nq = len(input_patterns)
    if nq == 0:
        print('     no patterns')
        return
    for q in range(nq):
        assert isinstance(input_patterns[q],list)
        assert isinstance(output_patterns[q],list)
        is_correct = output_patterns[q] == target_patterns[q]
        print('     ',end='')
        print(' '.join(input_patterns[q]),end='')
        print(' -> ',end='')
        print(' '.join(output_patterns[q]),end='')
        if not is_correct:
            print(' (** target: ',end='')
            print(' '.join(target_patterns[q]),end='')
            print(')',end='')
        print('')

def tabu_update(tabu_list,identifier):
    # Add all elements of "identifier" to the 'tabu_list', and return updated list
    if isinstance(identifier,(list,set)):
        tabu_list = tabu_list.union(identifier)
    elif isinstance(identifier,str):
        tabu_list.add(identifier)
    else:
        assert False
    return tabu_list
