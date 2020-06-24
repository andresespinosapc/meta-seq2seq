# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from comet_ml import Experiment

import random
import os
import time
from copy import deepcopy, copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import describe_model, BaselineModel
from masked_cross_entropy import *
from generate_episode import get_episode_generator
from utils import timeSince, display_input_output, extract, tabu_update, read_args
from utils import SOS_token, EOS_token, PAD_token


comet_args = {
    'project_name': os.environ.get('COMET_PROJECT_NAME'),
    'workspace': os.environ.get('COMET_WORKSPACE'),
}
if os.environ.get('COMET_DISABLE'):
    comet_args['disabled'] = True
    comet_args['api_key'] = ''
if os.environ.get('COMET_OFFLINE'):
    comet_args['api_key'] = ''
    comet_args['offline_directory'] = 'comet_offline'
    experiment = OfflineExperiment(**comet_args)
else:
    experiment = Experiment(**comet_args)

def log_comet_parameters(args):
    args_dict = vars(args)
    for key in args_dict.keys():
        experiment.log_parameter(key, args_dict[key])

def validate_args(parser, args):
    if args.exp_name is None and not comet_args.get('disabled'):
        parser.error('Please provide exp_name if logging to CometML')

# --
# Main routine for training meta seq2seq models
# --

# We based the seq2seq code on the PyTorch tutorial of Sean Robertson
#   https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

# Robertson's asMinutes and timeSince helper functions to print time elapsed and estimated time
# remaining given the current time and progress

def evaluation_battery(sample_eval_list, model, input_lang, output_lang, max_length, verbose=False):
    # Evaluate a list of episodes
    #
    # Input
    #   sample_eval_list : list of evaluation sets to iterate through
    #   ...
    #   input_lang: Language object for input sequences
    #   output_lang: Language object for output sequences
    #   max_length : maximum length of a generated sequence
    #   verbose : print outcome or not?
    #
    # Output
    #   (acc_novel, acc_autoencoder) average accuracy for novel items in query set, and support items in query set
    list_acc_val_novel = []
    list_acc_val_autoencoder = []
    for idx,sample in enumerate(sample_eval_list):
        acc_val_novel, acc_val_autoencoder, yq_predict, in_support, all_attention_by_query, memory_attn_steps = evaluate(sample, model, input_lang, output_lang, max_length)
        list_acc_val_novel.append(acc_val_novel)
        list_acc_val_autoencoder.append(acc_val_autoencoder)
        if verbose:
            print('')
            print('Evaluation episode ' + str(idx))
            if sample['grammar']:
                print("")
                print(sample['grammar'])
            print('  support items: ')
            display_input_output(sample['xs'],sample['ys'],sample['ys'])
            print('  retrieval items; ' + str(round(acc_val_autoencoder,3)) + '% correct')
            display_input_output(extract(in_support,sample['xq']),extract(in_support,yq_predict),extract(in_support,sample['yq']))
            print('  generalization items; ' + str(round(acc_val_novel,3)) + '% correct')
            display_input_output(extract(np.logical_not(in_support),sample['xq']),extract(np.logical_not(in_support),yq_predict),extract(np.logical_not(in_support),sample['yq']))
    return np.mean(list_acc_val_novel), np.mean(list_acc_val_autoencoder)

def evaluate(sample, model, input_lang, output_lang, max_length):
    encoder = model.encoder
    decoder = model.decoder
    # Evaluate an episode
    #
    # Input
    #   sample : [dict] generated validation episode, produced by "build_sample"
    #   ...
    #   input_lang: Language object for input sequences
    #   output_lang: Language object for output sequences
    #   max_length : maximum length of generated output sequence
    #
    # Output
    #   acc_novel : accuracy (percent correct) on novel items in query set
    #   acc_autoencoder : accuracy (percent correct) on reconstructing support items in query set
    #   yq_predict : list of predicted output sequences for all items
    #   is_support : [n x 1 bool] indicates for each query item whether it is in the support set
    #   all_attn_by_time : list (over time step) of batch_size x max_input_length tensors
    #   memory_attn_steps : attention over support items at every step for each query [max_xq_length x nq x ns]
    model.eval()

    # Prepare input and output variables
    nq = len(sample['yq'])

    # run model
    all_decoder_outputs, memory_attn_steps, all_attn_by_time = model(sample, output_lang)

    # get predictions
    in_support = np.array([x in sample['xs'] for x in sample['xq']])
    yq_predict = []
    for q in range(nq):
        myseq = output_lang.symbolsFromVector(all_decoder_outputs[q,:])
        yq_predict.append(myseq)

    # compute accuracy
    v_acc = np.zeros(nq)
    for q in range(nq):
        v_acc[q] = yq_predict[q] == sample['yq'][q]
    acc_autoencoder = np.mean(v_acc[in_support])*100.
    acc_novel = np.mean(v_acc[np.logical_not(in_support)])*100.
    return acc_novel, acc_autoencoder, yq_predict, in_support, all_attn_by_time, memory_attn_steps

def train(sample, model, optimizer, input_lang, output_lang):
    # Update the model for a single training episode
    #
    # Input
    #   sample : [dict] generated training episode, produced by "build_sample"
    #   ...
    #   input_lang: Language object for input sequences
    #   output_lang: Language object for output sequences
    #
    model.train()

    target_batches = torch.transpose(sample['yq_padded'], 0, 1) # (max_length x nq tensor) ... batch targets with padding
    target_lengths = sample['yq_lengths']

    # run model
    all_decoder_outputs = model(sample, output_lang)

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        torch.transpose(all_decoder_outputs, 0, 1).contiguous(), # -> nq x max_length
        torch.transpose(target_batches, 0, 1).contiguous(), # nq x max_length
        target_lengths
    )

    # gradient update
    optimizer.zero_grad()
    loss.backward()
    encoder_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), args.clip)
    decoder_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.clip)
    if max(encoder_norm, decoder_norm) > args.clip:
        print("Gradient clipped:")
        print("  Encoder norm: " + str(encoder_norm))
        print("  Decoder norm: " + str(decoder_norm))
    optimizer.step()
    return loss.cpu().item()

if __name__ == "__main__":
    # Adjustable parameters
    parser, args = read_args()
    validate_args(parser, args)
    experiment.set_name(args.exp_name)
    log_comet_parameters(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fn_out_model = args.fn_out_model
    episode_type = args.episode_type
    dir_model = args.dir_model
    num_episodes = args.num_episodes
    adam_learning_rate = args.lr
    lr_decay_schedule = args.lr_decay_schedule
    max_length_eval = args.max_length_eval
    use_attention = not args.disable_attention
    args.use_cuda = args.use_cuda if torch.cuda.is_available() else False

    if fn_out_model=='':
        fn_out_model = experiment.get_key()
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    fn_out_model = os.path.join(dir_model, fn_out_model)


    if not os.path.isfile(fn_out_model):
        print("Training a new network...")
        print("  Episode type is " + episode_type)
        generate_episode_train, generate_episode_test, input_lang, output_lang = get_episode_generator(args)
        if args.use_cuda:
            print('  Training on GPU ' + str(torch.cuda.current_device()), end='')
        else:
            print('  Training on CPU', end='')
        print(' for ' + str(num_episodes) + ' episodes')

        model = BaselineModel(args, input_lang, output_lang)
        if args.use_cuda:
            model = model.cuda()
        criterion = nn.NLLLoss()
        print('  Set learning rate to ' + str(adam_learning_rate))
        optimizer = optim.Adam(model.parameters(),lr=adam_learning_rate)
        print("")
        print("Architecture options...")
        print(" Decoder attention is USED" ) if use_attention else print(" Decoder attention is NOT used")
        print(" External memory is USED" ) if not args.disable_memory else print(" External memory is NOT used")
        print(" Reconstruction loss is USED" ) if not args.disable_recon_loss else print(" Reconstruction loss is NOT used")
        print("")
        describe_model(model)

        # create validation episodes
        tabu_episodes = set([])
        samples_val = []
        for i in range(args.num_episodes_val):
            sample = generate_episode_test(tabu_episodes)
            samples_val.append(sample)
            tabu_episodes = tabu_update(tabu_episodes,sample['identifier'])

        # train over a set of random episodes
        avg_train_loss = 0.
        counter = 0 # used to count updates since the loss was last reported
        start = time.time()
        for episode in range(1,num_episodes+1):

            # generate a random episode
            sample = generate_episode_train(tabu_episodes)

            # batch updates (where batch includes the entire support set)
            train_loss = train(sample, model, optimizer, input_lang, output_lang)
            avg_train_loss += train_loss
            counter += 1

            if episode == 1 or episode % args.evaluate_every == 0 or episode == num_episodes:
                acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, model, input_lang, output_lang, max_length_eval)
                print('{:s} ({:d} {:.0f}% finished) TrainLoss: {:.4f}, ValAccRetrieval: {:.1f}, ValAccGeneralize: {:.1f}'.format(timeSince(start, float(episode) / float(num_episodes)),
                                         episode, float(episode) / float(num_episodes) * 100., avg_train_loss/counter, acc_val_retrieval, acc_val_gen))

                metrics = {
                    'episode': episode,
                    'train_loss': avg_train_loss / counter,
                    'acc_val_gen': acc_val_gen,
                    'acc_val_retrieval': acc_val_retrieval
                }
                experiment.log_metrics(metrics)

                avg_train_loss = 0.
                counter = 0

            if episode % args.checkpoint_every == 0 or episode == num_episodes:
                state = {'state_dict': model.state_dict(),
                            'input_lang': input_lang,
                            'output_lang': output_lang,
                            'episodes_validation': samples_val,
                            'episode_type': episode_type,
                            'emb_size':args.emb_size,
                            'dropout':args.dropout,
                            'nlayers':args.nlayers,
                            'episode':episode,
                            'disable_memory':args.disable_memory,
                            'disable_recon_loss':args.disable_recon_loss,
                            'use_attention':use_attention,
                            'max_length_eval':max_length_eval,
                            'num_episodes':num_episodes,
                            'args':args}
                print('Saving model as: ' + fn_out_model)
                torch.save(state, fn_out_model)

            # decay learning rate according to schedule
            # TODO: CURRENTLY resets optimizer data!
            if episode % lr_decay_schedule == 0:
                adam_learning_rate = adam_learning_rate * 0.1
                optimizer = optim.Adam(model.parameters(), lr=adam_learning_rate)
                print('Set learning rate to ' + str(adam_learning_rate))

        print('Training complete')
        acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, model, input_lang, output_lang, max_length_eval, verbose=False)
        print('Acc Retrieval (val): ' + str(round(acc_val_retrieval,1)))
        print('Acc Generalize (val): ' + str(round(acc_val_gen,1)))
    else: # evaluate model if filename already exists
        print('Results file already exists. Loading file and evaluating...')
        print('Loading model: ' + fn_out_model)
        checkpoint = torch.load(fn_out_model, map_location='cpu') # evaluate model on CPU
        if 'episode' in checkpoint: print(' Loading epoch ' + str(checkpoint['episode']) + ' of ' + str(checkpoint['num_episodes']))
        input_lang = checkpoint['input_lang']
        output_lang = checkpoint['output_lang']
        emb_size = checkpoint['emb_size']
        nlayers = checkpoint['nlayers']
        dropout_p = checkpoint['dropout']
        input_size = input_lang.n_symbols
        output_size = output_lang.n_symbols
        samples_val = checkpoint['episodes_validation']
        disable_memory = checkpoint['disable_memory']
        max_length_eval = checkpoint['max_length_eval']
        if 'args' not in checkpoint or 'disable_attention' not in checkpoint['args']:
            use_attention = True
        else:
            args = checkpoint['args']
            use_attention = not args.disable_attention
        args.use_cuda = False
        model = BaselineModel(args, input_lang, output_lang)
        model.load_state_dict(checkpoint['state_dict'])
        describe_model(model)

        generate_episode_train, generate_episode_test, input_lang,output_lang = get_episode_generator(args)
        acc_train_gen, acc_train_retrieval = evaluation_battery([generate_episode_train([]) for _ in range(5)], model, input_lang, output_lang, max_length_eval, verbose=False)
        print('Acc Retrieval (train): ' + str(round(acc_train_retrieval,1)))
        print('Acc Generalize (train): ' + str(round(acc_train_gen,1)))

        acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, model, input_lang, output_lang, max_length_eval, verbose=True)
        print('Acc Retrieval (val): ' + str(round(acc_val_retrieval,1)))
        print('Acc Generalize (val): ' + str(round(acc_val_gen,1)))
