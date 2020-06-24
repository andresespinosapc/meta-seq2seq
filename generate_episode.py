# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
from copy import copy, deepcopy
import torch

from utils import SOS_token, EOS_token, PAD_token

class Lang:
    # Class for converting strings/words to numerical indices, and vice versa.
    #  Should use separate class for input language (English) and output language (actions)
    #
    def __init__(self, symbols, args):
        # symbols : list of all possible symbols
        n = len(symbols)
        self.symbols = symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token}
        self.symbol2index = {SOS_token : n, EOS_token : n+1}
        for idx,s in enumerate(symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx
        self.n_symbols = len(self.index2symbol)
        self.args = args

    def variableFromSymbols(self, mylist, add_eos=True):
        # Convert a list of symbols to a tensor of indices (adding a EOS token at end)
        #
        # Input
        #  mylist : list of m symbols
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] indices of each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos:
            mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices)
        if self.args.use_cuda:
            output = output.cuda()
        return output

    def symbolsFromVector(self, v):
        # Convert indices to symbols, breaking where we get a EOS token
        #
        # Input
        #  v : list of m indices
        #
        # Output
        #  mylist : list of m or m-1 symbols (excluding EOS)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist


def make_hashable(G):
    # Separate and sort stings, to make unique string identifier for an episode
    #
    # Input
    #   G : string of elements separate by \n, specifying the structure of an episode
    G_str = str(G).split('\n')
    G_str.sort()
    out = '\n'.join(G_str)
    return out.strip()

def get_unique_words(sentences):
    # Get a list of all the unique words in a list of sentences
    #
    # Input
    #  sentences: list of sentence strings
    # Output
    #   words : list of all unique words in sentences
    words = []
    for s in sentences:
        for w in s.split(' '): # words
            if w not in words:
                words.append(w)
    return words

def pad_seq(seq, max_length):
    # Pad sequence with the PAD_token symbol to achieve max_length
    #
    # Input
    #  seq : list of symbols
    #
    # Output
    #  seq : padded list now extended to length max_length
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def build_padded_var(list_seq, lang, args):
    # Transform python list to a padded torch tensor
    #
    # Input
    #  list_seq : list of n sequences (each sequence is a python list of symbols)
    #  lang : language object for translation into indices
    #
    # Output
    #  z_padded : LongTensor (n x max_length)
    #  z_lengths : python list of sequence lengths (list of scalars)
    n = len(list_seq)
    if n==0: return [],[]
    z_eos = [z+[EOS_token] for z in list_seq]
    z_lengths = [len(z) for z in z_eos]
    max_len = max(z_lengths)
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.variableFromSymbols(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded,dim=0)
    if args.use_cuda:
        z_padded = z_padded.cuda()
    return z_padded,z_lengths

def build_sample(args,x_support,y_support,x_query,y_query,input_lang,output_lang,myhash,grammar=''):
    # Build an episode from input/output examples
    #
    # Input
    #  x_support [length ns list of lists] : input sequences (each a python list of words/symbols)
    #  y_support [length ns list of lists] : output sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : input sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : output sequences (each a python list of words/symbols)
    #  input_lang: Language object for input sequences (see Language)
    #  output_lang: Language object for output sequences
    #  myhash : unique string identifier for this episode (should be order invariant for examples)
    #  grammar : (optional) grammar object
    #
    # Output
    #  sample : dict that stores episode information
    sample = {}

    # store input and output sequences
    sample['identifier'] = myhash
    sample['xs'] = x_support
    sample['ys'] = y_support
    sample['xq'] = x_query
    sample['yq'] = y_query
    sample['grammar'] = grammar

    # convert strings to indices, pad, and create tensors ready for input to network
    sample['xs_padded'],sample['xs_lengths'] = build_padded_var(x_support,input_lang, args) # (ns x max_length)
    sample['ys_padded'],sample['ys_lengths'] = build_padded_var(y_support,output_lang, args) # (ns x max_length)
    sample['xq_padded'],sample['xq_lengths'] = build_padded_var(x_query,input_lang, args) # (nq x max_length)
    sample['yq_padded'],sample['yq_lengths'] = build_padded_var(y_query,output_lang, args) # (nq x max_length)
    return sample

# --
# Generating episodes for meta learning.
# --

def sample_ME_concat_data(nquery,input_symbols,output_symbols,maxlen,maxntry=500,inc_support_in_query=False):
	# Sample ME episode based on current ordering of input/output symbols (already randomized)
	#
	# Input
	#  nquery : number of query examples
	#  input_symbols : list of nprim input symbols (already permuted)
	#  output_symbols : list of nprim output symbols (already permuted)
	#  maxlen : maximum sequence length in query set
	#  inc_support_in_query : true/false, where true indicates that we include the "support loss" in paper (default=False)
	nprim = len(input_symbols)
	pairs = list(zip(input_symbols,output_symbols))

	# support set with all singleton primitives
	D_support = []
	for dat in pairs[:nprim-1]:
		D_support.append(dat)

	# query set with random concatenations
	D_query = set([])
	ntry = 0
	while len(D_query)<nquery:
		mylen = random.randint(2,maxlen)
		dat_list = [random.choice(pairs) for _ in range(mylen)]
		dat_in, dat_out = zip(*dat_list)
		dat_in = ' '.join(dat_in)
		dat_out = ' '.join(dat_out)
		D_query.add((dat_in,dat_out))
		if ntry > maxntry:
			raise Exception('Maximum number of tries to generate valid dataset')
	D_query = list(D_query)
	if inc_support_in_query:
		D_query += deepcopy(D_support)

	return D_support, D_query

def load_scan_file(mytype,split):
	# Load SCAN dataset from file
	#
	# Input
	#  mytype : type of SCAN experiment
	#  split : 'train' or 'test'
	#
	# Output
	#  commands : list of input/output strings (as tuples)
	assert mytype in ['simple','addprim_jump','length','addprim_turn_left','all','template_around_right','viz','examine']
	assert split in ['train','test']
	fn = 'data/tasks_' + split + '_' + mytype + '.txt'
	fid = open(fn,'r')
	lines = fid.readlines()
	fid.close()
	lines = [l.strip() for l in lines]
	lines = [l.lstrip('IN: ') for l in lines]
	commands = [l.split(' OUT: ') for l in lines]
	return commands

def sentence_replace_var(sentence,list_source,list_target):
	# Swap each source word in sentence with corresponding target word
	#
	# Input
	#  sentence: string of words
	#  list_source : length k list of words to be replaced
	#  list_target : length k list of words to replace source words
	#
	# Output
	#   sentence: new string of words
	assert(len(list_source)==len(list_target))
	for i in range(len(list_source)):
		sentence = sentence.replace(list_source[i],list_target[i])
	return sentence

def load_scan_var(mytype,split):
	# Load SCAN tasks from file and replace action primitives (walk, look, run, jump) with variables
	#   Replace all input primitives with deterministic placeholders primitive1, primitive2, etc.
	#   Replace all output primitives with deterministic placeholders I_ACT_1, I_ACT_2, etc.
	#
	# Input
	#  mytype : type of SCAN experiment
	#  split : 'train' or 'test'
	#
	# Output
	#  commands : list of input/output strings (as tuples)
	scan_tuples = load_scan_file(mytype,split)
	scan_tuples_variable = deepcopy(scan_tuples)

	# original primitives
	scan_primitive_tuples = [('walk','I_WALK'),('look','I_LOOK'),('run','I_RUN'),('jump','I_JUMP')]
	nprim = len(scan_primitive_tuples)
	list_source_command = [p[0] for p in scan_primitive_tuples] # each input primitive in source
	list_source_output = [p[1] for p in scan_primitive_tuples] # each output primitive in source

	# replacement placeholder primitives
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim+1)]
	list_target_command = [p[0] for p in scan_placeholder_tuples] # each input primitive as target
	list_target_output = [p[1] for p in scan_placeholder_tuples] # each output primitive as target

	# do replacement
	for i in range(len(scan_tuples_variable)):
		scan_tuples_variable[i][0] = sentence_replace_var(scan_tuples_variable[i][0], list_source_command, list_target_command)
		scan_tuples_variable[i][1] = sentence_replace_var(scan_tuples_variable[i][1], list_source_output, list_target_output)
	return scan_tuples_variable

def load_scan_dir_var(mytype,split):
	# Load SCAN tasks from file and replace turning primitives (right, left) with variables
	#   Replace all input primitives with deterministic placeholders primitive1, primitive2, etc.
	#   Replace all output primitives with deterministic placeholders I_ACT_1, I_ACT_2, etc.
	#
	# Input
	#  mytype : type of SCAN experiment
	#  split : 'train' or 'test'
	#
	# Output
	#  commands : list of input/output strings
	scan_tuples = load_scan_file(mytype,split)
	scan_tuples_variable = deepcopy(scan_tuples)

	# original primitives
	scan_primitive_tuples = [('right','I_TURN_RIGHT'),('left','I_TURN_LEFT')]
	nprim = len(scan_primitive_tuples)
	list_source_command = [p[0] for p in scan_primitive_tuples] # each input primitive in source
	list_source_output = [p[1] for p in scan_primitive_tuples] # each output primitive in source

	# replacement placeholder primitives
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim+1)]
	list_target_command = [p[0] for p in scan_placeholder_tuples] # each input primitive as target
	list_target_output = [p[1] for p in scan_placeholder_tuples] # each output primitive as target

	# do replacement
	for i in range(len(scan_tuples_variable)):
		scan_tuples_variable[i][0] = sentence_replace_var(scan_tuples_variable[i][0], list_source_command, list_target_command)
		scan_tuples_variable[i][1] = sentence_replace_var(scan_tuples_variable[i][1], list_source_output, list_target_output)
	return scan_tuples_variable

def sample_augment_scan(nsupport,nquery,scan_tuples_variable,shuffle,nextra=0,inc_support_in_query=False):
	# Both the query and the support set contain example input/output patterns
	#  Create an episode with shuffled input/output primitives (run, jump, walk, look), potentially with primitive augmentation
	#
	# Input
	#  nsupport : number of support items to sample
	#  nquery : number of query items to sample
	#  scan_tuples_variable : list of input/output tuples to draw from (in VARIABLE form, as generated by 'load_scan_var')
	#  shuffle : true/false; randomly shuffle the scan input/output primitives, or use semantic alignment?
	#  nextra : number of extra abstract input/output primitives to include
	#  inc_support_in_query : true/false; include the support set in the query set? (e.g., use support loss)
	#
	# Output
	#  D_support : list of support input/output pairs; in this case, it's the remapped primitives ONLY
	#  D_query : list of query input/output pairs
	#  D_primitive : list of primitive input/output pairs
	#
	# Sample query items from scan tuples (before variable replacement)
	# Distribute the patterns to query and support items
	scan_tuples_variable = deepcopy(scan_tuples_variable)
	random.shuffle(scan_tuples_variable)
	D_query = scan_tuples_variable[:nquery]
	D_support = scan_tuples_variable[nquery:nquery+nsupport]

	# Shuffle assignment of primitive commands to primitive actions
	nprim_replace = 4
	scan_primitive_tuples = [('walk','I_WALK'),('look','I_LOOK'),('run','I_RUN'),('jump','I_JUMP')]
	if nextra > 0:
		scan_primitives_extra = [(str(i),'I_'+str(i)) for i in range(1,nextra+1)]
		scan_primitive_tuples += scan_primitives_extra
	unzip = list(zip(*scan_primitive_tuples))
	list_target_command = list(unzip[0])
	list_target_output = list(unzip[1])
	if shuffle: # shuffle assignment if desired
		random.shuffle(list_target_command)
		random.shuffle(list_target_output)
	list_target_command = list_target_command[:nprim_replace]
	list_target_output = list_target_output[:nprim_replace]

	# Replace placeholders with grounded commands and actions
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim_replace+1)]
	list_source_command = [p[0] for p in scan_placeholder_tuples]
	list_source_output = [p[1] for p in scan_placeholder_tuples]
	for i in range(len(D_query)):
		D_query[i][0] = sentence_replace_var(D_query[i][0], list_source_command, list_target_command)
		D_query[i][1] = sentence_replace_var(D_query[i][1], list_source_output, list_target_output)
	for i in range(len(D_support)):
		D_support[i][0] = sentence_replace_var(D_support[i][0], list_source_command, list_target_command)
		D_support[i][1] = sentence_replace_var(D_support[i][1], list_source_output, list_target_output)

	D_primitive = list(zip(list_target_command,list_target_output))

	if inc_support_in_query:
		D_query += deepcopy(D_support)
	return D_support, D_query, D_primitive

def sample_augment_direction_scan(nsupport,nquery,scan_tuples_variable,shuffle,nextra=0,inc_support_in_query=False):
	# Both the query and the support set contain example input/output patterns
	#  Create an episode with shuffled input/output directions (right, left, etc.), potentially with augmentation
	#
	# Input
	#  nsupport : number of support items to sample
	#  nquery : number of query items to sample
	#  scan_tuples_variable : list of input/output tuples to draw from (in VARIABLE form, as generated by 'load_scan_var')
	#  shuffle : true/false; randomly shuffle the scan input/output primitives, or use semantic alignment?
	#  nextra : number of extra abstract input/output primitives to include
	#  inc_support_in_query : true/false; include the support set in the query set?
	#
	# Output
	#  D_support : list of support input/output pairs; in this case, it's the remapped primitives ONLY
	#  D_query : list of query input/output pairs
	#  D_primitive : list of primitive input/output pairs
	#
	# Sample query items from scan tuples (before variable replacement)
	# Distribute the patterns to query and support items
	scan_tuples_variable = deepcopy(scan_tuples_variable)
	random.shuffle(scan_tuples_variable)
	D_query = scan_tuples_variable[:nquery]
	D_support = scan_tuples_variable[nquery:nquery+nsupport]

	# Shuffle assignment of primitive commands to primitive actions
	nprim_replace = 2
	scan_primitive_tuples = [('right','I_TURN_RIGHT'),('left','I_TURN_LEFT')]
	if nextra > 0:
		scan_primitives_extra = [(str(i),'I_'+str(i)) for i in range(1,nextra+1)]
		scan_primitive_tuples += scan_primitives_extra
	unzip = list(zip(*scan_primitive_tuples))
	list_target_command = list(unzip[0])
	list_target_output = list(unzip[1])
	if shuffle: # shuffle assignment if desired
		random.shuffle(list_target_command)
		random.shuffle(list_target_output)
	list_target_command = list_target_command[:nprim_replace]
	list_target_output = list_target_output[:nprim_replace]

	# Replace placeholders with grounded commands and actions
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim_replace+1)]
	list_source_command = [p[0] for p in scan_placeholder_tuples]
	list_source_output = [p[1] for p in scan_placeholder_tuples]
	for i in range(len(D_query)):
		D_query[i][0] = sentence_replace_var(D_query[i][0], list_source_command, list_target_command)
		D_query[i][1] = sentence_replace_var(D_query[i][1], list_source_output, list_target_output)
	for i in range(len(D_support)):
		D_support[i][0] = sentence_replace_var(D_support[i][0], list_source_command, list_target_command)
		D_support[i][1] = sentence_replace_var(D_support[i][1], list_source_output, list_target_output)

	D_primitive = list(zip(list_target_command,list_target_output))
	if inc_support_in_query:
		D_query += deepcopy(D_support)
	return D_support, D_query, D_primitive

def sample_augment_scan_separate(nsupport,nquery,scan_tuples_support_variable,scan_tuples_query_variable,shuffle,nextra=0,inc_support_in_query=False):
	# ** This version takes a SEPARATE set of examples for sampling the support and query items **
	#  Both the query and the support set contain example input/output patterns
	#   Create an episode with shuffled input/output primitives, potentially with augmentation
	#
	# Input
	#  nsupport : number of support items to sample
	#  nquery : number of query items to sample
	#  scan_tuples_support_variable : list of input/output tuples to draw support examples from (in VARIABLE form, as generated by 'load_scan_var')
	#  scan_tuples_query_variable : list of input/output tuples to draw query examples from (in VARIABLE form, as generated by 'load_scan_var')
	#  shuffle : true/false; randomly shuffle the scan input/output primitives, or use semantic alignment?
	#  nextra : number of extra abstract input/output primitives to include
	#  inc_support_in_query : true/false; include the support set in the query set?
	#
	# Output
	#  D_support : list of support input/output pairs; in this case, it's the remapped primitives ONLY
	#  D_query : list of query input/output pairs
	#  D_primitive : list of primitive input/output pairs
	#
	# Sample query items from scan tuples (before variable replacement)
	# Distribute the patterns to query and support items
	scan_tuples_support_variable = deepcopy(scan_tuples_support_variable)
	random.shuffle(scan_tuples_support_variable)
	D_support = scan_tuples_support_variable[:nsupport]

	scan_tuples_query_variable = deepcopy(scan_tuples_query_variable)
	random.shuffle(scan_tuples_query_variable)
	D_query = scan_tuples_query_variable[:nquery]

	# Shuffle assignment of primitive commands to primitive actions
	nprim_replace = 4
	scan_primitive_tuples = [('walk','I_WALK'),('look','I_LOOK'),('run','I_RUN'),('jump','I_JUMP')]
	if nextra > 0:
		scan_primitives_extra = [(str(i),'I_'+str(i)) for i in range(1,nextra+1)]
		scan_primitive_tuples += scan_primitives_extra
	unzip = list(zip(*scan_primitive_tuples))
	list_target_command = list(unzip[0])
	list_target_output = list(unzip[1])
	if shuffle: # shuffle assignment if desired
		random.shuffle(list_target_command)
		random.shuffle(list_target_output)
	list_target_command = list_target_command[:nprim_replace]
	list_target_output = list_target_output[:nprim_replace]

	# Replace placeholders with grounded commands and actions
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim_replace+1)]
	list_source_command = [p[0] for p in scan_placeholder_tuples]
	list_source_output = [p[1] for p in scan_placeholder_tuples]
	for i in range(len(D_query)):
		D_query[i][0] = sentence_replace_var(D_query[i][0], list_source_command, list_target_command)
		D_query[i][1] = sentence_replace_var(D_query[i][1], list_source_output, list_target_output)
	for i in range(len(D_support)):
		D_support[i][0] = sentence_replace_var(D_support[i][0], list_source_command, list_target_command)
		D_support[i][1] = sentence_replace_var(D_support[i][1], list_source_output, list_target_output)

	D_primitive = list(zip(list_target_command,list_target_output))

	if inc_support_in_query:
		D_query += deepcopy(D_support)
	return D_support, D_query, D_primitive

def get_episode_generator(args):
    #  Returns function that generates episodes,
    #   and language class for the input and output language
    #
    # Input
    #  episode_type : string specifying type of episode
    #
    # Output
    #  generate_episode: function handle for generating episodes
    #  input_lang: Language object for input sequence
    #  output_lang: Language object for output sequence

    if args.episode_type == 'ME': # NeurIPS Exp 1 : Mutual exclusivity
        input_lang = Lang(['dax', 'lug', 'wif', 'zup'], args)
        output_lang = Lang(['RED', 'YELLOW', 'GREEN', 'BLUE'], args)
        generate_episode_train = lambda tabu_episodes : generate_ME(args, nquery=20,nprims=len(input_lang.symbols),input_lang=input_lang,output_lang=output_lang,tabu_list=tabu_episodes)
        generate_episode_test = generate_episode_train
    elif args.episode_type == 'scan_prim_permutation': # NeurIPS Exp 2 : Adding a new primitive through permutation meta-training
        scan_all = load_scan_file('all','train')
        scan_all_var = load_scan_var('all','train')
        input_symbols_scan = get_unique_words([c[0] for c in scan_all])
        output_symbols_scan = get_unique_words([c[1] for c in scan_all])
        input_lang = Lang(input_symbols_scan, args)
        output_lang = Lang(output_symbols_scan, args)
        generate_episode_train = lambda tabu_episodes : generate_prim_permutation(args, shuffle=True, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, nextra=0, tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes : generate_prim_permutation(args, shuffle=False, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, nextra=0, tabu_list=tabu_episodes)
    elif args.episode_type == 'scan_prim_augmentation': # NeurIPS Exp 3 : Adding a new primitive through augmentation meta-training
        nextra_prims = 20
        scan_all = load_scan_file('all','train')
        scan_all_var = load_scan_var('all','train')
        input_symbols_scan = get_unique_words([c[0] for c in scan_all]  + [str(i) for i in range(1,nextra_prims+1)])
        output_symbols_scan = get_unique_words([c[1] for c in scan_all] + ['I_' + str(i) for i in range(1,nextra_prims+1)])
        input_lang = Lang(input_symbols_scan, args)
        output_lang = Lang(output_symbols_scan, args)
        generate_episode_train = lambda tabu_episodes : generate_prim_augmentation(args, shuffle=True, nextra=nextra_prims, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes : generate_prim_augmentation(args, shuffle=False, nextra=0, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, tabu_list=tabu_episodes)
    elif args.episode_type == 'scan_around_right': # NeurIPS Exp 4 : Combining familiar concepts through meta-training
        nextra_prims = 2
        scan_all = load_scan_file('all','train')
        scan_all_var = load_scan_dir_var('all','train')
        input_symbols_scan = get_unique_words([c[0] for c in scan_all]  + [str(i) for i in range(1,nextra_prims+1)])
        output_symbols_scan = get_unique_words([c[1] for c in scan_all] + ['I_' + str(i) for i in range(1,nextra_prims+1)])
        input_lang = Lang(input_symbols_scan, args)
        output_lang = Lang(output_symbols_scan, args)
        generate_episode_train = lambda tabu_episodes : generate_right_augmentation(args, shuffle=True, nextra=nextra_prims, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes : generate_right_augmentation(args, shuffle=False, nextra=0, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, tabu_list=tabu_episodes)
    elif args.episode_type == 'scan_length': # NeurIPS Exp 5 : Generalizing to longer instructions through meta-training
        nextra_prims = 20 # number of additional primitives to augment the episodes with
        support_threshold = 12 # items with action length less than this belong in the support,
                               # and greater than or equal to this length belong in the query
        scan_length_train = load_scan_file('length','train')
        scan_length_test = load_scan_file('length','test')
        scan_all = scan_length_train+scan_length_test
        scan_length_train_var = load_scan_var('length','train')
        scan_length_test_var = load_scan_var('length','test')
        input_symbols_scan = get_unique_words([c[0] for c in scan_all]  + [str(i) for i in range(1,nextra_prims+1)])
        output_symbols_scan = get_unique_words([c[1] for c in scan_all] + ['I_' + str(i) for i in range(1,nextra_prims+1)])
        input_lang = Lang(input_symbols_scan, args)
        output_lang = Lang(output_symbols_scan, args)
        scan_length_support_var = [pair for pair in scan_length_train_var if len(pair[1].split(' ')) < support_threshold] # partition based on number of output actions
        scan_length_query_var = [pair for pair in scan_length_train_var if len(pair[1].split(' ')) >= support_threshold] # long sequences
        generate_episode_train = lambda tabu_episodes : generate_length(args, shuffle=True, nextra=nextra_prims, nsupport=100, nquery=20, input_lang=input_lang, output_lang=output_lang,
                                                            scan_tuples_support_variable=scan_length_support_var, scan_tuples_query_variable=scan_length_query_var, tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes : generate_length(args, shuffle=False, nextra=0, nsupport=100, nquery=20, input_lang=input_lang, output_lang=output_lang,
                                                            scan_tuples_support_variable=scan_length_train_var, scan_tuples_query_variable=scan_length_test_var, tabu_list=tabu_episodes)

    return generate_episode_train, generate_episode_test, input_lang, output_lang

def generate_ME(args, nquery,nprims,input_lang,output_lang,maxlen=6,tabu_list=[]):
    # Sample mutual exclusivity episode
    #
    # Input
    #  nquery : number of query examples
    #  nprims : number of unique primitives (support set includes all but one)
    #  maxlen : maximum length of a sequence in the episode
    #  ...
    #  tabu_list : identifiers of episodes we should not produce
    #

    input_symbols = deepcopy(input_lang.symbols)
    output_symbols = deepcopy(output_lang.symbols)
    assert(nprims == len(input_symbols))
    count = 0
    for _ in range(args.max_try_novel):
        random.shuffle(input_symbols)
        random.shuffle(output_symbols)
        D_str = '\n'.join([input_symbols[idx] + ' -> ' + output_symbols[idx] for idx in range(nprims)])
        identifier = make_hashable(D_str)
        if identifier not in tabu_list:
            D_support,D_query = sample_ME_concat_data(nquery=nquery,input_symbols=input_symbols,output_symbols=output_symbols,maxlen=maxlen,inc_support_in_query=not args.disable_recon_loss)
            break
    else:
        raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(args, x_support,y_support,x_query,y_query,input_lang,output_lang,identifier,args)

def generate_prim_permutation(args, shuffle,nsupport,nquery,input_lang,output_lang,scan_var_tuples,nextra,tabu_list=[]):
    # Generate a SCAN episode with primitive permutation.
    #  The tabu list identifier is based on the permutation of primitive inputs to primitive actions.
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_var_tuples : scan input/output sequences with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities
    #
    count = 0
    for _ in range(args.max_try_novel):
        D_support, D_query, D_primitive = sample_augment_scan(nsupport,nquery,scan_var_tuples,shuffle,nextra,inc_support_in_query=not args.disable_recon_loss)
        D_str = '\n'.join([s[0] + ' -> ' + s[1] for s in D_primitive])
        identifier = make_hashable(D_str)
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
    else:
        raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(args, x_support,y_support,x_query,y_query,input_lang,output_lang,identifier,args)

def generate_prim_augmentation(args, shuffle,nsupport,nquery,input_lang,output_lang,scan_var_tuples,nextra,tabu_list=[]):
    # Generate a SCAN episode with primitive augmentation,
    #  The tabu list identifier is only determined based on the assignment of the "jump" primitive
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_var_tuples : scan input/output patterns with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities
    #
    special_prim = 'jump'
    count = 0
    for _ in range(args.max_try_novel):
        D_support, D_query, D_primitive = sample_augment_scan(nsupport,nquery,scan_var_tuples,shuffle,nextra,inc_support_in_query=not args.disable_recon_loss)
        input_prim_list = [s[0] for s in D_primitive]
        try:
            index_prim = input_prim_list.index(special_prim)
            D_str = D_primitive[index_prim][0] + ' -> ' + D_primitive[index_prim][1]
        except ValueError:
            D_str = 'no jump'
        identifier = D_str
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
    else:
        raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(args, x_support,y_support,x_query,y_query,input_lang,output_lang,identifier,args)

def generate_right_augmentation(args, shuffle,nsupport,nquery,input_lang,output_lang,scan_var_tuples,nextra,tabu_list=[]):
    # Generate a SCAN episode with primitive augmentation,
    #  The tabu list is only determined based on the assignment of the "right" primitive
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_var_tuples : scan input/output patterns with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities
    #
    special_prim = 'right'
    count = 0
    for _ in range(args.max_try_novel):
        D_support, D_query, D_angles = sample_augment_direction_scan(nsupport,nquery,scan_var_tuples,shuffle,nextra,inc_support_in_query=not args.disable_recon_loss)
        input_angle_list = [s[0] for s in D_angles]
        try:
            index_prim = input_angle_list.index(special_prim)
            D_str = D_angles[index_prim][0] + ' -> ' + D_angles[index_prim][1]
        except ValueError:
            D_str = 'no right'
        identifier = D_str
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
    else:
        raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(args, x_support,y_support,x_query,y_query,input_lang,output_lang,identifier,args)

def generate_length(args, shuffle,nsupport,nquery,input_lang,output_lang,scan_tuples_support_variable,scan_tuples_query_variable,nextra,tabu_list=[]):
    # ** This episode allows different sets of input/output patterns for the support and query **
    # Generate a SCAN episode with primitive augmentation.
    #  The tabu list is based on the assignment of all of the primitive inputs to primitive actions.
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_tuples_support_variable : scan input/output patterns with placeholder replacement
    #  scan_tuples_query_variable : scan input/output patterns with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities
    #
    count = 0
    for _ in range(args.max_try_novel):
        D_support, D_query, D_primitive = sample_augment_scan_separate(nsupport,nquery,scan_tuples_support_variable,scan_tuples_query_variable,shuffle,nextra,inc_support_in_query=not args.disable_recon_loss)
        D_str = '\n'.join([s[0] + ' -> ' + s[1] for s in D_primitive])
        identifier = make_hashable(D_str)
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
    else:
        raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(args, x_support,y_support,x_query,y_query,input_lang,output_lang,identifier,args)
