#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
import gep_cnn_train as cnn

def gep_arg_wrapper_mp(args):
    return args[0](*args[1:])

def gep_cnn_eval(net, gpu_id, epoch_num, batchsize, dataset, valid_data_ratio, verbose):

    print('\tgpu_id:', gpu_id, ',', net)
    train = cnn.GEP_CNN_train(dataset, validation=True, valid_data_ratio=valid_data_ratio, verbose=verbose)
    evaluation = train(net, gpu_id, epoch_num=epoch_num, batchsize=batchsize,
                               comp_graph='CNN%d.dot'%(gpu_id), out_model=None, init_model=None)
    print('\tgpu_id:', gpu_id, ', eval:', evaluation)
    return evaluation

class GEPCNNEvaluation(object):
    def __init__(self, gpu_num, epoch_num=50, batchsize=256, dataset='cifar10', valid_data_ratio=0.1, verbose=True):
        self.gpu_num = gpu_num
        self.epoch_num = epoch_num
        self.batchsize = batchsize
        self.dataset = dataset
        self.valid_data_ratio = valid_data_ratio
        self.verbose = verbose

    def __call__(self, chromosome_list):
        evaluations = np.zeros(len(chromosome_list))

        for i in np.arange(0, len(chromosome_list), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(chromosome_list))) - i

            pool = mp.Pool(process_num)
            arg_data = [(gep_cnn_eval, chromosome_list[i+j], j, self.epoch_num, self.batchsize, self.dataset,
                         self.valid_data_ratio, self.verbose) for j in range(process_num)]
            evaluations[i:i+process_num] = pool.map(gep_arg_wrapper_mp, arg_data)
            pool.terminate()

        return evaluations

class GepInfoConvSet(object):
    def __init__(self, head_length=10, num_genes=5, mutation_rate=0.1, crossover_rate=0.3):
        self.head_length = head_length
        self.num_genes = num_genes
        max_arity = 2
        tail_length = head_length * (max_arity - 1) + 1
        self.gene_length = head_length + tail_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.func_type = ['ConvBlock32_3', 'ConvBlock32_5',
                          'ConvBlock64_3', 'ConvBlock64_5',
                          'ConvBlock128_3', 'ConvBlock128_5',
                          'pool_max', 'pool_ave',
                          'concat', 'sum']
        self.func_in_num = [1, 1,
                            1, 1,
                            1, 1,
                            1, 1,
                            2, 2]
        
        self.terminal_set = ['input']
        self.min_active_genes = num_genes // 2
        self.max_active_genes = num_genes

        self.input_num = 1
        self.out_num = 1
        self.out_type = ['full']


class GepInfoResSet(object):
    def __init__(self, head_length=10, num_genes=5, mutation_rate=0.1, crossover_rate=0.3):
        self.head_length = head_length
        self.num_genes = num_genes
        max_arity = 2
        tail_length = head_length * (max_arity - 1) + 1
        self.gene_length = head_length + tail_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.func_type = ['ResBlock32_3', 'ResBlock32_5',
                          'ResBlock64_3', 'ResBlock64_5',
                          'ResBlock128_3', 'ResBlock128_5',
                          'pool_max', 'pool_ave',
                          'concat', 'sum']
        self.func_in_num = [1, 1,
                            1, 1,
                            1, 1,
                            1, 1,
                            2, 2]

        self.terminal_set = ['input']
        self.min_active_genes = num_genes // 2
        self.max_active_genes = num_genes

        self.input_num = 1
        self.out_num = 1
        self.out_type = ['full']

