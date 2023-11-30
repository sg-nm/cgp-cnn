#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import numpy as np
from gep_cnn_model import *

class GEPIndividual(object):
    def __init__(self, net_info):
        self.net_info = net_info
        self.chromosome = self.init_chromosome()
        self.eval = None

    def init_chromosome(self):
        chromosome = ''
        for _ in range(self.net_info.num_genes):
            gene = ''.join(np.random.choice(self.net_info.function_set + self.net_info.terminal_set, self.net_info.gene_length))
            chromosome += gene
        return chromosome

    def mutate(self, mutation_rate):
        mutated = False
        new_chromosome = list(self.chromosome)
        for i in range(len(new_chromosome)):
            if np.random.rand() < mutation_rate:
                if i % self.net_info.gene_length < self.net_info.head_length:
                    new_chromosome[i] = np.random.choice(self.net_info.function_set + self.net_info.terminal_set)
                else:
                    new_chromosome[i] = np.random.choice(self.net_info.terminal_set)
                mutated = True
        self.chromosome = ''.join(new_chromosome)
        return mutated

    def forced_mutate(self, mutation_rate):
        while not self.mutate(mutation_rate):
            pass
        return True

    def neutral_mutation(self):
        # Neutral mutation logic
        pass

    def copy(self, source):
        self.net_info = source.net_info
        self.chromosome = source.chromosome
        self.eval = source.eval

    def translate_to_cnn(self):
        layers = []
        for i in range(0, len(self.chromosome), self.net_info.gene_length):
            gene = self.chromosome[i:i + self.net_info.gene_length]
            layer_type = gene[0]

            if layer_type == 'C':
                # Convolutional layer
                filters, ksize = int(gene[1:3]), int(gene[4])
                layers.append(ConvBlock(ksize, filters, chainer.initializers.HeNormal()))
            elif layer_type == 'R':
                # Residual block
                filters, ksize = int(gene[1:3]), int(gene[4])
                layers.append(ResBlock(ksize, filters, chainer.initializers.HeNormal()))
            elif layer_type == 'P':
                # Max pooling layer
                pool_size = int(gene[1])  # Assuming gene[1] encodes the pool size
                layers.append(MaxPoolingLayer(pool_size))
            elif layer_type == 'A':
                # Average pooling layer
                pool_size = int(gene[1])  # Assuming gene[1] encodes the pool size
                layers.append(AveragePoolingLayer(pool_size))
            elif layer_type == 'S':
                # Summation layer (assuming it sums two previous layers)
                layers.append(SumLayer())
            elif layer_type == 'F':
                # Fully connected layer
                num_units = int(gene[1:4])  # Assuming gene[1:4] encodes the number of units
                layers.append(L.Linear(None, num_units))

        return layers

class GEP(object):
    def __init__(self, net_info, eval_func, lam=4):
        self.lam = lam
        self.pop = [GEPIndividual(net_info) for _ in range(1 + self.lam)]
        self.eval_func = eval_func

        self.num_gen = 0
        self.num_eval = 0
    
    def _evaluation(self, pop, eval_flag):
        # create network list
        evaluations = np.zeros(len(pop))
        for i, ind in enumerate(pop):
            cnn_architecture = ind.translate_to_cnn()
            evaluations[i] = self.eval_func(cnn_architecture)
            ind.eval = evaluations[i]
        self.num_eval += len(pop)
        return evaluations
    
    def _log_data(self):
        log_list = [self.num_gen, self.num_eval, time.perf_counter(), self.pop[0].eval]
        log_list.append(self.pop[0].chromosome)
        return log_list
    
    def load_log(self, log_data):
        self.num_gen, self.num_eval, _, best_eval, best_chromosome = log_data
        self.pop[0].eval = best_eval
        self.pop[0].chromosome = best_chromosome
    
    def evolution(self, max_eval=100, mutation_rate=0.01, log_file='./log.txt'):
        with open(log_file, 'w', newline='') as fw:
            writer = csv.writer(fw, lineterminator='\n')

            self._evaluation([self.pop[0]])
            print(self._log_data())

            while self.num_eval < max_eval:
                self.num_gen += 1

                for i in range(self.lam):
                    self.pop[i+1].copy(self.pop[0])
                    self.pop[i+1].mutate(mutation_rate)

                evaluations = self._evaluation(self.pop[1:])
                best_arg = np.argmax(evaluations)
                if evaluations[best_arg] >= self.pop[0].eval:
                    self.pop[0].copy(self.pop[best_arg+1])

                log_data = self._log_data()
                print(log_data)
                writer.writerow(log_data)

    def modified_evolution(self, max_eval=100, mutation_rate=0.01, log_file='./log.txt'):
        with open(log_file, 'w', newline='') as fw:
            writer = csv.writer(fw, lineterminator='\n')

            self._evaluation([self.pop[0]])
            print(self._log_data())

            while self.num_eval < max_eval:
                self.num_gen += 1

                for i in range(self.lam):
                    self.pop[i + 1].copy(self.pop[0])
                    while not self.pop[i + 1].forced_mutate(mutation_rate):
                        pass

                evaluations = self._evaluation(self.pop[1:])
                best_arg = np.argmax(evaluations)
                if evaluations[best_arg] >= self.pop[0].eval:
                    self.pop[0].copy(self.pop[best_arg + 1])
                else:
                    self.pop[0].neutral_mutation()

                log_data = self._log_data()
                print(log_data)
                writer.writerow(log_data)