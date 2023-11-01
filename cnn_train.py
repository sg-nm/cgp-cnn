#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
from chainer import computational_graph
import six
import time
import numpy as np
from chainer import serializers

from cnn_model import CGP2CNN


# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self, dataset_name, validation=True, valid_data_ratio=0.1, verbose=True):
        # dataset_name: name of data set ('cifar10' or 'cifar100' or 'mnist')
        # validation: [True]  model validation mode
        #                     (split training data set according to valid_data_ratio for evaluation of CGP individual)
        #             [False] model test mode for final evaluation of the evolved model
        #                     (raining data : all training data, test data : all test data)
        # valid_data_ratio: ratio of the validation data
        #                    (e.g., if the number of all training data=50000 and valid_data_ratio=0.2, 
        #                       the number of training data=40000, validation=10000)
        # verbose: flag of display
        self.verbose = verbose

        # load dataset
        if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'mnist':
            if dataset_name == 'cifar10':
                self.n_class = 10
                self.channel = 3
                self.pad_size = 4
                train, test = chainer.datasets.get_cifar10(withlabel=True, ndim=3, scale=1.0)
            elif dataset_name == 'cifar100':
                self.n_class = 100
                self.channel = 3
                self.pad_size = 4
                train, test = chainer.datasets.get_cifar100(withlabel=True, ndim=3, scale=1.0)
            else:    # mnist
                self.n_class = 10
                self.channel = 1
                self.pad_size = 4
                train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3, scale=1.0)

            # model validation mode
            if validation:
                # split into train and validation data
                np.random.seed(2016)  # always same data splitting
                order = np.random.permutation(len(train))
                np.random.seed()
                if self.verbose:
                    print('\tdata split order: ', order)
                train_size = int(len(train) * (1. - valid_data_ratio))
                # train data
                self.x_train, self.y_train = train[order[:train_size]][0], train[order[:train_size]][1]
                # test data (for validation)
                self.x_test, self.y_test = train[order[train_size:]][0], train[order[train_size:]][1]
            # model test mode
            else:
                # train data
                self.x_train, self.y_train = train[range(len(train))][0], train[range(len(train))][1]
                # test data
                self.x_test, self.y_test = test[range(len(test))][0], test[range(len(test))][1]
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

        # preprocessing (subtraction of mean pixel values)
        x_mean = 0
        for x in self.x_train:
            x_mean += x
        x_mean /= len(self.x_train)
        self.x_train -= x_mean
        self.x_test -= x_mean

        # data size
        self.train_data_num = len(self.x_train)
        self.test_data_num = len(self.x_test)
        if self.verbose:
            print('\ttrain data shape:', self.x_train.shape)
            print('\ttest data shape :', self.x_test.shape)

    def __call__(self, cgp, gpuID, epoch_num=200, batchsize=256, weight_decay=1e-4, eval_epoch_num=10,
                 data_aug=True, comp_graph='comp_graph.dot', out_model='mymodel.model', init_model=None,
                 retrain_mode=False):
        if self.verbose:
            print('\tGPUID    :', gpuID)
            print('\tepoch_num:', epoch_num)
            print('\tbatchsize:', batchsize)

        chainer.cuda.get_device(gpuID).use()  # Make a specified GPU current
        model = CGP2CNN(cgp, self.n_class)
        if init_model is not None:
            if self.verbose:
                print('\tLoad model from', init_model)
            serializers.load_npz(init_model, model)
        model.to_gpu(gpuID)
        optimizer = chainer.optimizers.Adam() if not retrain_mode else chainer.optimizers.MomentumSGD(lr=0.01)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        eval_epoch_num = np.min((eval_epoch_num, epoch_num))
        test_accuracies = np.zeros(eval_epoch_num)
        for epoch in six.moves.range(1, epoch_num+1):
            if self.verbose:
                print('\tepoch', epoch)
            perm = np.random.permutation(self.train_data_num)
            train_accuracy = train_loss = 0
            start = time.time()
            for i in six.moves.range(0, self.train_data_num, batchsize):
                xx_train = self.data_augmentation(self.x_train[perm[i:i + batchsize]]) if data_aug else self.x_train[perm[i:i + batchsize]]
                x = chainer.Variable(cuda.to_gpu(xx_train))
                t = chainer.Variable(cuda.to_gpu(self.y_train[perm[i:i + batchsize]]))
                try:
                    optimizer.update(model, x, t)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0.

                if comp_graph is not None and epoch == 1 and i == 0:
                    with open(comp_graph, 'w') as o:
                        g = computational_graph.build_computational_graph((model.loss, ))
                        o.write(g.dump())
                        del g
                        if self.verbose:
                            print('\tCNN graph generated.')

                train_loss += float(model.loss.data) * len(t.data)
                train_accuracy += float(model.accuracy.data) * len(t.data)
            elapsed_time = time.time() - start
            throughput = self.train_data_num / elapsed_time
            if self.verbose:
                print('\ttrain mean loss={}, train accuracy={}, time={}, throughput={} images/sec, paramNum={}'.format(train_loss / self.train_data_num, train_accuracy / self.train_data_num, elapsed_time, throughput, model.param_num))

            # apply the model to test data
            # use the maximum validation accuracy in the last 10 epoch as the fitness value
            eval_index = epoch - (epoch_num - eval_epoch_num) -1
            if self.verbose or eval_index >= 0:
                test_accuracy, test_loss = self.__test(model, batchsize)
                if self.verbose:
                    print('\tvalid mean loss={}, valid accuracy={}'.format(test_loss / self.test_data_num, test_accuracy / self.test_data_num))
                if eval_index >= 0:
                    test_accuracies[eval_index] = test_accuracy / self.test_data_num

            # decay the learning rate
            if not retrain_mode and epoch % 30 == 0:
                optimizer.alpha *= 0.1
            elif retrain_mode:
                if epoch == 5:
                    optimizer.lr = 0.1
                if epoch == 250:
                    optimizer.lr *= 0.1
                if epoch == 375:
                    optimizer.lr *= 0.1

        # test_accuracy, test_loss = self.__test(model, batchsize)
        if out_model is not None:
            model.to_cpu()
            serializers.save_npz(out_model, model)

        return np.max(test_accuracies)

    def test(self, cgp, model_file, comp_graph='comp_graph.dot', batchsize=256):
        chainer.cuda.get_device(0).use()  # Make a specified GPU current
        model = CGP2CNN(cgp, self.n_class)
        print('\tLoad model from', model_file)
        serializers.load_npz(model_file, model)
        model.to_gpu(0)
        test_accuracy, test_loss = self.__test(model, batchsize)
        print('\tparamNum={}'.format(model.param_num))
        print('\ttest mean loss={}, test accuracy={}'.format(test_loss / self.test_data_num, test_accuracy / self.test_data_num))

        if comp_graph is not None:
            with open(comp_graph, 'w') as o:
                g = computational_graph.build_computational_graph((model.loss,))
                o.write(g.dump())
                del g
                print('\tCNN graph generated ({}).'.format(comp_graph))

        return test_accuracy, test_loss

    def __test(self, model, batchsize):
        model.train = False
        test_accuracy = test_loss = 0
        for i in six.moves.range(0, self.test_data_num, batchsize):
            with chainer.using_config('train', False):
                # x = chainer.Variable(cuda.to_gpu(self.x_test[i:i + batchsize]))
                x = chainer.Variable(cuda.to_gpu(self.x_test[i:i + batchsize]))
                t = chainer.Variable(cuda.to_gpu(self.y_test[i:i + batchsize]))
            loss = model(x, t)
            test_loss += float(loss.data) * len(t.data)
            test_accuracy += float(model.accuracy.data) * len(t.data)
        model.train = True
        return test_accuracy, test_loss

    def data_augmentation(self, x_train):
        _, c, h, w = x_train.shape
        pad_h = h + 2 * self.pad_size
        pad_w = w + 2 * self.pad_size
        aug_data = np.zeros_like(x_train)
        for i, x in enumerate(x_train):
            pad_img = np.zeros((c, pad_h, pad_w))
            pad_img[:, self.pad_size:h+self.pad_size, self.pad_size:w+self.pad_size] = x

            # Randomly crop and horizontal flip the image
            top = np.random.randint(0, pad_h - h + 1)
            left = np.random.randint(0, pad_w - w + 1)
            bottom = top + h
            right = left + w
            if np.random.randint(0, 2):
                pad_img = pad_img[:, :, ::-1]

            aug_data[i] = pad_img[:, top:bottom, left:right]

        return aug_data
