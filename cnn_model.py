#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy
from chainer import reporter


# CONV -> Batch -> ReLU
class ConvBlock(chainer.Chain):
    def __init__(self, ksize, n_out, initializer):
        super(ConvBlock, self).__init__()
        pad_size = ksize // 2
        links = [('conv1', L.Convolution2D(None, n_out, ksize, pad=pad_size, initialW=initializer))]
        links += [('bn1', L.BatchNormalization(n_out))]
        for link in links:
            self.add_link(*link)
        self.forward = links
    
    def __call__(self, x, train):
        param_num = 0
        for name, f in self.forward:
            if 'conv1' in name:
                x = getattr(self, name)(x)
                param_num += (f.W.shape[0]*f.W.shape[2]*f.W.shape[3]*f.W.shape[1]+f.W.shape[0])
            elif 'bn1' in name:
                x = getattr(self, name)(x, not train)
                param_num += x.data.shape[1]*2
        return (F.relu(x), param_num)


# [(CONV -> Batch -> ReLU -> CONV -> Batch) + (x)]
class ResBlock(chainer.Chain):
    def __init__(self, ksize, n_out, initializer):
        super(ResBlock, self).__init__()
        pad_size = ksize // 2
        links  = [('conv1', L.Convolution2D(None, n_out, ksize, pad=pad_size, initialW=initializer))]
        links += [('bn1', L.BatchNormalization(n_out))]
        links += [('_act1', F.relu)]
        links += [('conv2', L.Convolution2D(n_out, n_out, ksize, pad=pad_size, initialW=initializer))]
        links += [('bn2', L.BatchNormalization(n_out))]
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
    
    def __call__(self, x, h, train):
        xp = chainer.cuda.get_array_module(x)
        param_num = 0
        for name, f in self.forward:
            if 'conv' in name:
                x = getattr(self, name)(x)
                param_num += (f.W.shape[0]*f.W.shape[2]*f.W.shape[3]*f.W.shape[1]+f.W.shape[0])
            elif 'bn' in name:
                x = getattr(self, name)(x, not train)
                param_num += x.data.shape[1]*2
            elif 'act' in name:
                x = f(x)
            else:
                print('not defined function at ResBlock __call__')
                exit(1)
        in_data = [x, h]
        # check of the image size
        small_in_id, large_in_id = (0, 1) if in_data[0].shape[2] < in_data[1].shape[2] else (1, 0)
        pool_num = xp.floor(xp.log2(in_data[large_in_id].shape[2] / in_data[small_in_id].shape[2]))
        for _ in xp.arange(pool_num):
            in_data[large_in_id] = F.max_pooling_2d(in_data[large_in_id], self.pool_size, self.pool_size, 0, False)
        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].shape[1] < in_data[1].shape[1] else (1, 0)
        pad_num = int(in_data[large_ch_id].shape[1] - in_data[small_ch_id].shape[1])
        tmp = in_data[large_ch_id][:, :pad_num, :, :]
        in_data[small_ch_id] = F.concat((in_data[small_ch_id], tmp * 0), axis=1)
        return (F.relu(in_data[0]+in_data[1]), param_num)


# Construct a CNN model using CGP (list)
class CGP2CNN(chainer.Chain):
    def __init__(self, cgp, n_class, lossfun=softmax_cross_entropy.softmax_cross_entropy, accfun=accuracy.accuracy):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        initializer = chainer.initializers.HeNormal()
        links = []
        i = 1
        for name, in1, in2 in self.cgp:
            if name == 'pool_max':
                links += [('_'+name+'_'+str(i), F.MaxPooling2D(self.pool_size, self.pool_size, 0, False))]
            elif name == 'pool_ave':
                links += [('_'+name+'_'+str(i), F.average_pooling_2d(self.pool_size, self.pool_size, 0, False))]
            elif name == 'concat':
                links += [('_'+name+'_'+str(i), F.concat())]
            elif name == 'sum':
                links += [('_'+name+'_'+str(i), F.concat())] # the F.Concat() is dummy
            elif name == 'ConvBlock32_3':
                links += [(name+'_'+str(i), ConvBlock(3, 32, initializer))]
            elif name == 'ConvBlock32_5':
                links += [(name+'_'+str(i), ConvBlock(5, 32, initializer))]
            elif name == 'ConvBlock32_7':
                links += [(name+'_'+str(i), ConvBlock(7, 32, initializer))]
            elif name == 'ConvBlock64_3':
                links += [(name+'_'+str(i), ConvBlock(3, 64, initializer))]
            elif name == 'ConvBlock64_5':
                links += [(name+'_'+str(i), ConvBlock(5, 64, initializer))]
            elif name == 'ConvBlock64_7':
                links += [(name+'_'+str(i), ConvBlock(7, 64, initializer))]
            elif name == 'ConvBlock128_3':
                links += [(name+'_'+str(i), ConvBlock(3, 128, initializer))]
            elif name == 'ConvBlock128_5':
                links += [(name+'_'+str(i), ConvBlock(5, 128, initializer))]
            elif name == 'ConvBlock128_7':
                links += [(name+'_'+str(i), ConvBlock(7, 128, initializer))]
            elif name == 'ResBlock32_3':
                links += [(name+'_'+str(i), ResBlock(3, 32, initializer))]
            elif name == 'ResBlock32_5':
                links += [(name+'_'+str(i), ResBlock(5, 32, initializer))]
            elif name == 'ResBlock32_7':
                links += [(name+'_'+str(i), ResBlock(7, 32, initializer))]
            elif name == 'ResBlock64_3':
                links += [(name+'_'+str(i), ResBlock(3, 64, initializer))]
            elif name == 'ResBlock64_5':
                links += [(name+'_'+str(i), ResBlock(5, 64, initializer))]
            elif name == 'ResBlock64_7':
                links += [(name+'_'+str(i), ResBlock(7, 64, initializer))]
            elif name == 'ResBlock128_3':
                links += [(name+'_'+str(i), ResBlock(3, 128, initializer))]
            elif name == 'ResBlock128_5':
                links += [(name+'_'+str(i), ResBlock(5, 128, initializer))]
            elif name == 'ResBlock128_7':
                links += [(name+'_'+str(i), ResBlock(7, 128, initializer))]
            elif name == 'full':
                links += [(name+'_'+str(i), L.Linear(None, n_class, initialW=initializer))]
            i += 1
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
        self.train = True
        self.lossfun = lossfun
        self.accfun = accfun
        self.loss = None
        self.accuracy = None
        self.outputs = [None for _ in range(len(self.cgp))]
        self.param_num = 0

    def __call__(self, x, t):
        xp = chainer.cuda.get_array_module(x)
        outputs = self.outputs
        outputs[0] = x    # input image
        nodeID = 1
        param_num = 0
        for name, f in self.forward:
            if 'ConvBlock' in name:
                outputs[nodeID], tmp_num = getattr(self, name)(outputs[self.cgp[nodeID][1]], self.train)
                nodeID += 1
                param_num += tmp_num
            elif 'ResBlock' in name:
                outputs[nodeID], tmp_num = getattr(self, name)(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][1]], self.train)
                nodeID += 1
                param_num += tmp_num
            elif 'pool' in name:
                # check of the image size
                if outputs[self.cgp[nodeID][1]].shape[2] > 1:
                    outputs[nodeID] = f(outputs[self.cgp[nodeID][1]])
                    nodeID += 1
                else:
                    outputs[nodeID] = outputs[self.cgp[nodeID][1]]
                    nodeID += 1
            elif 'concat' in name:
                in_data = [outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]]]
                # check of the image size
                small_in_id, large_in_id = (0, 1) if in_data[0].shape[2] < in_data[1].shape[2] else (1, 0)
                pool_num = xp.floor(xp.log2(in_data[large_in_id].shape[2] / in_data[small_in_id].shape[2]))
                for _ in xp.arange(pool_num):
                    in_data[large_in_id] = F.max_pooling_2d(in_data[large_in_id], self.pool_size, self.pool_size, 0, False)
                # concat
                outputs[nodeID] = f(in_data[0], in_data[1])
                nodeID += 1
            elif 'sum' in name:
                in_data = [outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]]]
                # check of the image size
                small_in_id, large_in_id = (0, 1) if in_data[0].shape[2] < in_data[1].shape[2] else (1, 0)
                pool_num = xp.floor(xp.log2(in_data[large_in_id].shape[2] / in_data[small_in_id].shape[2]))
                for _ in xp.arange(pool_num):
                    in_data[large_in_id] = F.max_pooling_2d(in_data[large_in_id], self.pool_size, self.pool_size, 0, False)
                # check of the channel size
                small_ch_id, large_ch_id = (0, 1) if in_data[0].shape[1] < in_data[1].shape[1] else (1, 0)
                pad_num = int(in_data[large_ch_id].shape[1] - in_data[small_ch_id].shape[1])
                tmp = in_data[large_ch_id][:, :pad_num, :, :]
                in_data[small_ch_id] = F.concat((in_data[small_ch_id], tmp * 0), axis=1)
                # summation
                outputs[nodeID] = in_data[0] + in_data[1]
                nodeID += 1
            elif 'full' in name:
                outputs[nodeID] = getattr(self, name)(outputs[self.cgp[nodeID][1]])
                nodeID += 1
                param_num += f.W.data.shape[0] * f.W.data.shape[1] + f.b.data.shape[0]
            else:
                print('not defined function at CGP2CNN __call__')
                exit(1)
        self.param_num = param_num

        if t is not None:
            self.loss = None
            self.accuracy = None
            self.loss = self.lossfun(outputs[-1], t)
            reporter.report({'loss': self.loss}, self)
            self.accuracy = self.accfun(outputs[-1], t)
            reporter.report({'accuracy': self.accuracy}, self)
            return self.loss
        else:
            return outputs[-1]
