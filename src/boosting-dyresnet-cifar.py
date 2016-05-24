#!/usr/bin/env python
import numpy as np
np.random.seed(100)
import theano
import math
from resnet import *
import resnet
import os
import copy
# from resnet import _basic_block
import time
import sys
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import RemoteMonitor, Callback, LearningRateScheduler
import argparse


def argparser():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest = "mode")
    train_parser = subparser.add_parser("train", 
            help = "training mode")
    train_parser.add_argument("--weight-path",
            dest = "weight_path", action = "store",
            help = "Path to save weight of network for training phase",
            default = "./result/snapshot/DyResNet-cifar/")
    train_parser.add_argument("--output-path", 
            dest = "output_path", action = "store", 
            help = "Path to save output of each subnetwork",
            default = "./data/DyResNet-cifar/")
    test_parser = subparser.add_parser("test",
            help = "testing mode")
    test_parser.add_argument("--weight-path",
            dest = "weight_path", action = "store",
            help = "Path to load weight of network for testing phase",
            default = "./result/snapshot/DyResNet-cifar/")

    return parser

def gen_weight_path(weight_path_prefix, block, repeatation):
    return weight_path_prefix + "/DyResNet-cifar-" \
            + str(block) + "-" + str(repeatation) + "/weight"

def gen_output_path(output_path_prefix, block, repeatation):
    return output_path_prefix + "/DyResNet-cifar" + str(block) \
            + "-" + str(repeatation) + "/output"

def main():
    parser = argparser()
    args = parser.parse_args()
    mode = args.mode
    print mode
    weight_path_prefix = args.weight_path
    if "train" == "mode":
        output_path_prefix = args.output_path

    batch_size = 250
    nb_epoch = 1
    nb_classes = 10
    base_lr = 0.1

    start = time.time()
    print "Loading data"
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    duration = time.time() - start
    print "{} s to load data".format(duration)
    
    train_sample_num = X_train.shape[0] 
    sample_weight = np.ones(train_sample_num) / train_sample_num

    repeatations = [3, 3, 3]
    filters = [16, 32, 64]
    input_shape = [3, 32, 32]
    if len(repeatations) != len(filters):
        print "length of repeatations and length of repeatations must match"
        sys.exit()

    start = time.time()
    print "Making model"
    
    def lr_schedule(epoch):
        if 0 <= epoch and epoch < 160:
            return 0.1
        elif 160 <= epoch and epoch <= 240:
            return 0.01
        else:
            return 0.001
    r = [1]
    blocks = 1
    alpha = np.zeros(np.asarray(repeatations).sum())
    net_num = 0
    for i in range(len(repeatations)):
        f = [filters[i]]
        for j in range(repeatations[i]):
            net_num = net_num + 1
            sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=False)
            print "i, ", i, "j, ", j
            if j == 0:
                subsamples = [True]
            else:
                subsamples = [False]
            print "input_shape: ", input_shape
            is_first_network = False
            if i == 0 and j == 0:
                is_first_network = True
            cifar_dymodel = resnet.cifar_dyresnet(blocks, subsamples, 
                    f, r, input_shape = input_shape, 
                    is_first_network = is_first_network)
            print cifar_dymodel
            plot(cifar_dymodel, 
                    to_file = "./images/dyresnet-cifar{}-{}.png".format(i, j), 
                    show_shapes = True)
            input_shape = cifar_dymodel.layers[len(cifar_dymodel.layers) - 3].output_shape
            print "input_shape: ", input_shape
            input_shape = list((input_shape[1:]))
            print "input_shape: ", input_shape
            get_feature = theano.function([cifar_dymodel.layers[0].input],
                    cifar_dymodel.layers[len(cifar_dymodel.layers) - 3].output)
            sys.exit()
            cifar_dymodel.compile(loss = "categorical_crossentropy", 
                    optimizer = sgd,
                    metrics=['accuracy'])
            
            if "train" == mode:
                history = cifar_dymodel.fit(X_train, Y_train,
                          batch_size=batch_size,
                          sample_weight = sample_weight, 
                          nb_epoch=nb_epoch,
                          callbacks = [LearningRateScheduler(lr_schedule)],
                          validation_data=(X_test, Y_test),
                          shuffle=True)
               
                # b = np.asarray(cifar_dymodel.layers[len(cifar_dymodel.layers) - 3].output)
                # print "type of b: ", type(b)
                # print "shape of b: ", b.shape
                # sys.exit()
                # f = theano.function([a], b)
                sys.exit()
                error = 0
                for i in range(train_sample_num):
                    score = cifar_dymodel.evaluate(X_train[i : i + 1], Y_train[i : i + 1])
                    print score
                    sys.exit()
                    
                # get and save the output of the subnetwork
                X_train = get_feature(X_train)
                X_test = get_feature(X_test)
                print "shape of X_train: ", X_train.shape
                output_path = gen_output_path(output_path_prefix, i, j)
                if os.isdir(output_path):
                    os.makedirs(output_path)
                cifar_dymodel.save_weights(weight_path)

                # save the weight of the subnetwork
                weight_path = gen_weight_path(weight_path_prefix, i, j)
                if os.isdir(weight_path):
                    os.makedirs(weight_path)
                np.savez(X_train = X_train, Y_train = Y_train, 
                        X_test = X_test, Y_test = Y_test)
            elif "test" == mode:
                weight_path = gen_weight_path(weight_path_prefix, i, j)
                if os.isdir(weight_path):
                    print weight_path + " not exist"
                    sys.exit()
                cifar_dymodel.load_weights(weight_path)
                score = cifar_model.evaluate(X_test, Y_test, batch_size = 200) 
                X_test = get_feature(X_test)
                print "Testing score = ", score
            else:
                print "Illegal Running mode."
                sys.exit()
            sys.exit()

    sys.exit()
    
if "__main__" == __name__:
    main() 
