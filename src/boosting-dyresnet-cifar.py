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
from keras.layers import Input
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
            + str(block) + "-" + str(repeatation) 

def gen_output_path(output_path_prefix, block, repeatation):
    return output_path_prefix + "/DyResNet-cifar" + str(block) \
            + "-" + str(repeatation)

def main():
    parser = argparser()
    args = parser.parse_args()
    mode = args.mode
    print mode
    weight_path_prefix = args.weight_path
    if "train" == mode:
        output_path_prefix = args.output_path

    batch_size = 250
    nb_epoch = 40
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
    test_sample_num = X_test.shape[0] 
    sample_weight = np.ones(train_sample_num)

    repeatations = [3, 3, 3]
    filters = [16, 32, 64]
    input_shape = [3, 32, 32]
    if len(repeatations) != len(filters):
        print "length of repeatations and length of repeatations must match"
        sys.exit()

    start = time.time()
    print "Making model"
    
    def lr_schedule(epoch):
        if 0 <= epoch and epoch < 25:
            return 0.1
        elif 25 <= epoch and epoch <= 35:
            return 0.01
        else:
            return 0.001
    reap = [1]
    blocks = 1
    alpha = np.zeros(np.asarray(repeatations).sum())
    if "test" == mode:
        total_pred = np.zeros((test_sample_num, nb_classes))
        total_acc = np.zeros(np.asarray(repeatations).sum())
        alpha = np.loadtxt("alpha")
    net_num = 0
    for r in range(len(repeatations)):
        f = [filters[r]]
        for j in range(repeatations[r]):
            net_num = net_num + 1
            train_acc = np.zeros(train_sample_num)
            sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=False)
            print "r, ", r, "j, ", j
            print "X_train.shape: ", X_train.shape
            if j == 0:
                subsamples = [True]
            else:
                subsamples = [False]
            is_first_network = False
            if r == 0 and j == 0:
                is_first_network = True
            input = Input(shape = input_shape)
            cifar_dymodel = resnet.cifar_dyresnet(blocks, subsamples, 
                    f, reap, input_shape  = input_shape, 
                    is_first_network = is_first_network, phase = "train")
            cifar_pred_dymodel = resnet.cifar_dyresnet(blocks, subsamples, 
                    f, reap, input_shape  = input_shape, 
                    is_first_network = is_first_network, phase = "test")
            # print cifar_dymodel
            plot(cifar_dymodel, 
                    to_file = "./images/dyresnet-cifar{}-{}.png".format(r, j), 
                    show_shapes = True)
            plot(cifar_pred_dymodel, 
                    to_file = "./images/dyresnet-cifar-pred{}-{}.png".format(r, j), 
                    show_shapes = True)
            input_shape = cifar_dymodel.layers[len(cifar_dymodel.layers) - 4].output_shape
            # print "input_shape: ", input_shape
            input_shape = list((input_shape[1:]))
            # print "input_shape: ", input_shape
            input_layer = cifar_dymodel.layers[0]
            # print "input_layer: ", input_layer
            # print "shape of input_layer: ", input_layer.input_shape
            cifar_dymodel.compile(loss = "categorical_crossentropy", 
                    optimizer = sgd,
                    metrics=['accuracy'])
            cifar_pred_dymodel.compile(loss = "categorical_crossentropy", 
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
                # print "weight"
                # print cifar_pred_dymodel.layers[1].get_weights()
                for i in range(len(cifar_pred_dymodel.layers)):
                    cifar_pred_dymodel.layers[i].set_weights(cifar_dymodel.layers[i].get_weights())
                error = 0
                # Update sample_weight
                start = time.time()
                for i in range(train_sample_num):
                    score = cifar_dymodel.evaluate(X_train[i : i + 1], Y_train[i : i + 1], verbose = 0)
                    train_acc[i] = score[1]
                duration = time.time() - start
                print "{} s to evaluate".format(duration)
                error_rate = (train_sample_num - train_acc.sum()) / float(train_sample_num)
                print "error_rate: ", error_rate
                alpha[net_num - 1] = 0.5 * np.log((1 - error_rate) / error_rate) 
                increse_rate = np.exp(alpha[net_num - 1])
                decrease_rate = np.exp(-1 * alpha[net_num - 1])
                for i in range(train_sample_num):
                    if train_acc[i] == 1:
                        sample_weight[i] = sample_weight[i] * decrease_rate
                    else:
                        sample_weight[i] = sample_weight[i] * increse_rate
                # sample_weight = sample_weight / sample_weight.sum()
                    
                # get and save the output of the subnetwork
                X_train = cifar_pred_dymodel.predict(X_train, batch_size = 250)
                X_test = cifar_pred_dymodel.predict(X_test, batch_size = 100)

                # print "shape of X_train: ", X_train.shape
                output_path = gen_output_path(output_path_prefix, r, j)
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                np.savez(output_path + "/output", X_train = X_train, Y_train = Y_train, 
                        X_test = X_test, Y_test = Y_test)

                # save the weight of the subnetwork
                weight_path = gen_weight_path(weight_path_prefix, r, j)
                if not os.path.isdir(weight_path):
                    os.makedirs(weight_path)
                cifar_dymodel.save_weights(weight_path + "/weight")
            elif "test" == mode:
                print "testing"
                weight_path = gen_weight_path(weight_path_prefix, r, j)
                if not os.path.isdir(weight_path):
                    print weight_path + " not exist"
                    sys.exit()
                cifar_dymodel.load_weights(weight_path + "/weight")
                for i in range(len(cifar_pred_dymodel.layers)):
                    cifar_pred_dymodel.layers[i].set_weights(cifar_dymodel.layers[i].get_weights())
                pred = cifar_dymodel.predict(X_test, batch_size = 200) 
                total_pred = total_pred + alpha[net_num - 1] * pred
                pred_labels = np.argmax(total_pred, axis = 1)
                labels = np.argmax(Y_test, axis = 1)
                total_acc[net_num - 1]  = (labels == pred_labels).sum() / float(X_test.shape[0])
                print "total_acc = ", total_acc
                X_test = cifar_pred_dymodel.predict(X_test, batch_size = 100)
            else:
                print "Illegal Running mode."
                sys.exit()
    print "alpha: ", alpha
    np.savetxt("alpha", alpha)
    sys.exit()
    
if "__main__" == __name__:
    main() 
