#!/usr/bin/env python
import numpy as np
np.random.seed(100)
import theano
import subprocess
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
from keras.preprocessing.image import ImageDataGenerator
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
        loss_path_prefix = "./result/loss/DyResNet-cifar/"
        loss_path = loss_path_prefix + "DyResNet-cifar-loss"
        val_loss_path = loss_path_prefix + "DyResNet-cifar-val-loss"
        acc_path_prefix = "./result/accuracy/DyResNet-cifar/"
        acc_path = acc_path_prefix + "DyResNet-cifar-acc"
        val_acc_path = acc_path_prefix + "DyResNet-cifar-val-acc"
        total_loss = []
        total_acc = []
        total_val_loss = []
        total_val_acc = []

    train_batch_size = 250
    test_batch_size = 100
    nb_epoch = 40
    nb_classes = 10
    base_lr = 0.1

    start = time.time()
    print "Loading data"
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
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
            reap = [2]
            blocks = 1
            is_first_network = False
            if j == 0:
                if r == 0:
                    reap = [1]
                    blocks = 1
                    f = [filters[r]]
                    subsamples = [False]
                    test_reap = []
                    test_blocks = 0
                    test_f = []
                    test_subsamples = []
                    is_first_network = True
                else:
                    reap = [1, 1]
                    blocks = 2
                    f = [filters[r - 1], filters[r]]
                    subsamples = [False, True]
                    test_reap = [1]
                    test_blocks = 1
                    test_f = [filters[r - 1]]
                    test_subsamples = [False]

            elif j == 1:
                if r != 0:
                    reap = [2]
                    blocks = 1
                    f = [filters[r]]
                    subsamples = [True]
                    test_reap = [1]
                    test_blocks = 1
                    test_f = [filters[r]]
                    test_subsamples = [True]
                else:
                    reap = [2]
                    blocks = 1
                    f = [filters[r]]
                    subsamples = [False]
                    test_reap = [1]
                    test_blocks = 1
                    test_f = [filters[r]]
                    test_subsamples = [False]
            else:
                reap = [2]
                blocks = 1
                f = [filters[r]]
                subsamples = [False]
                test_reap = [1]
                test_blocks = 1
                test_f = [filters[r]]
                test_subsamples = [False]
            # if j == 0:
            #     subsamples = [True]
            # else:
            #     subsamples = [False]
            # is_first_network = False
            # if r == 0 and j == 0:
            #     is_first_network = True
            input = Input(shape = input_shape)
            cifar_dymodel = resnet.cifar_dyresnet(blocks, subsamples, 
                    f, reap, input_shape  = input_shape, 
                    is_first_network = is_first_network, phase = "train")
            cifar_pred_dymodel = resnet.cifar_dyresnet(test_blocks, test_subsamples, 
                    test_f, test_reap, input_shape  = input_shape, 
                    is_first_network = is_first_network, phase = "test")
            # print cifar_dymodel
            plot(cifar_dymodel, 
                    to_file = "./images/dyresnet-cifar{}-{}.png".format(r, j), 
                    show_shapes = True)
            plot(cifar_pred_dymodel, 
                    to_file = "./images/dyresnet-cifar-pred{}-{}.png".format(r, j), 
                    show_shapes = True)
            output_layer = len(cifar_dymodel.layers) - 11
            if j == 0 and r != 0:
                output_layer = len(cifar_dymodel.layers) - 12
            input_shape = cifar_dymodel.layers[output_layer].output_shape
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
                # copy weight from previous subnetwork
                # if r != 0 or j != 0:
                #     for i in range(len(weights)):
                #         print "i: ", i
                #         print cifar_dymodel.layers[i].get_config()
                #         cifar_dymodel.layers[i].set_weights(weights[i])

                if j == 0 and r == 0:
                    datagen = ImageDataGenerator(featurewise_center = True,
                            width_shift_range = 0.125, 
                            height_shift_range = 0.125, 
                            horizontal_flip = True)
                    test_datagen = ImageDataGenerator(featurewise_center = True,
                            width_shift_range = 0.125, 
                        height_shift_range = 0.125, 
                        horizontal_flip = True)

                    datagen.fit(X_train)
                    test_datagen.fit(X_test)

                    history = cifar_dymodel.fit_generator(datagen.flow(X_train, Y_train, batch_size = train_batch_size),
                            samples_per_epoch = train_sample_num, 
                            nb_epoch = nb_epoch,
                            callbacks = [LearningRateScheduler(lr_schedule)],
                            validation_data=test_datagen.flow(X_test, Y_test, batch_size = test_batch_size),
                            nb_val_samples = test_sample_num)
                else:
                    history = cifar_dymodel.fit(X_train, Y_train,
                              batch_size=train_batch_size,
                              sample_weight = sample_weight, 
                              nb_epoch=nb_epoch,
                              callbacks = [LearningRateScheduler(lr_schedule)],
                              validation_data=(X_test, Y_test),
                              shuffle=True)
                loss = history.history["loss"]
                val_loss = history.history["val_loss"]
                acc = history.history["acc"]
                val_acc = history.history["val_acc"]
                total_loss = np.hstack((total_loss, loss))
                total_acc = np.hstack((total_acc, acc))
                total_val_loss = np.hstack((total_val_loss, val_loss))
                total_val_acc = np.hstack((total_val_acc, val_acc))

                np.savetxt("alpha", alpha)
                np.savetxt(loss_path, total_loss)
                np.savetxt(val_loss_path, total_val_loss)
                np.savetxt(acc_path, total_acc)
                np.savetxt(val_acc_path, total_val_acc)
                
                # save weight for the next subnetwork
                print "saving weight"
                weights = []
                for i in range(output_layer, len(cifar_dymodel.layers) - 3):
                    print "i: ", i
                    print cifar_dymodel.layers[i].get_config()
                    weights.append(cifar_dymodel.layers[i].get_weights())
                print "length of weights: ", len(weights)
                # print "weight"
                # print cifar_pred_dymodel.layers[1].get_weights()
                for i in range(len(cifar_pred_dymodel.layers)):
                    cifar_pred_dymodel.layers[i].set_weights(cifar_dymodel.layers[i].get_weights())
                error = 0
                # Update sample_weight
                start = time.time()
                # for i in range(train_sample_num):
                #     score = cifar_dymodel.evaluate(X_train[i : i + 1], Y_train[i : i + 1], verbose = 0)
                #     train_acc[i] = score[1]
                prob = cifar_dymodel.predict(X_train, batch_size = train_batch_size, verbose = 1)
                pred_labels = np.argmax(prob, axis = 1)
                train_acc = pred_labels == y_train
                duration = time.time() - start
                print "{} s to evaluate".format(duration)
                train_error = 1 - train_acc
                sample_weight = sample_weight / float(train_sample_num)
                error_rate = (train_error *  sample_weight).sum() / sample_weight.sum()
                alpha[net_num - 1] = np.log(nb_classes - 1) + np.log((1 - error_rate) / error_rate)
                sample_weight = sample_weight * np.exp(alpha[net_num - 1] * train_error)
                print "sahpe of sample_weight", sample_weight.shape
                sample_weight = sample_weight / float(sample_weight.sum())
                sample_weight = sample_weight * train_sample_num

                sample_weight = np.ones(train_sample_num)
                # print "sample_weight:"
                # print sample_weight[0 : 100]
                    
                # get and save the output of the subnetwork
                X_train = cifar_pred_dymodel.predict(X_train, batch_size = train_batch_size)
                X_test = cifar_pred_dymodel.predict(X_test, batch_size = test_batch_size)
                print "shape of X_train: ", X_train.shape
                print "shape of X_test: ", X_test.shape

                # print "shape of X_train: ", X_train.shape
                output_path = gen_output_path(output_path_prefix, r, j)
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                np.savez(output_path + "/output", X_train = X_train, Y_train = Y_train, 
                        X_test = X_test, Y_test = Y_test)

                # save the weight of the subnetwork
                weight_path = gen_weight_path(weight_path_prefix, r, j)
                child = subprocess.Popen("rm -r " + weight_path, shell = True)
                child.wait()
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
                pred = cifar_dymodel.predict(X_test, batch_size = test_batch_size) 
                total_pred = total_pred + alpha[net_num - 1] * pred
                pred_labels = np.argmax(total_pred, axis = 1)
                labels = np.argmax(Y_test, axis = 1)
                total_acc[net_num - 1]  = (labels == pred_labels).sum() / float(X_test.shape[0])
                print "total_acc = ", total_acc
                X_test = cifar_pred_dymodel.predict(X_test, batch_size = test_batch_size)
            else:
                print "Illegal Running mode."
                sys.exit()
    print "alpha: ", alpha
    sys.exit()
    
if "__main__" == __name__:
    main() 
