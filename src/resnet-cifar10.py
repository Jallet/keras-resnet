#!/usr/bin/env python
import numpy as np
from resnet import *
import resnet
# from resnet import _basic_block
import time
import sys
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import RemoteMonitor, Callback, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

def main():
    batch_size = 250
    nb_epoch = 1
    nb_classes = 10
    start = time.time()
    base_lr = 0.1
    print "Loading data"
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # zero-center x
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    duration = time.time() - start
    print "{} s to load data".format(duration)

    start = time.time()
    print "Making model"
    cifar_model = resnet.cifar_resnet(3, [16, 32, 64], [3, 3, 3])
    plot(cifar_model, to_file = "./images/resnet-cifar.png", show_shapes = True)
    duration = time.time() - start
    print "{} s to make model".format(duration)
    print "Compiling model"
    
    start = time.time()
    sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=False)
    def lr_schedule(epoch):
        if 0 <= epoch and epoch < 160:
            return 0.1
        elif 160 <= epoch and epoch <= 240:
            return 0.01
        else:
            return 0.001
    cifar_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    duration = time.time() - start
    print "{} s to compile model".format(duration)

    print "Fitting model"
    start = time.time()
    class LRPrinter (Callback):
        def __init__(self):
            super(LRPrinter, self).__init__()
        def on_epoch_begin(self, epoch, logs = {}):
            assert hasattr(self.model.optimizer, "lr"), \
                    "Optimizer must have a lr attribute"
            lr = self.model.optimizer.lr.get_value()
            momentum = self.model.optimizer.momentum.get_value()
            decay = self.model.optimizer.decay.get_value()
            print "lr: ", lr, " momentum: ", momentum, " decay: ", decay
    
    # datagen = ImageDataGenerator(featurewise_center = True,
    #         featurewise_std_normalization = True,
    #         rotation_range = 20,
    #         width_shift_range = 0.2,
    #         height_shift_range = 0.2,
    #         horizontal_flip = True)
    
    # datagen.fit(X_train)
    # history = cifar_model.fit_generator(datagen.flow(X_train, Y_train ,batch_size = 250),
    #         samples_per_epoch = X_train.shape[0], nb_epoch = nb_epoch,
    #         callbacks = [LearningRateScheduler(lr_schedule), LRPrinter()],
    #         validation_data=(X_test, Y_test))
    # sys.exit()
    history = cifar_model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              callbacks = [LearningRateScheduler(lr_schedule), LRPrinter()],
              validation_data=(X_test, Y_test),
              shuffle=True)
    print "type of history: ", type(history)
    loss = history.history["loss"]
    duration = time.time() - start
    print "{} s to fit model".format(duration)
    print history.history
    
if "__main__" == __name__:
    main()
