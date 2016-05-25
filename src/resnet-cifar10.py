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
from PIL import Image

def random_crop(X_train, pad = 4):
    pad_x = np.pad(X_train, pad_width = ((0, 0), (0, 0), (4, 4), (4, 4)), 
            mode = "constant", constant_values = 0)
    cropped_x = np.zeros(X_train.shape)
    random_col_index = np.random.randint(0, 2 * pad, size = X_train.shape[0]) 
    random_row_index = np.random.randint(0, 2 * pad, size = X_train.shape[0]) 
    cropped_x_cols = cropped_x.shape[2]
    cropped_x_rows = cropped_x.shape[3]
    for i in range(X_train.shape[0]):
        cropped_x[i, :, :, :] = pad_x[i, :, 
                random_col_index[i] : random_col_index[i] + cropped_x_cols,
                random_row_index[i] : random_row_index[i] + cropped_x_rows]
    
    return cropped_x.astype(np.uint8)

def main():
    batch_size = 250
    nb_epoch = 300
    nb_classes = 10
    start = time.time()
    base_lr = 0.1
    print "Loading data"
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    cropped_x = random_crop(X_train, pad = 4)
    # im_array = X_train[5, :, :, :]
    # cim_array = cropped_x[5, :, :, :]
    # im_array = np.transpose(im_array, (1, 2, 0))
    # cim_array = np.transpose(cim_array, (1, 2, 0))
    # print "shape of im: ", im_array.shape
    # print "shape of cim: ", cim_array.shape
    # # print im_array
    # # print cim_array
    # im = Image.fromarray(im_array)
    # cim = Image.fromarray(cim_array)
    # im.show()
    # cim.show()
    # sys.exit()
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
    
    datagen = ImageDataGenerator(horizontal_flip = True)
    # datagen = ImageDataGenerator(featurewise_center = True,
    #         featurewise_std_normalization = True,
    #         rotation_range = 20,
    #         width_shift_range = 0.2,
    #         height_shift_range = 0.2,
    #         horizontal_flip = True)
    
    datagen.fit(X_train)
    history = cifar_model.fit_generator(datagen.flow(X_train, Y_train ,batch_size = 250),
            samples_per_epoch = X_train.shape[0], nb_epoch = nb_epoch,
            callbacks = [LearningRateScheduler(lr_schedule)],  
            validation_data=(X_test, Y_test))
    # sys.exit()
    # history = cifar_model.fit(X_train, Y_train,
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           callbacks = [LearningRateScheduler(lr_schedule), LRPrinter()],
    #           validation_data=(X_test, Y_test),
    #           shuffle=True)
    print "type of history: ", type(history)
    loss = history.history["loss"]
    duration = time.time() - start
    print "{} s to fit model".format(duration)
    print history.history
    
if "__main__" == __name__:
    main()
