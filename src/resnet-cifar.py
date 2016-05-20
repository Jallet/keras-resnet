#!/usr/bin/env python
from resnet import *
import resnet
# from resnet import _basic_block
import time

def cifar_resnet(blocks, filters, repetations):
    if blocks != len(repetations) or blocks != len(filters):
        print "size of blocks, size of repetations, size of filters should match"
        return None
    input = Input(shape = (3, 32, 32))
    conv1 = Convolution2D(nb_filter = 16, nb_row = 3, nb_col = 3, border_mode = "same")(input)
    norm1 = BatchNormalization(mode = 0, axis = 1)(conv1)
    relu1 = Activation("relu")(norm1)
    
    # block_fun = getattr(resnet, "_basic_block")
    block_fun = resnet._basic_block
    
    data = relu1
    for i in range(blocks):
        is_first_layer = False
        if i == 0:
            is_first_layer = True
        data = resnet._residual_block(block_fun, nb_filters = filters[i], 
                repetations = repetations[i], is_first_layer = is_first_layer)(data) 
    global_pool = AveragePooling2D(pool_size = (8, 8), strides = (8, 8), border_mode = "valid")(data)
    flatten = Flatten()(global_pool)
    dense = Dense(output_dim = 10, init = "he_normal", activation = "softmax")(flatten)
    
    model = Model(input = input, output = dense)
    return model
    
def main():
    start = time.time()
    cifar_model = cifar_resnet(3, [16, 32, 64], [5, 8, 10])
    duration = time.time() - start
    print "{} s to make model".format(duration)
    plot(cifar_model, to_file = "resnet-cifar.png", show_shapes = True)
if "__main__" == __name__:
    main()
