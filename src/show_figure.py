#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-t", "--title",
            help = "title of figure", 
            action = "store", default = "data")
    return parser

parser = argparser()
args = parser.parse_args()
path = args.path
title = args.title

data = np.loadtxt(path)
plt.figure(1)
plt.plot(data)
plt.legend()
plt.grid(True)
plt.title(title)
plt.show()
