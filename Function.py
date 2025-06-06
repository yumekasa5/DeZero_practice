#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from Variable import Variable

class Function:
    """関数の基底クラス"""
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        self.input = input
        output = Variable(y)
        output.set_creator(self)
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx