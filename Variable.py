#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    """Base class for all functions."""
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        self.input = input
        output = Variable(y)
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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
y = A(x)
y = B(y)
y = C(y)
print(y.data)

