#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from Variable import Variable
from Function import Function, Square, Exp, Add

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def add(x0, x1):
    return Add()(x0, x1)

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad) 