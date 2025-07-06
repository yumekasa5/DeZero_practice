#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from dezero.core_simple import Variable, Function, as_array, Add, Square, Exp


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)
