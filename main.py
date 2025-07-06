#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from dezero.core_simple import Variable, Function, as_array, add, mul, square, exp

x = Variable(np.array(2.0))
y = np.array([12.0]) + x
print(y)
