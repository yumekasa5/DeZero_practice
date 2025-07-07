#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def goldstein_numpy(x, y):
    """NumPyを使ったGoldstein-Price関数"""
    z = (1 + ((x + y + 1) ** 2) * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
        (30 + ((2 * x - 3 * y) ** 2) * (18 - 32 * x +
         12 * x ** 2 + 12 * y - 36 * x * y + 27 * y ** 2))
    return z


def numerical_gradient(f, x, y, h=1e-4):
    """数値微分で勾配を計算"""
    grad_x = (f(x + h, y) - f(x - h, y)) / (2 * h)
    grad_y = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return grad_x, grad_y


# 点(1, 1)での勾配を数値微分で計算
x, y = 1.0, 1.0
grad_x, grad_y = numerical_gradient(goldstein_numpy, x, y)
print(f"数値微分による勾配: dx={grad_x:.6f}, dy={grad_y:.6f}")

# 関数値も確認
f_val = goldstein_numpy(x, y)
print(f"関数値: {f_val}")
