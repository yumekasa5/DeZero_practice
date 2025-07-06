#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported.')
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            # 微分がなければself.dataと同じ形状かつ同じデータ型で、その要素が1のndarrayインスタンスを生成する
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            """ここでインターナルなメソッドを定義する理由は以下の要件を満足するからである
            1. 親となるbackwardメソッドの中でしか利用しない
            2. 親となるbackwardメソッドの中で利用している変数にアクセスする必要がある
            """
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)  # TODO: 優先度付きキューに変更する

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    # 別のオブジェクトとして計算したいのでin-place演算(+=)はしない
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

    def clearegrad(self):
        """微分を初期化する"""
        self.grad = None


class Function:
    """関数の基底クラス"""

    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # 入力変数の内、最も大きい世代を関数の世代として設定する
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
