#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import weakref
import contextlib


class Config:
    """設定クラス"""
    enable_backprop = True  # 逆伝播の有効/無効を切り替えるフラグ


@contextlib.contextmanager
def using_config(name, value):
    """設定を一時的に変更するコンテキストマネージャ"""
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    __array_priority__ = 200  # ndarrayの演算子より優先されるようにする

    def __init__(self, data: np.ndarray, name: str = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported.')
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
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
            gys = [output().grad for output in f.outputs]
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

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def clearegrad(self):
        """微分を初期化する"""
        self.grad = None

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'


class Function:
    """関数の基底クラス"""

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            # 入力変数のうち、最も大きい世代を関数の世代として設定する
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            # weakrefにより出力は弱参照で循環参照を防ぐ
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


def as_array(x):
    """オブジェクトをndarrayに変換する"""
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    """オブジェクトをVariableに変換する"""
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


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


def add(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


# 演算子のオーバーロードを定義する
Variable.__add__ = add   # Variableクラスのインスタンスに対して+演算子を使えるようにする
Variable.__radd__ = add  # Variableクラスのインスタンスに対して+演算子を使えるようにする（右辺のとき）
Variable.__mul__ = mul   # Variableクラスのインスタンスに対して*演算子を使えるようにする
Variable.__rmul__ = mul  # Variableクラスのインスタンスに対して*演算子を使えるようにする（右辺のとき）
