import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import  BatchNormalization, Reshape, ReLU , Input ,Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)   #alpha：负数部分的斜率。默认为 0。
original_dim = 1024
class Shuffle(Layer):
#"""打乱层，提供两种方式打乱输入维度
#一种是直接反转，一种是随机打乱，默认是直接反转维度
##通过随机的方式将向量打乱，可以使信息混合得更加充分，最终的loss可以更低,
## 在NICE中，作者通过交错的方式来混合信息流（这也理论等价于直接反转原来的向量）
#"""
    def __init__(self, idxs=None, mode='reverse', **kwargs):
        super(Shuffle, self).__init__(**kwargs)
        self.idxs = idxs
        self.mode = mode
    def call(self, inputs):

        v_dim = K.int_shape(inputs)[-1]
        if self.idxs == None:
            self.idxs = list(range(v_dim))
            if self.mode == 'reverse':
                self.idxs = self.idxs[::-1]
            elif self.mode == 'random':
                np.random.shuffle(self.idxs)
        inputs = K.transpose(inputs)
        outputs = K.gather(inputs, self.idxs)
        outputs = K.transpose(outputs)
        return outputs
    def inverse(self):
        v_dim = len(self.idxs)
        _ = sorted(zip(range(v_dim), self.idxs), key=lambda s: s[1])
        reverse_idxs = [i[0] for i in _]
        return Shuffle(reverse_idxs)

class SplitVector(Layer):
    """将输入分区为两部分，交错分区
    ##就是指将每一步flow输出的两个向量h1,h2拼接成一个向量h，然后将这个向量重新随机排序。
    """
    def __init__(self, **kwargs):
        super(SplitVector, self).__init__(**kwargs)
    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        inputs = K.reshape(inputs, (-1, v_dim//2, 2))
        return [inputs[:,:,0], inputs[:,:,1]]
    def compute_output_shape(self, input_shape):
        v_dim = input_shape[-1]
        return [(None, v_dim//2), (None, v_dim//2)]
    def inverse(self):
        layer = ConcatVector()
        return layer

class ConcatVector(Layer):
    """将分区的两部分重新合并
    """
    def __init__(self, **kwargs):
        super(ConcatVector, self).__init__(**kwargs)
    def call(self, inputs):
        inputs = [K.expand_dims(i, 2) for i in inputs]
        inputs = K.concatenate(inputs, 2)
        return K.reshape(inputs, (-1, np.prod(K.int_shape(inputs)[1:])))
    def compute_output_shape(self, input_shape):
        return (None, sum([i[-1] for i in input_shape]))
    def inverse(self):
        layer = SplitVector()
        return layer

class AddCouple(Layer):
    """加性耦合层
    """
    def __init__(self, isinverse=False, **kwargs):
        self.isinverse = isinverse
        super(AddCouple, self).__init__(**kwargs)
    def call(self, inputs):
        part1, part2, mpart1 = inputs
        if self.isinverse:
            return [part1, part2 + mpart1] # 逆为加
        else:
            return [part1, part2 - mpart1] # 正为减
    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]]
    def inverse(self):
        layer = AddCouple(True)
        return layer

class Scale(Layer):

    """Scale transformation layer
    flow, a model based on invertible transformations, inherently has a serious dimensional waste problem: the input data are clearly not D-dimensional flows, but they are encoded as a D-dimensional flow.
    But it is feasible to encode it as a D-dimensional flow? In order to solve this situation, NICE introduces a scale transformation layer.
    NICE introduces a scale transformation layer, which does a scale transformation on each dimension of the final encoded features
    """
    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[1]),
                                      initializer='glorot_normal',
                                      trainable=True)
    def call(self, inputs):
        self.add_loss(-K.sum(self.kernel))
        return K.exp(self.kernel) * inputs
    def inverse(self):
        scale = K.exp(-self.kernel)
        return Lambda(lambda x: scale * x)

def build_basic_model_v2(v_dim):
    """
    The base model, i.e., the additive coupling layer in the m
    """
    _in = Input(shape=(v_dim,))

    _in = BatchNormalization()(_in)
    _ = _in
    _ = Dense(v_dim, activation=None)(_)
    for i in range(5):
        _ = Dense(1000, activation=None, kernel_regularizer=tf.keras.regularizers.l1(l=0.1))(_)   #加入L1正则化，防止过拟合
        _ = BatchNormalization()(_)
        _ = _leaky_relu(_)
    _ = Dense(v_dim, activation=_leaky_relu)(_)
    out = _
    return Model(_in, out)
shuffle1 = Shuffle()
shuffle2 = Shuffle()
shuffle3 = Shuffle()
shuffle4 = Shuffle()
split = SplitVector()
couple = AddCouple()
concat = ConcatVector()
scale = Scale()

basic_model_1 = build_basic_model_v2(original_dim//2)
basic_model_2 = build_basic_model_v2(original_dim//2)
basic_model_3 = build_basic_model_v2(original_dim//2)
basic_model_4 = build_basic_model_v2(original_dim//2)

def build_NICE1(original_dim):

    #Structure of the encoder
    x_in = Input(shape=(original_dim,))
    x = x_in

    # Add negative noise to the input to prevent overfitting
    x = Lambda(lambda s: K.in_train_phase(s-0.01*K.random_uniform(K.shape(s)), s))(x)
    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_1(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle2(x)
    x1,x2 = split(x)
    mx1 = basic_model_2(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle3(x)
    x1,x2 = split(x)
    mx1 = basic_model_3(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle4(x)
    x1,x2 = split(x)
    mx1 = basic_model_4(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = scale(x)

    return x_in,x

def build_NICE2(original_dim):

    # Structure of the encoder
    x_in = Input(shape=(original_dim,))
    x = x_in

    # Add negative noise to the input to prevent overfitting
    x = Lambda(lambda s: K.in_train_phase(s-0.01*K.random_uniform(K.shape(s)), s))(x)
    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_1(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle2(x)
    x1,x2 = split(x)
    mx1 = basic_model_2(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle3(x)
    x1,x2 = split(x)
    mx1 = basic_model_3(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle4(x)
    x1,x2 = split(x)
    mx1 = basic_model_4(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = scale(x)

    return x_in,x

def build_NICE_reverse(original_dim):

    # Build the inverse model (generative model) and execute all operations in reverse
    x_in = Input(shape=(original_dim,))
    x = x_in

    x = scale.inverse()(x)

    x1,x2 = concat.inverse()(x)
    mx1 = basic_model_4(x1)
    x1, x2 = couple.inverse()([x1, x2, mx1])
    x = split.inverse()([x1, x2])
    x = shuffle4.inverse()(x)

    x1,x2 = concat.inverse()(x)
    mx1 = basic_model_3(x1)
    x1, x2 = couple.inverse()([x1, x2, mx1])
    x = split.inverse()([x1, x2])
    x = shuffle3.inverse()(x)

    x1,x2 = concat.inverse()(x)
    mx1 = basic_model_2(x1)
    x1, x2 = couple.inverse()([x1, x2, mx1])
    x = split.inverse()([x1, x2])
    x = shuffle2.inverse()(x)

    x1,x2 = concat.inverse()(x)
    mx1 = basic_model_1(x1)
    x1, x2 = couple.inverse()([x1, x2, mx1])
    x = split.inverse()([x1, x2])
    x = shuffle1.inverse()(x)

    return x_in, x
