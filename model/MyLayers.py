#!/usr/bin/Python
# -*- coding: utf-8 -*-

from lasagne.layers import Layer, MergeLayer
import theano.tensor as T
from lasagne import nonlinearities, init
# import config


class Tensor4LinearLayer(Layer):
    def __init__(
        self, incoming, num_units,
        W=init.Constant(0.1),
        b=init.Constant(0.),
        nonlinearity=nonlinearities.rectify,
        **kwargs
    ):
        super(Tensor4LinearLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.num_units = num_units
        self.W = self.add_param(
            W, (num_inputs, num_units),
            name="W"
        )
        if b:
            self.b = self.add_param(
                b,
                (
                    self.input_shape[1],
                    self.input_shape[2], self.num_units
                )
            )
        else:
            self.b = None
        if nonlinearity:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = nonlinearities.identity

    def get_output_for(self, input, **kwargs):
        r = T.dot(input, self.W)
        if self.b:
            r = r + self.b
        return self.nonlinearity(r)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.num_units)


class Tensor3LinearLayer(Layer):
    def __init__(
        self, incoming, num_units,
        W=init.Constant(0.1),
        b=init.Constant(0.),
        nonlinearity=nonlinearities.rectify,
        **kwargs
    ):
        super(Tensor3LinearLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.num_units = num_units
        self.W = self.add_param(
            W, (num_inputs, num_units),
            name="W"
        )
        if b:
            self.b = self.add_param(
                b,
                (
                    self.input_shape[1], self.num_units
                )
            )
        else:
            self.b = None
        if nonlinearity:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = nonlinearities.identity

    def get_output_for(self, input, **kwargs):
        r = T.dot(input, self.W)
        if self.b:
            r = r + self.b
        return self.nonlinearity(r)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)


class MeanPoolTensor4(Layer):
    def __init__(self, incoming, axis=-2, **kwargs):
        super(MeanPoolTensor4, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return T.mean(input, axis=self.axis)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[-1])


class Tensor3Sub(Layer):
    def __init__(self, incoming, idx, **kwargs):
        super(Tensor3Sub, self).__init__(incoming, **kwargs)
        self.idx = idx
        self.dim = self.input_shape[-1]
        self.batchsize = self.input_shape[0]

    def get_output_for(self, input, **kwargs):
        out = input[:, self.idx, :]
        out = out.reshape((-1, self.dim))
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


class RepeatLayer(Layer):
    def __init__(self, incoming, num_copies, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.copies = num_copies

    def get_output_for(self, input, **kwargs):
        out = T.stack([input for i in range(self.copies)], axis=0)
        out = out.dimshuffle(1, 0, 2)
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.copies, input_shape[-1])


class AttenLayer(MergeLayer):
    # st, ht, H
    def __init__(
        self, incomings, num_units,
        W_g=init.Normal(0.1),
        W_h=init.Normal(0.1),
        W_v=init.Normal(0.1),
        W_s=init.Normal(0.1),
        W_p=init.Normal(0.1),
        nonlinearity=nonlinearities.tanh,
        nonlinearity_atten=nonlinearities.softmax,
        **kwargs
    ):
        super(AttenLayer, self).__init__(incomings, **kwargs)
        self.batch_size = self.input_shapes[0][0]  # None
        num_inputs = self.input_shapes[2][1]  # k
        feature_dim = self.input_shapes[0][1]  # d
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.nonlinearity_atten = nonlinearity_atten
        self.W_h_to_attenGate = self.add_param(
            W_h, (num_inputs, 1),
            name='W_h_to_atten'
        )
        self.W_g_to_attenGate = self.add_param(
            W_g,
            (feature_dim, num_inputs),
            name='W_g_to_atten'
        )
        self.W_v_to_attenGate = self.add_param(
            W_v,
            (feature_dim, num_inputs),
            name='W_v_to_atten'
        )
        self.W_s_to_attenGate = self.add_param(
            W_s,
            (feature_dim, num_inputs),
            name='W_s_to_atten'
        )
        self.W_p = self.add_param(
            W_p,
            (feature_dim, num_units),
            name='W_p_to_atten'
        )
        self.num_inputs = num_inputs

    def get_output_for(self, inputs, **kwargs):
        s_hat_t = inputs[0]
        h_hat_t = inputs[1]
        # s_hat_t = s_hat_t.dimshuffle(1, 0)
        # h_hat_t = h_hat_t.dimshuffle(1, 0)
        H = inputs[2]
        # H = H.dimshuffle(2, 0, 1)
        # H_len = H.shape[-1]
        # z_t 1*none*k
        zt = T.dot(
            self.nonlinearity(
                T.dot(H, self.W_v_to_attenGate) +
                T.dot(
                    T.dot(h_hat_t, self.W_g_to_attenGate).dimshuffle(0, 1, 'x'),
                    T.ones((1, self.num_inputs))
                )
            ),
            self.W_h_to_attenGate
        )[:, :, 0]
        vt = T.dot(
            self.nonlinearity(
                T.dot(
                    s_hat_t, self.W_s_to_attenGate
                ) +
                T.dot(
                    h_hat_t, self.W_g_to_attenGate
                )
            ),
            self.W_h_to_attenGate
        )

        alpha_hat_t = self.nonlinearity_atten(T.concatenate(
            [zt, vt],
            axis=-1
        ))
        feature = T.concatenate(
            [H, s_hat_t.dimshuffle(0, 'x', 1)],
            axis=1
        ).dimshuffle(2, 0, 1)
        c_hat_t = T.sum(alpha_hat_t*feature, axis=-1)
        out = T.dot(
            (c_hat_t.T+h_hat_t), self.W_p
        )

        return nonlinearities.softmax(out)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], self.num_units)


class TensorSplitLayer(Layer):
    def __init__(self, incoming, idx, **kwargs):
        super(TensorSplitLayer, self).__init__(incoming, **kwargs)
        self.idx = idx
        self.num_split = self.input_shape[1]/2

    def get_output_for(self, input, **kwargs):
        return input[:, self.num_split*self.idx: self.num_split*(self.idx+1)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_split)
