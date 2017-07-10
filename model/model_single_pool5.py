#!/usr/bin/Python
# -*- coding: utf-8 -*-

from lasagne.layers import InputLayer, LSTMLayer
from lasagne.layers import ConcatLayer, DenseLayer, DropoutLayer
from collections import OrderedDict
from MyLayers import Tensor3Sub
from MyLayers import Tensor3LinearLayer
from MyLayers import RepeatLayer, AttenLayer, TensorSplitLayer
# from MyRNNLayers import AttenLSTMLayer, OutSTLSTMLayer
from AdaptiveLSTM import AdaptiveLSTMLayer
from OutStHtLSTM import OutStHtLSTMLayer
import config


def build_model(
    video_f, video_mask,
    text_input_before, text_mask_before,
    text_input_after, text_mask_after
):
    net = OrderedDict()

    # video netword
    net['video_input'] = InputLayer(
        shape=(
            None, config.video_frames, config.video_feature_dim
        ),
        input_var=video_f
    )
    net['video_mask'] = InputLayer(
        shape=(None, config.video_frames),
        input_var=video_mask
    )
    # linear
    net['video_linear'] = Tensor3LinearLayer(
        net['video_input'],
        num_units=config.video_linear_dim
    )
    net['drop_video'] = DropoutLayer(net['video_linear'], p=config.video_drop)
    # mean pooling
    # net['video_mean_pooling'] = MeanPoolTensor4(net['video_linear'])
    # video LSTM===>H
    net['video_lstm'] = LSTMLayer(
        net['drop_video'],
        num_units=config.video_lstm_dim,
        mask_input=net['video_mask'],
        only_return_final=False
    )

    # generation model
    net['word_input_before'] = InputLayer(
        shape=(None, config.word_len_before, config.word_emb),
        input_var=text_input_before
    )
    net['word_input_after'] = InputLayer(
        shape=(None, config.word_len_after, config.word_emb),
        input_var=text_input_after
    )

    net['word_mask_before'] = InputLayer(
        shape=(None, config.word_len_before),
        input_var=text_mask_before
    )
    net['word_mask_after'] = InputLayer(
        shape=(None, config.word_len_before),
        input_var=text_mask_after
    )

    net['hn'] = Tensor3Sub(
        net['video_lstm'],
        idx=-1
    )

    net['hn_repeat_before'] = RepeatLayer(
        net['hn'],
        num_copies=config.word_len_before
    )

    net['hn_repeat_after'] = RepeatLayer(
        net['hn'],
        num_copies=config.word_len_after
    )

    net['merge_word_hn_before'] = ConcatLayer(
        [net['word_input_before'], net['hn_repeat_before']],
        axis=-1
    )

    net['merge_word_hn_after'] = ConcatLayer(
        [net['word_input_after'], net['hn_repeat_after']],
        axis=-1
    )

    net['atten_lstm_before'] = AdaptiveLSTMLayer(
        net['merge_word_hn_before'],
        num_units=config.ht_dim,
        num_dims=config.It_dim,
        hid_init=net['hn'],
        cell_init=net['hn'],
        mask_input=net['word_mask_before'],
        visual_input=net['video_lstm']
    )

    net['atten_lstm_after'] = AdaptiveLSTMLayer(
        net['merge_word_hn_after'],
        num_units=config.ht_dim,
        num_dims=config.It_dim,
        hid_init=net['hn'],
        cell_init=net['hn'],
        backwards=True,
        visual_input=net['video_lstm'],
        mask_input=net['word_mask_after']
    )

    net['ht_st_before'] = OutStHtLSTMLayer(
        net['atten_lstm_before'],
        num_units=config.st_dim,
        hid_init=net['hn'],
        cell_init=net['hn'],
        mask_input=net['word_mask_before']
    )

    net['ht_lstm_before'] = TensorSplitLayer(net['ht_st_before'], idx=0)
    net['st_lstm_before'] = TensorSplitLayer(net['ht_st_before'], idx=1)

    net['ht_st_after'] = OutStHtLSTMLayer(
        net['atten_lstm_after'],
        num_units=config.st_dim,
        hid_init=net['hn'],
        cell_init=net['hn'],
        backwards=True,
        mask_input=net['word_mask_after']
    )

    net['ht_lstm_after'] = TensorSplitLayer(net['ht_st_after'], idx=0)
    net['st_lstm_after'] = TensorSplitLayer(net['ht_st_after'], idx=1)

    net['ht_concat'] = ConcatLayer(
        [net['ht_lstm_before'], net['ht_lstm_after']],
        axis=-1
    )

    net['st_concat'] = ConcatLayer(
        [net['st_lstm_before'], net['st_lstm_after']],
        axis=-1
    )

    net['ht_concat_linear'] = DenseLayer(
        net['ht_concat'],
        num_units=config.ht_linear_dim
    )

    net['ht_c_drop'] = DropoutLayer(net['ht_concat_linear'])

    net['st_concat_linear'] = DenseLayer(
        net['st_concat'],
        num_units=config.st_linear_dim
    )

    net['st_c_drop'] = DropoutLayer(net['st_concat_linear'])

    net['fill'] = AttenLayer(
        [net['st_c_drop'], net['ht_c_drop'], net['video_lstm']],
        num_units=config.vocab_size
    )

    return net


