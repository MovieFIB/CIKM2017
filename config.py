#!/usr/bin/Python
# -*- coding: utf-8 -*-

video_data_dir = '/path/to/feature'
text_data_dir = '/path/to/data'
word2vec_model_dir = '/path/to/data'
modelsave_dir = '/path/to/save/models'
performance_dir = '/path/to/save/performance/files'

thr = 5
startidx = 0
batchsize = 256
num_epoches = 200
weight_decay = 0.00001
lr = 0.001
video_drop = 0.5
lr_change = 1000

video_frames = 30
video_feature_dim = 2048
video_linear_dim = 1024
video_lstm_dim = 512

word_emb = 300  # the dimensonality of word feature
word_len = 30  # the length of sentence
word_len_before = 50
word_len_after = 50

# video_lstm_dim == ht_dim == st_dim == ht_linear_dim == st_linear_dim

It_dim = 512  # the dimensonality of It
ht_dim = 512  # the dimensonality of ht

st_dim = 512  # the dimensonality of st

ht_linear_dim = 512
st_linear_dim = 512

vocab_size = 13976  # the size of vocabulary
