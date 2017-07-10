#!/usr/bin/Python
# -*- coding: utf-8 -*-

import model.model_single_pool5 as msd
import theano
import theano.tensor as T
from lasagne.layers import get_output, get_all_param_values, get_all_params
from lasagne.layers import set_all_param_values, get_all_layers
from lasagne.objectives import categorical_crossentropy
from lasagne.utils import floatX
from lasagne.regularization import regularize_layer_params, l2
from lasagne.updates import adam
import config
import numpy as np
import os
import time
import sys
from utils.dataprovider import DataProvider
sys.setrecursionlimit(1500)


def compile_func(model=None):
    video_data = T.tensor3(name='video_data')
    video_mask = T.imatrix(name='video_mask')
    text_before = T.tensor3(name='text_before')
    text_mask_before = T.imatrix(name='text_mask_before')
    text_after = T.tensor3(name='text_after')
    text_mask_after = T.imatrix(name='text_mask_after')
    label = T.ivector('label')

    net = msd.build_model(
        video_data, video_mask,
        text_before, text_mask_before,
        text_after, text_mask_after
    )

    # for pretrain
    if model is not None:
        set_all_param_values(net['fill'], model)

    prediction = get_output(net['fill'])
    loss = categorical_crossentropy(prediction, label)
    loss = loss.mean()
    all_layers = get_all_layers(net['fill'])
    l2_penalty = regularize_layer_params(all_layers, l2)*config.weight_decay
    loss += l2_penalty
    sh_lr = theano.shared(floatX(config.lr))
    paramaters = get_all_params(net['fill'], trainable=True)
    updates = adam(loss, paramaters, sh_lr)

    train_fun = theano.function(
        inputs=[
            video_data, video_mask,
            text_before, text_mask_before,
            text_after, text_mask_after,
            label
        ],
        outputs=loss,
        updates=updates
    )

    test_prediction = get_output(net['fill'], deterministic=True)
    test_acc = T.mean(
        T.eq(T.argmax(test_prediction, axis=1), label),
        dtype=theano.config.floatX
    )

    test_fun = theano.function(
        inputs=[
            video_data, video_mask,
            text_before, text_mask_before,
            text_after, text_mask_after,
            label
        ],
        outputs=[test_prediction, test_acc]
    )

    val_fun = theano.function(
        inputs=[
            video_data, video_mask,
            text_before, text_mask_before,
            text_after, text_mask_after,
            label
        ],
        outputs=[test_prediction, loss, test_acc]
    )

    return net, train_fun, test_fun, val_fun, sh_lr


def testModel(modelidx):
        dp = DataProvider(
            config.batchsize, config.video_frames,
            config.video_feature_num, config.video_feature_dim,
            config.word_len_before, config.word_len_after,
            video_dir=config.video_data_dir,
            dataset_dir=config.text_data_dir,
            word2vec_dir=config.word2vec_model_dir,
            wordemb_dim=config.word_emb
        )
        modelFile = os.path.join(config.modelsave_dir, str(modelidx)+'_pool5.npy')
        model = np.load(modelFile)
        net, train_fun, test_fun, val_fun, sh_lr = compile_func(model)
        # test
        test_sample = 0
        total_acc = 0

        test_data = dp.load_dataset(datatype='test')
        for batch in dp.iterator(test_data, shuffle=False):
            num_samples,\
                videos, video_masks,\
                befores, before_masks, afters, afters_masks,\
                lables = \
                dp.loadOneBatch(batch)

            prediction, acc = test_fun(
                videos, video_masks,
                befores, before_masks,
                afters, afters_masks,
                lables
            )
            total_acc += acc*num_samples
            test_sample += num_samples
        # fp_test.write('epoch: '+str(epoch)+'err: '+str(total_err)+'\n')
        total_acc /= test_sample
        print 'acc: '+str(total_acc)


def main():
    dp = DataProvider(
        config.batchsize, config.video_frames,
        config.video_feature_num, config.video_feature_dim,
        config.word_len_before, config.word_len_after,
        video_dir=config.video_data_dir,
        dataset_dir=config.text_data_dir,
        word2vec_dir=config.word2vec_model_dir,
        wordemb_dim=config.word_emb
    )
    train_data = dp.load_dataset(datatype='train')
    val_data = dp.load_dataset(datatype='val')

    model = None
    if config.startidx > 0:
        modelFile = os.path.join(config.modelsave_dir, str(config.startidx-1)+'_pool5.npy')
        model = np.load(modelFile)

    print '----------compile-------------'
    net, train_fun, test_fun, val_fun, sh_lr = compile_func(model)
    print '-------compile end------------'

    train_file = os.path.join(config.performance_dir, 'train_pool5.txt')
    val_file = os.path.join(config.performance_dir, 'val_pool5.txt')

    fp_train = open(train_file, 'a+')
    fp_train.write('--------------\n')
    fp_train.write(time.strftime('%Y-%m-%d\t%H:%M:%S', time.localtime(time.time()))+'\n')
    fp_train.write(str(config.batchsize)+'\n')
    fp_train.write(str(config.num_epoches)+'\n')
    fp_train.write(str(config.weight_decay)+'\n')
    fp_train.write(str(config.lr)+'\n')
    fp_train.write('-------result-------\n')
    fp_train.close()

    fp_val = open(val_file, 'a+')
    fp_val.write('--------------\n')
    fp_val.write(time.strftime('%Y-%m-%d\t%H:%M:%S', time.localtime(time.time()))+'\n')
    fp_val.write(str(config.batchsize)+'\n')
    fp_val.write(str(config.num_epoches)+'\n')
    fp_val.write(str(config.weight_decay)+'\n')
    fp_val.write(str(config.lr)+'\n')
    fp_val.write('-------result-------\n')
    fp_val.close()

    for epoch in range(config.num_epoches):
        if epoch < config.startidx:
            continue
        print '-----------epoch '+str(epoch)+'-------------'

        # train
        train_sample = 0
        total_err = 0

        i = 0
        for batch in dp.iterator(train_data, shuffle=True):
            num_samples,\
                videos, video_masks,\
                befores, before_masks, afters, afters_masks,\
                lables = \
                dp.loadOneBatch(batch)

            err = train_fun(
                videos, video_masks,
                befores, before_masks,
                afters, afters_masks,
                lables
            )
            print 'train '+str(epoch)+':'+str(i)+':'+str(err)
            i += 1
            total_err += err*num_samples
            train_sample += num_samples
        model_train = get_all_param_values(net['fill'])
        model_train_file = os.path.join(
            config.modelsave_dir, str(epoch)+'_pool5.npy'
        )
        np.save(model_train_file, model_train)
        total_err /= train_sample
        fp_train = open(train_file, 'a+')
        fp_train.write('epoch: '+str(epoch)+'\t err: '+str(total_err)+'\n')
        fp_train.close()
        print 'train:\t epoch: '+str(epoch)+'\t err: '+str(total_err)

        # val
        val_sample = 0
        total_err = 0
        total_acc = 0
        i = 0
        for batch in dp.iterator(val_data, shuffle=False):
            num_samples,\
                videos, video_masks,\
                befores, before_masks, afters, afters_masks,\
                lables = \
                dp.loadOneBatch(batch)

            prediction, err, acc = val_fun(
                videos, video_masks,
                befores, before_masks,
                afters, afters_masks,
                lables
            )
            print 'val '+str(epoch)+':'+str(i)
            i += 1
            total_err += err*num_samples
            total_acc += acc*num_samples
            val_sample += num_samples
        total_err /= val_sample
        total_acc /= val_sample
        fp_val = open(val_file, 'a+')
        fp_val.write('epoch: '+str(epoch)+'\t err: '+str(total_err)+'\t acc: '+str(total_acc)+'\n')
        fp_val.close()
        print 'val:\t epoch: '+str(epoch)+'\t err: '+str(total_err)+'\t acc: '+str(total_acc)

        if (epoch+1) % config.lr_change == 0:
            sh_lr.set_value(sh_lr.get_value()/10)


if __name__ == '__main__':
    train = True
    modelidx = 11  # change to your model index.
    if train:
        main()
    else:
        testModel(modelidx)
