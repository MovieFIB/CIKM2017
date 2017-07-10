import sys
import os
import cv2
import numpy as np
import glob
import caffe
import feature_config as config
import skimage.transform
import logging

TARGET_IMG_SIZE = 224


def trim_image(img, resnet_mean):
    h, w, c = img.shape
    if c != 3:
        raise Exception('There are gray scale image in data.')

    if h < w:
        w = (w * 256) / h
        h = 256
    else:
        h = (h * 256) / w
        w = 256
    resized_img = skimage.transform.resize(img, (h, w), preserve_range=True)
    cropped_img = resized_img[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 +
                              112, :]
    transposed_img = np.swapaxes(np.swapaxes(cropped_img, 1, 2), 0, 1)
    ivec = transposed_img - resnet_mean
    return ivec[np.newaxis].astype('float32')


def load_videos(video_file):
    # print "load_videos"
    capture = cv2.VideoCapture(video_file)

    read_flag, frame = capture.read()
    vid_frames = []
    i = 1
    # print read_flag

    while (read_flag):
        # print i
        if i % 10 == 0:
            vid_frames.append(frame)
            #                print frame.shape
        read_flag, frame = capture.read()
        i += 1
    vid_frames = np.asarray(vid_frames, dtype='uint8')[:-1]
    # print 'vid shape'
    # print vid_frames.shape
    capture.release()
    print i
    return vid_frames


def compute_features(net, frames, resnet_mean):
    # print 'compute'
    feats = []
    i = 0
    for img in frames:
        # print i
        i += 1
        _img = trim_image(img, resnet_mean)
        net.blobs['data'].data[:] = _img
        net.forward()

        feat_pool5 = net.blobs[
            config.EXTRACT_LAYER_POOL5].data
        feat_pool5 = feat_pool5.flatten()
        feats.append(feat_pool5)
    # print 'compute done'
    return np.asarray(feats, dtype='float32')


def extract_features():
    logging.basicConfig(level=logging.DEBUG, filename='feature_msvd.log', filemode='w')
    pool5_dir = '../feature/pool5/'
    caffe.set_mode_gpu()
    caffe.set_device(config.GPU_ID)

    net = caffe.Net('/home/chenjie/feature/resnet/Resnet/ResNet-152-deploy.prototxt',
                    '/home/chenjie/feature/resnet/Resnet/ResNet-152-model.caffemodel', caffe.TEST)

    # mean substraction
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open('/home/chenjie/feature/resnet/Resnet/ResNet_mean.binaryproto', 'rb').read()
    blob.ParseFromString(data)
    resnet_mean = np.array(caffe.io.blobproto_to_array(blob))[
        0]

    videolistfile = 'videofilepath.txt'
    fp = open(videolistfile)
    i = 1
    for video_name in fp:
        i += 1
        video_name = video_name.strip()
        vid_name = video_name.split('/')[-1][:-4]
        frames = load_videos(video_name)
        logging.info('%d,%d' % (i, frames.shape[0]))
        res_feat_pool5 = compute_features(net, frames, resnet_mean)
        np.save(pool5_dir + vid_name, res_feat_pool5)
    print "DONE"

if __name__ == '__main__':
    extract_features()

