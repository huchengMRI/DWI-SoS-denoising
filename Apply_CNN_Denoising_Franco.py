# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:49:56 2020

@author: hucheng
"""

import nibabel as nib
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
dir1 = "/home/bacaron/acute_concussion_predenoised/2_036/2/"
sigLength = 199     # length of DWI signal

# construct the deep learning network
def getModel(x):
    # Input Layer
    input_layer = tf.reshape(x, [-1, sigLength, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=16,
        kernel_size=17,
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=32,
        kernel_size=9,
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

    # Dense Layer
    pool2_flat = tf.layers.flatten(pool2)

    logits = tf.layers.dense(inputs=pool2_flat, units=sigLength, activation=tf.nn.relu)

    return logits


def main():
    # brain mask
    
    tmp = nib.load(op.join(dir1,'mask', 'mask.nii.gz'))
    msk_img = tmp.get_fdata()
    # sz = msk_img.shape

    # take half of the brain as training
    # msk_train = np.zeros(sz)
    # msk_train[0:np.int(sz[0]/2), :, :] = msk_img[0:np.int(sz[0]/2), :, :]


    # High-noise data
    tmp = nib.load(op.join(dir1,'dwi','dwi.nii.gz'))
    normal_img = tmp.get_fdata()
    sz = normal_img.shape
    imgNoised = np.transpose(np.reshape(tmp.get_fdata(), (sz[0]*sz[1]*sz[2], sz[3])))

    input = tf.placeholder(tf.float32, [None, sigLength])
    output = getModel(input)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, "./ckpt_bs12000lr0005/model_bs12000lr0005.ckpt") # used to load a trained model in testing
    x = np.zeros((sz[3], sz[0] * sz[1] * sz[2]))    
    size = 0
    # Mini batching with the given batch size
    batch_size = 12000
    for i in range(0, np.int(sz[0] * sz[1] * sz[2]), batch_size):
        size += batch_size
        if size <= (sz[0] * sz[1] * sz[2]):
            batch_tmp = imgNoised[:, i: size]
            x[:, i: size] = output.eval(feed_dict={input: batch_tmp.T}).T
        else:
            batch_tmp = imgNoised[:, i: np.int(sz[0] * sz[1] * sz[2])]
            x[:, i: np.int(sz[0] * sz[1] * sz[2])] = output.eval(feed_dict={input: batch_tmp.T}).T

    x_img = np.float32( np.reshape(np.transpose(x), (sz[0],sz[1],sz[2],sz[3])) )
    img_denoised = x_img

    # use the following two lines of code if you want a result of half brain denoised and half brain high-noise
    # img_denoised = tmp.get_fdata()
    # img_denoised[np.int(sz[0] / 2):-1, :, :,:] = x_img[np.int(sz[0] / 2):-1, :, :,:]
    for ii in range(0,sigLength):
        img_denoised[:,:,:,ii] = img_denoised[:,:,:,ii] * msk_img

    # save the denoised result to a nifti
    x1 = nib.Nifti1Image(img_denoised, tmp.affine, tmp.header)
    nib.save(x1, op.join(dir1,'Denoised.nii.gz'))

if __name__ == '__main__':
    main()
