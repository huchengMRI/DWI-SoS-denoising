import nibabel as nib
import os
import csv
import numpy as np
import dipy
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from numpy.random import rand
from random import seed
from datetime import datetime
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

sigLength = 199     # length of DWI signal
fbval = 'dwi.bvals'
fbvec = 'dwi.bvecs'
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)
tenmodel = dti.TensorModel(gtab)
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

def mutual_information(img1, img2, bins=20):
    """
    measure the mutual information of the given two images

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    bins: optional (default=20)
        bin size of the histogram

    Returns
    -------
    calculated mutual information: float

    """
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins)

    # convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal x over y
    py = np.sum(pxy, axis=0)  # marginal y over x
    px_py = px[:, None] * py[None, :]  # broadcast to multiply marginals

    # now we can do the calculation using the pxy, px_py 2D arrays
    nonzeros = pxy > 0  # filer out the zero values
    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))

def main():
    # brain mask
    tmp = nib.load('nodif_brain_mask.nii.gz')
    msk_img = tmp.get_fdata()
    sz = msk_img.shape
    numVol = np.int(sum(msk_img.ravel()))
    msk = np.reshape(msk_img, sz[0]*sz[1]*sz[2])
    # the normal noised data
    tmp = nib.load('dwi_SENSE.nii.gz')
    normal_img = tmp.get_fdata()
    sz = normal_img.shape
    imgRaw = np.transpose(np.reshape(normal_img, (sz[0]*sz[1]*sz[2], sz[3])))
    sigRaw = np.delete(imgRaw, np.where(msk != 1), axis=1)

    # High-noise data
    tmp = nib.load('dwi_SoS.nii.gz')
    imgNoised = np.transpose(np.reshape(tmp.get_fdata(), (sz[0]*sz[1]*sz[2], sz[3])))
    sigNoised = np.delete(imgNoised, np.where(msk != 1), axis=1)

    #use 75% random voxels for training
    rn = rand(numVol)
    #i = np.argsort(rn)
    sigRaw_rand = sigRaw
    sigNoised_rand = sigNoised
    sigRaw_for_training = sigRaw_rand[:,rn>0.25]
    sigNoised_for_training = sigNoised_rand[:,rn>0.25]
    sigRaw_for_testing = sigRaw_rand[:,rn<0.25]
    sigNoised_for_testing = sigNoised_rand[:,rn<0.25]
    numVol_training = np.int(sum(rn>0.25))
    numVol_testing = np.int(sum(rn<0.25))
    # important learning parameters
    learning_rate = 0.002
    num_epochs = 20000

    input = tf.placeholder(tf.float32, [None, sigLength])
    labels = tf.placeholder(tf.float32, [None, sigLength])

    output = getModel(input)
    loss = tf.reduce_mean(tf.square(output - labels))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    #saver.restore(sess, "./ckpt/RawSignal_model.ckpt") # used to load a trained model in testing

    mse_summary = tf.summary.scalar('Loss', loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    
    count = 0
    batch_size = 3000
    lr = str(learning_rate)
    basename = "bs"+str(batch_size)+"lr"+lr[2:]
    outfile = "loss_"+basename+".csv"
    with open(outfile, 'w', newline='') as csvfile:
        fieldnames = ['counts', 'loss']
        writer = csv.writer(csvfile, delimiter=',', 
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
    flag = True

    while flag:
        size = 0
        # Mini batching with the given batch size
        for i in range(0, numVol_training, batch_size):
            size += batch_size
            if size <= numVol_training:
                batch_x = sigNoised_for_training[:, i: size]
                batch_y = sigRaw_for_training[:, i: size]
            else:
                batch_x = sigNoised_for_training[:, i: numVol_training]
                batch_y = sigRaw_for_training[:, i: numVol_training]

            feed_dict = {input: batch_x.T, labels: batch_y.T}
            train_step.run(feed_dict=feed_dict)

        if count % 100 == 0: # '100' is the step to print current info

            summary_str = mse_summary.eval(feed_dict=feed_dict)
            file_writer.add_summary(summary_str, count)

            loss_calc = loss.eval(feed_dict=feed_dict)
            print("Epoch %d, loss %g" % (count, loss_calc))
            with open(outfile, 'a') as fd:
                writer = csv.writer(fd)
                writer.writerow([count, loss_calc])

            # save checkpoint to a file so that we can start from the learned model (checkpoint) and continue learning
            saver.save(sess, "./ckpt_"+basename+"/model_"+basename+".ckpt")

        # Once all the epochs are completed, training is stopped
        if count >= num_epochs:
            flag = False

        count += 1

    # save checkpoint to a file so that we can load the model (checkpoint) in testing
    saver.save(sess, "./ckpt_"+basename+"/model_"+basename+".ckpt")

    x = np.zeros((sz[3], numVol_testing))
    size = 0
    # Mini batching with the given batch size
    for i in range(0, numVol_testing, batch_size):
        size += batch_size
        if size <= numVol_testing:
            batch_tmp = sigNoised_for_testing[:, i: size]
            x[:, i: size] = output.eval(feed_dict={input: batch_tmp.T}).T
        else:
            batch_tmp = sigNoised_for_testing[:, i: numVol_testing]
            x[:, i: numVol_testing] = output.eval(feed_dict={input: batch_tmp.T}).T
    mI = np.zeros(sz[3])
    corr = np.zeros(sz[3])
    mse = np.sqrt(np.mean(np.square(x.ravel()-sigRaw_for_testing.ravel())))
    tenfit_out = tenmodel.fit(np.transpose(x))
    FA_out = tenfit_out.fa
    tenfit_raw = tenmodel.fit(np.transpose(sigRaw_for_testing))
    FA_raw = tenfit_raw.fa
    mse_FA = np.sqrt(np.mean(np.square(FA_out.ravel()-FA_raw.ravel())))
    for ii in range(sz[3]):
        v1 = x[ii,:]
        v2 = sigRaw_for_testing[ii,:]
        mI[ii] = mutual_information(v1, v2)   
        r = np.corrcoef(v1,v2)
        corr[ii] = r[0,1]
    print("mutual information = ", mI)
    print("correlation = ", corr)
    with open(outfile, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(["mse", mse])
        writer.writerow(["mse_FA", mse_FA])
        writer.writerow(["mutual info", np.mean(mI)])
        writer.writerow(["correlation", np.mean(corr)])

if __name__ == '__main__':
    main()
