import matplotlib.pyplot as plt
import json, glob, os, cv2
from pathlib import Path
import numpy as np
from os.path import join
import itertools
import tensorflow as tf

def visualizeAugData(imgs, labels):
    gridH = 10
    gridW = 10
    fig, axs = plt.subplots(gridH, gridW)

    fig.set_size_inches(20, 20 * (gridH / gridW))

    numImgs = imgs.shape[0]
    Ids = list(range(numImgs))
    np.random.shuffle(Ids)
    print("Number corners:", numImgs)
    for iA, iC in itertools.product(range(gridH), range(gridW)):
        if iC + gridW * iA >= numImgs:
            break
        imgId = Ids[iC + gridW * iA]

        axs[iA, iC].imshow(np.squeeze(imgs[imgId, :, :]), cmap='gray')
        axs[iA, iC].set_title(str(np.argmax(labels[imgId, :])))
        axs[iA, iC].axis('off')

def toOneHot4Heads(data, numClasses=26):
    oneHot = [np.zeros((data.shape[0], numClasses,)) for i in range(4)]

    for i in range(4):
        for ic in range(data.shape[0]):
            oneHot[i][ic, data[ic, i]] = 1

    return oneHot

def toOneHot(label, numClasses):
    onh = np.zeros((label.shape[0], numClasses), dtype=np.float32)
    for i, s in enumerate(label):
        onh[i,s] = 1
    return onh

def getFlattenSize(layer):
    shape = layer.get_shape()
    return shape[-1] * shape[-2] * shape[-3]

def getConvLayers(inputLayer, inputChannel, convSizes, convChannels, maxPoolingPosition,
                  padding='VALID', training_ph=None, batchNorm=False, normalizeInput = True, netName=''):
    ws = []
    bs = []
    assert (len(convChannels) == len(convSizes))

    for i in range(len(convChannels)):
        if i == 0:
            w = tf.get_variable(netName + "WConv%d" % i,
                                shape=[convSizes[i], convSizes[i], inputChannel, convChannels[i]],
                                initializer=tf.initializers.glorot_normal())
        else:
            w = tf.get_variable(netName + "WConv%d" % i,
                                shape=[convSizes[i], convSizes[i], convChannels[i - 1], convChannels[i]],
                                initializer=tf.initializers.glorot_normal())

        b = tf.get_variable(netName + "bConv%d" % i, shape=[convChannels[i]], initializer=tf.initializers.zeros())

        ws.append(w)
        bs.append(b)

        # Should not use drop out in CNN layers!!!
        # if i == 0:
        #     cnn = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(inputLayer/ 255, w, strides = [1,1,1,1], padding = 'VALID') + b), pkeep_ph)
        # else:
        #     cnn = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(cnn, w, strides = [1,1,1,1], padding = 'VALID') + b), pkeep_ph)

        if i == 0:
            if normalizeInput:
                print("Normalizing the input layer by 255 in CNN.")
                cnn = tf.nn.conv2d(inputLayer / 255, w, strides=[1, 1, 1, 1], padding=padding) + b
            else:
                cnn = tf.nn.conv2d(inputLayer, w, strides=[1, 1, 1, 1], padding=padding) + b
        else:
            cnn = tf.nn.conv2d(cnn, w, strides=[1, 1, 1, 1], padding=padding) + b

        if batchNorm:
            cnn = tf.layers.batch_normalization(cnn, axis=[-1], training=training_ph)

        cnn = tf.nn.relu(cnn)

        if i in maxPoolingPosition:
            print("Add pooling at position:", i)

            cnn = tf.nn.max_pool(cnn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

        print("Conv layer:", cnn.get_shape())

    return cnn, ws, bs

def calculateAccuracy(sess, cnn, imgs, gd, printAcc=False, batchSize = 2000):
    numBatchs = int(np.ceil(imgs.shape[0] / batchSize))
    # print(numBatchs)

    numCorrectAll_1a = 0
    numCorrectAll = 0
    for iBatch in range(numBatchs):
        gdData_1a = gd[iBatch * batchSize:(iBatch + 1) * batchSize, 0]

        # gd = gd.astype(np.int32)
        feed = {
            cnn.imgs_ph: imgs[iBatch * batchSize:(iBatch + 1) * batchSize, :, :, :],
            cnn.pkeep_ph: 1,
        }
        sfmx1a = sess.run(cnn.softmax, feed_dict=feed)
        predict1a = np.argmax(sfmx1a, axis=1)

        if printAcc:
           numCorrectAll_1a = numCorrectAll_1a + np.count_nonzero(predict1a == gdData_1a)

        predict1Sign = predict1a == gdData_1a

        numCorrect = np.count_nonzero(predict1Sign)
        numCorrectAll = numCorrectAll + numCorrect
        # print(numCorrectAll)

    if printAcc:
        print("predict1a accuracy: ", numCorrectAll_1a / gd.shape[0])

    accuracy = numCorrectAll / gd.shape[0]
    return accuracy

def getFullyConnectedLayers(layterSizes, inputLayer, inputLayerDim, pkeep_ph, netName=''):
    for iL, denseLSize in enumerate(layterSizes):
        bd = tf.get_variable("b%s_%d" % (netName, iL), shape=[denseLSize], initializer=tf.initializers.zeros())

        if iL:
            wd = tf.get_variable("W%s_%d" % (netName, iL), shape=[layterSizes[iL - 1], denseLSize],
                                  initializer=tf.initializers.glorot_normal())
            flat = tf.nn.dropout(tf.nn.relu(tf.matmul(flat, wd) + bd), pkeep_ph)
        else:
            wd = tf.get_variable("W%s_%d" % (netName, iL), shape=[inputLayerDim, denseLSize],
                                  initializer=tf.initializers.glorot_normal())
            flat = tf.nn.dropout(tf.nn.relu(tf.matmul(inputLayer, wd) + bd), pkeep_ph)

    return flat


class CNNCfg:
    def __init__(self):
        self.inputW = 75
        self.inputH = 75
        self.inputChannel = 1

        self.paramSetName = 'Param1'
        self.convSizes = [3, 3, 3, 3, 3, 3, 3]
        self.convChannels = [32, 32, 64, 64, 128, 128, 256]

        self.maxPoolingPosition = [1, 3, 5]

        self.fullConnectionLayerHead1Sizes = [1000, 500, ]
        self.fullConnectionLayerHead2Sizes = self.fullConnectionLayerHead1Sizes

        # self.paramSetName = 'ParamSet2'
        # self.convSizes = [5,5,5]
        # self.convChannels = [16,32,64]
        #
        # self.maxPoolingPosition = [0,1,2]
        #
        # self.fullConnectionLayerHead1Sizes = [500, 200,]
        # self.fullConnectionLayerHead2Sizes = self.fullConnectionLayerHead1Sizes

        # self.paramSetName = 'ParamSet3'
        # self.convSizes = [5,5,5]
        # self.convChannels = [4,8,8]
        #
        # self.maxPoolingPosition = [0,1,2]
        #
        # self.fullConnectionLayerHead1Sizes = [100, 100, 50]
        # self.fullConnectionLayerHead2Sizes = self.fullConnectionLayerHead1Sizes
        # #
        #
        # # DoubleChannels: Epoch: 100 Train accuracy: 0.999934 Test accuracy: 0.970000 Loss on Train batch: 0.0366 time 64.70
        # self.convSizes = [3,3,3,3,3,3,3]
        # self.convChannels = [64,64,128,128,256,256,512]
        #
        # self.maxPoolingPosition = [1,3,5]
        #
        # self.fullConnectionLayerHead1Sizes = [1000, 500,]
        # self.fullConnectionLayerHead2Sizes = self.fullConnectionLayerHead1Sizes
        #
        self.outputSize = 26

        self.lrDecay = 0.995
        self.lrDecayStep = 100
        self.learningRate = 0.001



# class CNN:
#     def __init__(self, cfg=CNNCfg()):
#         tf.reset_default_graph()
#         self.cfg = cfg
#         self.imgs_ph = tf.placeholder(tf.float32, [None, self.cfg.inputW, self.cfg.inputH, 1], name="imgs_ph")
#         self.output_ph_1a = tf.placeholder(tf.float32, [None, self.cfg.outputSize], name="output_ph_1a")
#         self.output_ph_1b = tf.placeholder(tf.float32, [None, self.cfg.outputSize], name="output_ph_1b")
#         self.output_ph_2a = tf.placeholder(tf.float32, [None, self.cfg.outputSize], name="output_ph_2a")
#         self.output_ph_2b = tf.placeholder(tf.float32, [None, self.cfg.outputSize], name="output_ph_2b")
#         self.pkeep_ph = tf.placeholder(tf.float32, name="pkeep_ph")
#         self.learnrate_ph = tf.placeholder(tf.float32, name="learnrate_ph")
#         self.training_ph = tf.placeholder(tf.bool, name="training_ph")
#
#         self.lrDecayRate = tf.placeholder(tf.float32, name="lrDecayRate")
#         self.lrDecayStep = tf.placeholder(tf.int32, name="lrDecayStep")
#
#     def getCNN(self):
#         convLayer, ws, bs = getConvLayers(self.imgs_ph, self.cfg.inputChannel, self.cfg.convSizes,
#                                           self.cfg.convChannels, self.cfg.maxPoolingPosition)
#
#         print("Last conv layer:", convLayer.get_shape())
#         flattenSize = getFlattenSize(convLayer)
#         print("Last conv layer flattern:", flattenSize)
#         flat = tf.reshape(convLayer, [-1, flattenSize])
#
#         flat1a = flat
#         flat1b = flat
#         # output for head 1
#         if len(self.cfg.fullConnectionLayerHead1Sizes):
#
#             for iL, denseLSize in enumerate(self.cfg.fullConnectionLayerHead1Sizes):
#                 if iL:
#                     wda = tf.get_variable("WDHead1a%d" % iL,
#                                           shape=[self.cfg.fullConnectionLayerHead1Sizes[iL - 1], denseLSize],
#                                           initializer=tf.initializers.glorot_normal())
#                     wdb = tf.get_variable("WDHead1b%d" % iL,
#                                           shape=[self.cfg.fullConnectionLayerHead1Sizes[iL - 1], denseLSize],
#                                           initializer=tf.initializers.glorot_normal())
#                 else:
#
#                     wda = tf.get_variable("WDHead1a%d" % iL, shape=[flattenSize, denseLSize],
#                                           initializer=tf.initializers.glorot_normal())
#                     wdb = tf.get_variable("WDHead1b%d" % iL, shape=[flattenSize, denseLSize],
#                                           initializer=tf.initializers.glorot_normal())
#                 bda = tf.get_variable("bDHead1a%d" % iL, shape=[denseLSize], initializer=tf.initializers.zeros())
#                 bdb = tf.get_variable("bDHead1b%d" % iL, shape=[denseLSize], initializer=tf.initializers.zeros())
#
#                 flat1a = tf.nn.dropout(tf.nn.relu(tf.matmul(flat1a, wda) + bda), self.pkeep_ph)
#                 flat1b = tf.nn.dropout(tf.nn.relu(tf.matmul(flat1b, wdb) + bdb), self.pkeep_ph)
#
#         flat1aSize = flat1a.get_shape()[-1]
#         flat1bSize = flat1b.get_shape()[-1]
#
#         wOut1a = tf.get_variable("w1Outa", shape=[flat1aSize, self.cfg.outputSize],
#                                  initializer=tf.initializers.glorot_normal())
#         bOut1a = tf.get_variable("b1Outa", shape=[self.cfg.outputSize], initializer=tf.initializers.zeros())
#
#         wOut1b = tf.get_variable("w1Outb", shape=[flat1bSize, self.cfg.outputSize],
#                                  initializer=tf.initializers.glorot_normal())
#         bOut1b = tf.get_variable("b1Outb", shape=[self.cfg.outputSize], initializer=tf.initializers.zeros())
#
#         denseOut1a = tf.nn.relu(tf.matmul(flat1a, wOut1a) + bOut1a)
#         denseOut1b = tf.nn.relu(tf.matmul(flat1b, wOut1b) + bOut1b)
#
#         # output for head 2
#         flat2a = flat
#         flat2b = flat
#         if len(self.cfg.fullConnectionLayerHead2Sizes):
#
#             for iL, denseLSize in enumerate(self.cfg.fullConnectionLayerHead2Sizes):
#                 if iL:
#                     wda = tf.get_variable("WDHead2a%d" % iL,
#                                           shape=[self.cfg.fullConnectionLayerHead2Sizes[iL - 1], denseLSize],
#                                           initializer=tf.initializers.glorot_normal())
#                     wdb = tf.get_variable("WDHead2b%d" % iL,
#                                           shape=[self.cfg.fullConnectionLayerHead2Sizes[iL - 1], denseLSize],
#                                           initializer=tf.initializers.glorot_normal())
#                 else:
#                     wda = tf.get_variable("WDHead2a%d" % iL, shape=[flattenSize, denseLSize],
#                                           initializer=tf.initializers.glorot_normal())
#                     wdb = tf.get_variable("WDHead2b%d" % iL, shape=[flattenSize, denseLSize],
#                                           initializer=tf.initializers.glorot_normal())
#                 bda = tf.get_variable("bDHead2a%d" % iL, shape=[denseLSize], initializer=tf.initializers.zeros())
#                 bdb = tf.get_variable("bDHead2b%d" % iL, shape=[denseLSize], initializer=tf.initializers.zeros())
#
#                 flat2a = tf.nn.dropout(tf.nn.relu(tf.matmul(flat2a, wda) + bda), self.pkeep_ph)
#                 flat2b = tf.nn.dropout(tf.nn.relu(tf.matmul(flat2b, wdb) + bdb), self.pkeep_ph)
#
#         flat2aSize = flat2a.get_shape()[-1]
#         flat2bSize = flat2b.get_shape()[-1]
#
#         wOut2a = tf.get_variable("w2Outa", shape=[flat2aSize, self.cfg.outputSize],
#                                  initializer=tf.initializers.glorot_normal())
#         bOut2a = tf.get_variable("b2Outa", shape=[self.cfg.outputSize], initializer=tf.initializers.zeros())
#
#         wOut2b = tf.get_variable("w2Outb", shape=[flat2bSize, self.cfg.outputSize],
#                                  initializer=tf.initializers.glorot_normal())
#         bOut2b = tf.get_variable("b2Outb", shape=[self.cfg.outputSize], initializer=tf.initializers.zeros())
#
#         denseOut2a = tf.nn.relu(tf.matmul(flat2a, wOut2a) + bOut2a)
#         denseOut2b = tf.nn.relu(tf.matmul(flat2b, wOut2b) + bOut2b)
#
#         # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.output_ph, logits = denseOut))
#         # softmax = tf.nn.softmax(denseOut)
#         self.softmax1a = tf.nn.softmax(denseOut1a)
#         self.softmax1b = tf.nn.softmax(denseOut1b)
#         self.softmax2a = tf.nn.softmax(denseOut2a)
#         self.softmax2b = tf.nn.softmax(denseOut2b)
#
#         cross_entropy = (tf.reduce_mean(
#             tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_ph_1a, logits=denseOut1a)) \
#                          + tf.reduce_mean(
#                     tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_ph_1b, logits=denseOut1b)) \
#                          + tf.reduce_mean(
#                     tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_ph_2a, logits=denseOut2a)) \
#                          + tf.reduce_mean(
#                     tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_ph_2b, logits=denseOut2b))) / 4
#
#         return cross_entropy
#
#     def getTrainableModel(self):
#         self.cross_entropy = self.getCNN()
#
#         step = tf.Variable(0, trainable=False)
#         rate = tf.train.exponential_decay(self.learnrate_ph, step, self.lrDecayStep, self.lrDecayRate)
#
#         train_step = tf.train.AdamOptimizer(learning_rate=rate).minimize(self.cross_entropy, global_step=step)
#
#         return train_step