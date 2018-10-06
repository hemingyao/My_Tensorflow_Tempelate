from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from six.moves import xrange
import os, json, pickle, glob, sys
import numpy as np
import tensorflow as tf
import dataset_utils


CLASS = [0, 1]
PRE = ''
DATA_DIR = '/media/DensoML/DENSO ML/tfrecord/'

raw_data_path = '/home/spc/Dropbox/mnist_test_seq.npy'


def get_data(raw_data_path, set_id):
    
    save_path = os.path.join(DATA_DIR, set_id)

    if os.path.isdir(save_path):
        pass
        #sys.exit('{} is already exist'.format(save_path))
    else:
        os.makedirs(save_path)

    data = np.load(raw_data_path)
    for pf in range(0,5):
        tfrecord_filename = os.path.join(save_path, PRE+str(pf)+'.tfrecord')
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        for index in range (0, 2000):
            data_point = data[0:10,index+pf*2000,:,:]
            data_point = data_point.astype(np.uint8)
            data_point = data_point.tostring()

            label = data[10:20,index+pf*2000,:,:]
            label = label.astype(np.uint8)
            label = label.tostring()

            example = dataset_utils.image_to_tfexample_segmentation(data_point, label)
            tfrecord_writer.write(example.SerializeToString())


if __name__ == '__main__':
    get_data(raw_data_path, 'moving_mnist')

