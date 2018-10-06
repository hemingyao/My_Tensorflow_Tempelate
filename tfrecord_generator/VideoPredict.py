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
#DATA_DIR = '/home/spc/Documents/TFrecord'
DATA_DIR = '/media/DensoML/DENSO ML/tfrecord'

raw_data_path = '/media/DensoML/DENSO ML/DrowsinessData/raw_data_all_128'
train = ['001', '002', '005', '006', '008', '009','012', '013', '015', '020', '023',\
     '024','031', '032', '033', '034', '035', '036']
seq_length = 10
stride_frame = 5
stride_seq = 7
set_id = 'drowsiness_video'


def get_data_pickle(raw_data_path, set_id, seq_length, stride_frame, stride_seq, subject_index=[]):
    
    save_path = os.path.join(DATA_DIR, set_id+'_{}_{}_{}'.format(seq_length, stride_frame, stride_seq))

    if os.path.isdir(save_path):
        pass
        #sys.exit('{} is already exist'.format(save_path))
    else:
        os.makedirs(save_path)

    log_f = open(os.path.join(DATA_DIR, set_id+'_{}_{}_{}'.format(seq_length, stride_frame, stride_seq)+'_info'), 'a')

    for idx, pf in enumerate(subject_index):
        tfrecord_filename = os.path.join(save_path, PRE+str(idx)+'.tfrecord')
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        total = 0
        files = glob.glob(os.path.join(raw_data_path, '*'+pf+'*'))

        for each_file in files:
            fname = each_file.split('/')[-1]
            with open(each_file, 'rb') as rf:
                [imgs, annots, eye, head, mouth]= pickle.load(rf)
                total += imgs.shape[0]
                for index in range (0,len(imgs)-(seq_length+1)*stride_frame,stride_seq):
                    data_point = imgs[index:index+seq_length*stride_frame:stride_frame,:,:]
                    label = imgs[index+seq_length*stride_frame,:,:]

                    data_point = data_point.astype(np.uint8)
                    label = label.astype(np.uint8)
                    data_point = data_point.tostring()
                    label = label.tostring()
                    example = dataset_utils.image_to_tfexample_segmentation(data_point, label)
                    tfrecord_writer.write(example.SerializeToString())
            rf.close()
            print('Finish extracting data from %s'%(each_file))

        print('Finish writing data from {}'.format(pf))
        log_f.write('{}: {}\n'.format(pf, total))

    #log_f.write('{}'.format(json.dumps(dataset_stats)))


if __name__ == '__main__':
    get_data_pickle(raw_data_path, set_id, seq_length=seq_length, \
        stride_frame=stride_frame, stride_seq=stride_seq, subject_index=train)

