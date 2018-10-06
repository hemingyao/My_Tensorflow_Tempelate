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
DATA_DIR = '/home/spc/Documents/TFrecord'

raw_data_path = '/media/DensoML/DENSO ML/DrowsinessData/raw_data_all_128'
train = ['001', '002', '005', '006', '008', '009','012', '013', '015', '020', '023',\
     '024','031', '032', '033', '034', '035', '036']
seq_length = 1
stride_frame = 5
stride_seq = 7


def get_data_pickle(raw_data_path, set_id, seq_length, stride_frame, stride_seq, subject_index=[]):
    
    save_path = os.path.join(DATA_DIR, set_id+'_{}_{}_{}'.format(seq_length, stride_frame, stride_seq))

    if os.path.isdir(save_path):
        pass
        #sys.exit('{} is already exist'.format(save_path))
    else:
        os.makedirs(save_path)

    log_f = open(os.path.join(DATA_DIR, set_id+'_{}_{}_{}'.format(seq_length, stride_frame, stride_seq)+'_info'), 'a')

    dataset_stats = {}
    for each_class in CLASS:
        dataset_stats.setdefault(each_class,0)

    for pf in subject_index:
        tfrecord_filename = os.path.join(save_path, PRE+pf+'.tfrecord')
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        subject_stats = {}
        for each_class in CLASS:
            subject_stats.setdefault(each_class,0)
        total = 0

        files = glob.glob(os.path.join(raw_data_path, '*'+pf+'*'))

        for each_file in files:
            fname = each_file.split('/')[-1]
            with open(each_file, 'rb') as rf:
                [imgs, annots, eye, head, mouth]= pickle.load(rf)
                for index in range (0,len(imgs)-seq_length*stride_frame,stride_seq):
                    # IF 2D input
                    if seq_length == 1:
                        data_point = imgs[index,:,:,:]
                        label = int(annots[index])
                        subject_stats[label] += 1 
                        dataset_stats[label] += 1 
                        total += 1
                    # IF 3D input
                    else:
                        data_point = imgs[index:index+seq_length*stride_frame:stride_frame,:,:]
                        # For drowsiness
                        annots = annots.astype(np.int32)
                        label_array = annots[index:index+seq_length*stride_frame:stride_frame]
                        #if int(label[-1])==1:
                        if sum(label_array)==seq_length:
                            label = 1
                        #elif int(label[-1])==0:
                        elif sum(label_array)==0:
                            label = 0
                        else:
                            continue
                        subject_stats[label] += 1 
                        dataset_stats[label] += 1 
                        total += 1

                    data_point = data_point.tostring()
                    example = dataset_utils.image_to_tfexample(data_point, label)
                    tfrecord_writer.write(example.SerializeToString())
            rf.close()
            print('Finish extracting data from %s'%(each_file))

        print('Finish writing data from {}'.format(pf))
        log_f.write('{}: {}\t {}\n'.format(pf, total, json.dumps(subject_stats)))

    log_f.write('{}'.format(json.dumps(dataset_stats)))


if __name__ == '__main__':
    get_data_pickle(raw_data_path, 'drowsiness', seq_length=1, stride_frame=5, stride_seq=7, subject_index=train)

