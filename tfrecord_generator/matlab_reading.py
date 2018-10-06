from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from six.moves import xrange
import os, json, pickle, glob, sys
import numpy as np
import tensorflow as tf
import dataset_utils
import h5py

"""
 h5 file structure (from Matlab):
    DATASET, NAME (input, label)  
    For label: a mask (Note: The mask is 0/1 or 0/255?)
""" 
# allinone: Designed for one big matlab file for all data
# individual: Designed for multiple matlab files, each for one subject [TODO]

# Output: multiple tfrecord files, each for one subject

CLASS = [0, 1]
PRE = ''
DATASET = 'PatientsData_sel'
NAME = 'adjustImgs_static'
MASK = 'masks'
DATA_RANGE = range(0,62)
DATA_DIR = '/home/spc/Documents/TFrecord'

raw_data_path = '/home/spc/Dropbox/CVproject/PatientsData_sel.mat'
set_id = 'Hematoma_v2'


def get_data_mat_allinone(raw_data_path, set_id, subject_index):
    f = h5py.File(raw_data_path, 'r')
    save_path = os.path.join(DATA_DIR, set_id)

    if os.path.isdir(save_path):
        pass
        #sys.exit('{} is already exist'.format(save_path))
    else:
        os.makedirs(save_path)

    log_f = open(os.path.join(DATA_DIR, set_id+'_info'), 'a')
    total = 0
    
    dataset_stats = {}
    for each_class in CLASS:
        dataset_stats.setdefault(each_class,0)
    
    for pf in subject_index:
        tfrecord_filename = os.path.join(save_path, PRE+str(pf)+'.tfrecord')
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        refinput = f[DATASET][NAME][pf]
        refoutput = f[DATASET][MASK][pf]
        num = f[refinput[0]].shape[0]
        print(num)
        total+=num

        subject_stats = {}
        for each_class in CLASS:
            subject_stats.setdefault(each_class,0)

        Labels = f[refoutput[0]][:]

        for ind in range(num):
            Input= f[refinput[0]][ind,:,:]
            Label= f[refoutput[0]][ind,:,:]/255    

            if np.sum(Label)>0:
                subject_stats[1] += 1
                dataset_stats[1] += 1
            else: 
                subject_stats[0] += 1
                dataset_stats[0] += 1

            Label = Label.astype(np.int8)
            
            data_point = Input.tostring()
            label = Label.tostring()

            example = dataset_utils.image_to_tfexample_segmentation(data_point, label, subject_id=pf, index=ind)
            tfrecord_writer.write(example.SerializeToString())

        print('Finish writing data from {}'.format(pf))
        log_f.write('{}: {}\t {}\n'.format(pf, num, json.dumps(subject_stats)))

    log_f.write('{}'.format(json.dumps(dataset_stats)))


if __name__ == '__main__':
    #raw_data_path = '/media/DensoML/DENSO ML/DrowsinessData/'
    get_data_mat_allinone(raw_data_path, set_id, subject_index=DATA_RANGE)

