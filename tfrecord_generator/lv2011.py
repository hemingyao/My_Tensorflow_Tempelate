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
    For label: a mask (Note: The mask is 0/1)
""" 
# allinone: Designed for one big matlab file for all data
# individual: Designed for multiple matlab files, each for one subject [TODO]

# Output: multiple tfrecord files, each for one subject

CLASS = [0, 1]
PRE = ''

DATA_DIR = '/media/DensoML/DENSO ML/tfrecord'
raw_data_path = '/media/DensoML/DENSO ML/LVData/LV_256_mix_eq2_vali.h5' #LV_256_mix_eq_sel.h5, LV_256_mix_eq2_vali.h5
set_id = 'LV2011train'



def get_data_LV2011(raw_data_path, set_id):
    f = h5py.File(raw_data_path, 'r')
    save_path = os.path.join(DATA_DIR, set_id)

    if os.path.isdir(save_path):
        pass
        #sys.exit('{} is already exist'.format(save_path))
    else:
        os.makedirs(save_path)

    log_f = open(os.path.join(DATA_DIR, set_id+'_info'), 'a')
    total = 0
    
    subject_index = list(f['location'].keys())

    #subject_index = subject_index[57:77]
    for pid, pf in enumerate(subject_index):
        tfrecord_filename = os.path.join(save_path, PRE+str(100+pid)+'.tfrecord')
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        refinput = f['input/{}'.format(pf)][:]
        refoutput = f['label/{}'.format(pf)][:].astype(np.bool).astype(np.int8)

        num = f['input/{}'.format(pf)].attrs['Num'].astype(np.int)[0]
        print(num)
        total+=num

        for ind in range(num):
            Input= refinput[ind,:,:,0]
            #Input = Input.astype(np.uint8)
            #print(np.max(Input))
            Label= refoutput[ind,:,:,:]
            Label = Label>0.5
            #print(np.max(Label))
            
            #w = Label[:,:,1]
            #i = Label[:,:,2]
            #print(np.sum(w))

            #Label = np.zeros(w.shape) + w + i*2
            
            #print(np.sum(w==1))
            Label = Label.astype(np.int8)

            data_point = Input.tostring()
            label = Label.tostring()

            example = dataset_utils.image_to_tfexample_segmentation(data_point, label, subject_id=pid, index=ind)
            tfrecord_writer.write(example.SerializeToString())

        print('Finish writing data from {}'.format(pid))
        log_f.write('{}_{}: {}\n'.format(pid, pf, num))

    log_f.write('In total: {}'.format(total))



def get_data_Sunny(raw_data_path, set_id):
    f = h5py.File(raw_data_path, 'r')
    save_path = os.path.join(DATA_DIR, set_id)

    if os.path.isdir(save_path):
        pass
        #sys.exit('{} is already exist'.format(save_path))
    else:
        os.makedirs(save_path)

    log_f = open(os.path.join(DATA_DIR, set_id+'_info'), 'a')
    total = 0
    
    for prex, dataset in enumerate(['train', 'val', 'test']):
        subject_index = list(f['{}/location'.format(dataset)].keys())
        for pid, pf in enumerate(subject_index):
            tfrecord_filename = os.path.join(save_path, PRE+str(pid+prex*15)+'.tfrecord')
            tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)

            refinput = f['{}/input/{}'.format(dataset, pf)][:]
            refoutput = f['{}/label/{}'.format(dataset, pf)][:].astype(np.bool).astype(np.int8)
            num = f['{}/input/{}'.format(dataset, pf)].attrs['Num'].astype(np.int)
            num = num[0]
            print(num)
            total+=num

            for ind in range(num):
                Input= refinput[ind,:,:]
                Label= refoutput[ind,:,:,:]

                wall = Label[:,:,1]
                endo = Label[:,:,2]

                Label = np.zeros(wall.shape)
                Label = Label + wall + endo*2
                Label = Label.astype(np.int8)
                
                data_point = Input.tostring()
                label = Label.tostring()

                example = dataset_utils.image_to_tfexample_segmentation(data_point, label, subject_id=pid, index=ind)
                tfrecord_writer.write(example.SerializeToString())

            print('Finish writing data from {}'.format(pid))
            log_f.write('{}_{}: {}\n'.format(pid, pf, num))

    log_f.write('In total: {}'.format(total))


if __name__ == '__main__':
    #raw_data_path = '/media/DensoML/DENSO ML/DrowsinessData/'
    get_data_LV2011(raw_data_path, set_id)
    #get_data_Sunny(raw_data_path, set_id)
