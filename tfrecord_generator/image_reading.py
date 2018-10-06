from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os, json, pickle, glob, sys
import numpy as np
import tensorflow as tf
import dataset_utils
import cv2


set_id = 'ACDCclass_again'
ROOT = '/home/spc/Documents/ACDC/Train_again'
PRE = ''
DATA_DIR = '/media/DensoML/DENSO ML/tfrecord'


def get_data_images(ROOT, set_id):
    save_path = os.path.join(DATA_DIR, set_id)
    if os.path.isdir(save_path):
        pass
        #sys.exit('{} is already exist'.format(save_path))
    else:
        os.makedirs(save_path)

    log_f = open(os.path.join(DATA_DIR, set_id+'_info'), 'a')

    pids = glob.glob(os.path.join(ROOT, '*'))

    pids = sorted(pids, key=lambda x: int(x.split('/')[-1]))
    #pids = pids[131:]
    for i, pname in enumerate(pids):    
        tfrecord_filename = os.path.join(save_path, PRE+str(i)+'.tfrecord')
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)
        
        # read images
        print(pname)
        filenames = glob.glob(os.path.join(pname, '*'))
        #filenames = sorted(filenames, key=lambda x: int(x.split('/')[-1].split('_')[0]))
        for j, imgname in enumerate(filenames):
            #print(imgname)
            if imgname.endswith('.png'):
                img = cv2.imread(imgname)
                img = img[...,0]
                
                oriimg = img[:,0:256]
                label = img[:,256*1:256*2]
                label_wall = (label==255*np.ones(label.shape))
                label_endo = (label==127*np.ones(label.shape))
                pred = img[:,256*2:256*3]
                pred_wall = (pred==255*np.ones(pred.shape))
                pred_endo = (pred==127*np.ones(pred.shape))
                
                masks_wall = [label_wall, pred_wall]
                masks_endo = [label_endo, pred_endo]
                masks = [label, pred]
                for ind in range(2):
                    #Input = np.stack([oriimg*masks_wall[ind], oriimg*masks_endo[ind]], -1)
                    #Input = np.stack([oriimg, masks[ind]], -1)
                    Input = np.stack([oriimg, masks_wall[ind]*255, masks_endo[ind]*255], -1)
                    Input = Input.astype(np.int8)
                    label = ind
                    #label = random.randint(0,1)
                    data_point = Input.tostring()

                    example = dataset_utils.image_to_tfexample(data_point, label, subject_id=i, index=j)
                    tfrecord_writer.write(example.SerializeToString())
            
        print('Finish writing data from {}'.format(i))
        log_f.write('{}: {}\n'.format(i, len(imgname)))
        
        
if __name__ == '__main__':
    get_data_images(ROOT, set_id)