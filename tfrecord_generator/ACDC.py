#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:17:55 2018

@author: HemingY
"""
import os
import nibabel as nib
import configparser, itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import dataset_utils


#DATA_DIR = '/Volumes/med-kayvan-lab/Users/hemingy/Left Ventricle/Data'
#raw_data_path = '/Volumes/med-kayvan-lab/Users/hemingy/Left Ventricle/RawDataset/ACDC/training'
#raw_data_path = 'Z:/Users/hemingy/Left Ventricle/RawDataset/ACDC/training/'
#DATA_DIR = 'Z:/Users/hemingy/Left Ventricle/Data'
raw_data_path = '/media/DensoML/DENSO ML/LVData/ACDC/training'
DATA_DIR = '/media/DensoML/DENSO ML/tfrecord/'
set_id = 'ACDC'
    

def view_result(data, label):
    num = data.shape[0]
    ncol = 2
    nrow = num//ncol+1
    """
    for i in range(num):
        conimg = np.concatenate([data[i], label[i,:,:,1]*255], axis=-1)
        plt.subplot(nrow, ncol, i+1); plt.imshow(conimg, figsize=(15,15))
    """
    f = plt.figure(figsize=(10,10))
    for i in range(num):
        conimg = np.concatenate([data[i], label[i,:,:,1]*255], axis=-1)
        ax = f.add_subplot(nrow, ncol, i+1)#'{}{}{}'.format(nrow, ncol, j)
        ax.imshow(conimg)


def center_crop(ndarray, crop_size):
    '''Input ndarray is of rank 3 (height, width, depth).

    Argument crop_size is an integer for square cropping only.

    Performs padding and center cropping to a specified size.
    '''
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')
    if len(ndarray.shape)==3:
        n, h, w = ndarray.shape
        if any([dim < crop_size for dim in (h, w)]):
            # zero pad along each (h, w) dimension before center cropping
            pad_h = (crop_size - h) if (h < crop_size) else 0
            pad_w = (crop_size - w) if (w < crop_size) else 0
            rem_h = pad_h % 2
            rem_w = pad_w % 2
            pad_dim_h = (pad_h//2, pad_h//2 + rem_h)
            pad_dim_w = (pad_w//2, pad_w//2 + rem_w)
            # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
            npad = ((0,0), pad_dim_h, pad_dim_w)
            ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
            n, h, w = ndarray.shape
        # center crop
        h_offset = (h - crop_size) // 2
        w_offset = (w - crop_size) // 2
    
        cropped = ndarray[:, h_offset:(h_offset+crop_size),
                          w_offset:(w_offset+crop_size)]
        return cropped
    
    elif len(ndarray.shape)==4:
        n, h, w, d = ndarray.shape
        if any([dim < crop_size for dim in (h, w)]):
            # zero pad along each (h, w) dimension before center cropping
            pad_h = (crop_size - h) if (h < crop_size) else 0
            pad_w = (crop_size - w) if (w < crop_size) else 0
            rem_h = pad_h % 2
            rem_w = pad_w % 2
            pad_dim_h = (pad_h//2, pad_h//2 + rem_h)
            pad_dim_w = (pad_w//2, pad_w//2 + rem_w)
            # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
            npad = ((0,0), pad_dim_h, pad_dim_w, (0,0))
            ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
            n, h, w, d = ndarray.shape
        # center crop
        h_offset = (h - crop_size) // 2
        w_offset = (w - crop_size) // 2
    
        cropped = ndarray[:, h_offset:(h_offset+crop_size),
                          w_offset:(w_offset+crop_size),:]
        return cropped        
        
    
#img_new = imresize(img, float(pp[0]))
def test_img(path):
    #path = '/Users/HemingY/Developer/LeftVentricle/'
    pids = os.listdir(path)
    pids = [x for x in pids if 'patient' in x]
    pids = sorted(pids, key=lambda x: int(x.split('patient')[-1]))
    for pf, pid in enumerate(pids):
        pidpath = os.path.join(path, pid)
        
        cfg = configparser.ConfigParser()
        filename = os.path.join(pidpath, 'Info.cfg')
        with open(filename) as fp:
          cfg.read_file(itertools.chain(['[global]'], fp), source=filename)
        secs = dict(cfg.items('global'))
        
        # For ED
        ed = int(secs['ed'])
        # Read imgs
        name = '{0}_frame{1:02d}.nii.gz'.format(pid, ed)
        nimg = nib.load(os.path.join(pidpath, name))
        zooms = nimg.header.get_zooms()
        
        #print(pf, zooms[0])
        
        data_ed = nimg.get_data()
        if np.max(data_ed)<=255:
            data_ed = data_ed.astype(np.uint8)
        print(np.max(data_ed))
        """
        data_ed = np.transpose(data_ed, (2,0,1))
        orishape = data_ed.shape
        data_ed_new = resize(data_ed, order=1, mode='reflect', anti_aliasing=False,
            output_shape=(orishape[0], orishape[1]*zooms[0], orishape[2]*zooms[1]))
        data_ed_crop = center_crop(data_ed_new, 256)

        name = '{0}_frame{1:02d}_gt.nii.gz'.format(pid, ed)
        nimg = nib.load(os.path.join(pidpath, name))
        label_ed = nimg.get_data()
        label_ed = np.transpose(label_ed, (2,0,1))
         
        label_ed = np.stack([np.logical_or(label_ed==0,label_ed==1), label_ed==2, label_ed==3], axis=-1)
        
        orishape = label_ed.shape
        label_ed_new = resize(label_ed, order=1,  mode='reflect', anti_aliasing=True, anti_aliasing_sigma=0.1,
            output_shape=(orishape[0], orishape[1]*zooms[0], orishape[2]*zooms[1], orishape[3]))
        label_ed_new[label_ed_new<0.5] = 0
        label_ed_new[label_ed_new>0.5] = 1
        label_ed_crop = center_crop(label_ed_new, 256)
        
    return data_ed_crop, label_ed_crop
        """

def read_imgs_labels(pidpath, name, name_gt):
    nimg = nib.load(os.path.join(pidpath, name))
    zooms = nimg.header.get_zooms()
    data = nimg.get_data()
    data = np.transpose(data, (2,0,1))
    orishape = data.shape
    if np.max(data)<=255:
        data = data.astype(np.uint8)
        
    data_new = resize(data, order=1, mode='reflect', anti_aliasing=True,
        output_shape=(orishape[0], orishape[1]*zooms[0], orishape[2]*zooms[1]))
    data_crop = center_crop(data_new, 256)

    
    nimg = nib.load(os.path.join(pidpath, name))
    label = nimg.get_data()
    label = np.transpose(label, (2,0,1))
     
    label = np.stack([np.logical_or(label==0,label==1), label==2, label==3], axis=-1)
    
    orishape = label.shape
    label_new = resize(label, order=1, anti_aliasing=True, anti_aliasing_sigma=0.1,
        output_shape=(orishape[0], orishape[1]*zooms[0], orishape[2]*zooms[1], orishape[3]))
    label_new[label_new<0.5] = 0
    label_new[label_new>0.5] = 1
    label_crop = center_crop(label_new, 256)
    return data_crop, label_crop

        
def read_ACDC(path, set_id):
    #path = '/Users/HemingY/Developer/LeftVentricle/'
    pids = os.listdir(path)
    pids = [x for x in pids if 'patient' in x]
    pids = sorted(pids, key=lambda x: int(x.split('patient')[-1]))
    save_path = os.path.join(DATA_DIR, set_id)
    if os.path.isdir(save_path):
        pass
        #sys.exit('{} is already exist'.format(save_path))
    else:
        os.makedirs(save_path)
        
    log_f = open(os.path.join(DATA_DIR, set_id+'_info'), 'a')
    total_es = 0
    total_ed = 0
            
    for pf, pid in enumerate(pids):
        pidpath = os.path.join(path, pid)
        
        cfg = configparser.ConfigParser()
        filename = os.path.join(pidpath, 'Info.cfg')
        with open(filename) as fp:
          cfg.read_file(itertools.chain(['[global]'], fp), source=filename)
        secs = dict(cfg.items('global'))
        
        # For ED
        ed = int(secs['ed'])
        name = '{0}_frame{1:02d}.nii.gz'.format(pid, ed)
        name_gt = '{0}_frame{1:02d}_gt.nii.gz'.format(pid, ed)
        data_ed, label_ed = read_imgs_labels(pidpath, name, name_gt)

        # For ES
        es = int(secs['es'])
        name = '{0}_frame{1:02d}.nii.gz'.format(pid, es)
        name_gt = '{0}_frame{1:02d}_gt.nii.gz'.format(pid, es)
        data_es, label_es = read_imgs_labels(pidpath, name, name_gt)

        # Write tfrecords
        tfrecord_filename = os.path.join(save_path, str(pf)+'_es.tfrecord')
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)
        data_es = data_es/np.max(data_es)
        for ind in range(label_es.shape[0]):
            Label = label_es[ind]
            Label = Label.astype(np.int8)
            Input = data_es[ind]
            data_point = Input.tostring()
            label = Label.tostring()

            example = dataset_utils.image_to_tfexample_segmentation(data_point, label, subject_id=pf, index=ind+100)
            tfrecord_writer.write(example.SerializeToString())
            
        tfrecord_filename = os.path.join(save_path, str(pf)+'_ed.tfrecord')
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)
        data_ed = data_ed/np.max(data_ed)
        for ind in range(label_ed.shape[0]):
            Label = label_ed[ind]
            Label = Label.astype(np.int8)
            Input = data_ed[ind]
            data_point = Input.tostring()
            label = Label.tostring()

            example = dataset_utils.image_to_tfexample_segmentation(data_point, label, subject_id=pf, index=ind)
            tfrecord_writer.write(example.SerializeToString())
        
        total_es += label_es.shape[0]
        total_ed += label_ed.shape[0]
        
        print('Finish writing data from {}'.format(pid))
        log_f.write('{}: es: {}; ed: {}\n'.format(pid, label_es.shape[0], label_ed.shape[0]))
    
    log_f.write('In total: es: {}; ed: {}'.format(total_es, total_ed))
    
    
if __name__ == '__main__':
    read_ACDC(raw_data_path, set_id)
    #test_img(raw_data_path)