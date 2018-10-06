import os
import tensorflow as tf
from six.moves import xrange
from flags import FLAGS, IMG_SIZE, DATA_AUG
from data_aug import *


class DataSet(object):
    def __init__(self, data_dir, data_range, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion
        self.data_range = data_range

    def get_filenames(self):
        """
        Customized for different dataset
        
        #if self.subset in ['train', 'validation', 'test']:
        """
        return [os.path.join(self.data_dir, '{}.tfrecord'.format(str(i)))
                    for i in self.data_range]            
        

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        if FLAGS.seg:
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/encoded': tf.FixedLenFeature([], tf.string),
                    'image/class/label': tf.FixedLenFeature([], tf.string),
                    'image/subject': tf.FixedLenFeature([], tf.int64),
                    'image/index': tf.FixedLenFeature([], tf.int64)
                })

            image = tf.decode_raw(features['image/encoded'], out_type=tf.float64)
            image = tf.reshape(image, IMG_SIZE)
            image = tf.cast(image, tf.float32)

            label = tf.decode_raw(features['image/class/label'], out_type=tf.int8)
            #label = tf.reshape(tf.cast(label, tf.float32), IMG_SIZE)
            #label = tf.concat([label, label, label], axis=-1)
            #label = tf.equal(tf.cast(label, tf.float32), tf.ones([256*256]))

            #label = tf.one_hot(tf.reshape(tf.cast(label, tf.int32), IMG_SIZE[0:2]), FLAGS.num_labels)

            label = tf.cast(label, tf.float32)
            label = tf.reshape(label, [256,256,3])
            #label = tf.cast(tf.reshape(label, IMG_SIZE), tf.float32)

            imgall = tf.concat([image, label], -1)
            cut = image.shape[-1].value

            new = self.preprocess(imgall)

            image = new[:,:,0:cut]
            #image = tf.image.per_image_standardization(image)

            label = new[:,:,cut:]
            #label = tf.cast(label, np.int32)
            
            # Custom preprocessing.

            image.set_shape(IMG_SIZE)
            label.set_shape(IMG_SIZE[0:2]+(FLAGS.num_labels,))

        else:
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/encoded': tf.FixedLenFeature([], tf.string),
                    'image/class/label': tf.FixedLenFeature([], tf.int64),
                    'image/subject': tf.FixedLenFeature([], tf.int64),
                    'image/index': tf.FixedLenFeature([], tf.int64)
                })

            #image = tf.image.decode_png(features['image/encoded'], dtype=tf.uint8)
            image = tf.decode_raw(features['image/encoded'], out_type=tf.uint8)
            image = tf.reshape(image, IMG_SIZE)
            image = tf.cast(image, tf.float32)
            #image = image/255 - np.stack([np.ones([32,32])*0.4913, np.ones([32,32])*0.4822, np.ones([32,32])*0.4465], axis=-1)
            image = self.preprocess(image)
            #label = tf.cast(features['image/class/label'], tf.int32)
            label = features['image/class/label']

            label = tf.one_hot(label,FLAGS.num_labels)

            image.set_shape(IMG_SIZE)
            label.set_shape((FLAGS.num_labels,))

        return image, label, features['image/subject'], features['image/index']

    def make_batch(self, batch_size):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames).repeat()

        # Parse records.
        dataset = dataset.map(
            self.parser, num_parallel_calls=4)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = int(
                DataSet.num_examples_per_epoch(self.subset) * FLAGS.min_fraction_of_examples_in_queue)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        # Batch it up.
        #dataset = dataset.prefetch(buffer_size=2 * batch_size)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch, subejct_id, ind = iterator.get_next()

        return image_batch, label_batch, subejct_id, ind 


    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            if 'random_crop' in DATA_AUG:
                image = random_crop(image, crop_padding_height=40, crop_padding_width=40)
            if 'random_shift' in DATA_AUG:
                image = random_shift(image, max_h=30, max_w=30)
            if 'random_flip_left_right' in DATA_AUG:
                image = random_flip_left_right(image)
            if 'random_rotation' in DATA_AUG:
                image = random_rotation(image)
            if 'random_resize' in DATA_AUG:
                image = random_resize(image, max_r=280)
        return image

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return FLAGS.num_train_images
        elif subset == 'validation':
            return FLAGS.num_test_images
        elif subset == 'eval':
            return FLAGS.num_test_images
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


def input_fn(data_dir,
             subset,
             data_range,
             num_shards,
             batch_size,
             use_distortion_for_training=True):
    """Create input graph for model.
    Args:
      data_dir: Directory where TFRecords representing the dataset are located.
      subset: one of 'train', 'validate' and 'eval'.
      num_shards: num of towers participating in data-parallel training.
      batch_size: total batch size for training to be divided by the number of
      shards.
      use_distortion_for_training: True to use distortions.
    Returns:
      two lists of tensors for features and labels, each of num_shards length.
    """
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train' and use_distortion_for_training
        dataset = DataSet(data_dir, data_range, subset, use_distortion)
        image_batch, label_batch, subject_id_batch, ind_batch = dataset.make_batch(batch_size)
        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [image_batch], [label_batch], [subject_id_batch], [ind_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        subject_id_batch = tf.unstack(subject_id_batch, num=batch_size, axis=0)
        ind_batch =  tf.unstack(ind_batch , num=batch_size, axis=0)

        feature_shards = [[] for i in range(num_shards)]
        label_shards = [[] for i in range(num_shards)]
        subject_id_shards = [[] for i in range(num_shards)]
        ind_shards = [[] for i in range(num_shards)]

        for i in xrange(batch_size):
            idx = i % num_shards
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
            subject_id_shards[idx].append(subject_id_batch)
            ind_shards[idx].append(ind_batch[i])

        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        subject_id_shards = [tf.parallel_stack(x) for x in subject_id_shards]
        ind_shards = [tf.parallel_stack(x) for x in ind_shards]
        return feature_shards, label_shards, subject_id_shards, ind_shards


