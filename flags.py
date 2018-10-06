import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
############################################################################
# Experimental Configuration
option = 1  # 1: training, 2: test
IMG_SIZE = (32, 32, 3) 
DATA_AUG = ['random_flip_left_right', 'crop'] # options: 'elastic_transform', 'random_crop', 'random_shift', 'random_flip_left_right', 'random_rotation', 'random_resize'

fold = 0  # should be from 0 to 5
"""
VAL_RANGE = set(range(fold, 39, 5))
TRAIN_RANGE = set(range(0, 39)) - VAL_RANGE
TEST_RANGE = set(range(fold, 39, 5))
"""
TEST_RANGE = set(range(0,1))
VAL_RANGE = set(range(0,1))
TRAIN_RANGE = set(range(1,2)) 

tf.app.flags.DEFINE_string('set_id', 'cifar_tfrecord', #Hematoma512_new, Edema
	"""Specify the dataset used for training and test""") 

tf.app.flags.DEFINE_string('run_name', 'test',
	"""Name of your experiement""")

tf.app.flags.DEFINE_integer('save_epoch', 10,#'model3_add_loss_more_2', 'model3_depthwise'， 'baseline_rescale'
	"""Save the model at every x epoches""")

tf.app.flags.DEFINE_integer('batch_size', 512,  #2
	"""Batch size""")

tf.app.flags.DEFINE_integer('test_batch_size', 256, 
	"""Batch size""")

tf.app.flags.DEFINE_integer('first_channel', 32,#'model3_add_loss_more_2', 'model3_depthwise'， 'baseline_rescale'
	"""First channel of the network""")
############################################################################
"""
Hyper parameters.
You need tuning them to find the best combination for your task
using cross validation
"""

tf.app.flags.DEFINE_float('pscale', 1,#'model3_add_loss_more_2', 'model3_depthwise'， 'baseline_rescale'
	"""Class Weight to the positive samples""")

tf.app.flags.DEFINE_string('net_name', 'baseline',#'close', 'auto_Unet', 'mergeon_go_deeper', 'mergeon_fourier_deep_fuse', 'DCNN', 'mergeon_fourier_deep_group', mergeon_fourier_deep', 'model3_depthwise'， 'baseline_rescale',model3_add_loss_new
	"""The name of the network""")


tf.app.flags.DEFINE_float('learning_rate', 0.001,
	"""learning rate""")

tf.app.flags.DEFINE_float('weight_decay', 0.001, #0.001
	"""L2 weight decay rate""")

tf.app.flags.DEFINE_float('decay_rate', 0.1,#'model3_add_loss_more_2', 'model3_depthwise'， 'baseline_rescale'
	"""Decay rate for the learning rate""")

tf.app.flags.DEFINE_integer('decay_steps', 8000,#20000
	"""Decay step for the learning rate""")

tf.app.flags.DEFINE_float('keep_prob_fc', 0.5,
	"""Keep probability for fully connected layers""")

tf.app.flags.DEFINE_float('keep_prob_conv', 0.6,
	"""Keep probability for convolutional layers""")

tf.app.flags.DEFINE_string('optimizer', 'Adam', #Momentum
	"""The optimizer""")

tf.app.flags.DEFINE_string('loss_type', 'softmax_cross_entropy', #softmax_cross_entropy, dice
	"""Loss function""")


############################################################################
# Directories

tf.app.flags.DEFINE_string('log_dir', './Logs/',
	"""The diretory for logs""")

tf.app.flags.DEFINE_string('save_root_for_prediction', '/home/spc/Documents/Small_Dataset/', #'./Data/'
	"""The directory for saving predictions""")

tf.app.flags.DEFINE_string('data_dir', '/media/DensoML/DENSO ML/tfrecord/', #'./Data/'
	"""The directory of dataset""")


############################################################################
# Data loading

tf.app.flags.DEFINE_float('min_fraction_of_examples_in_queue', 0.5,
	"""Minimul fraction of examples in queue. Used for shuffling data""")

tf.app.flags.DEFINE_integer('num_train_images', 50000, #3600
	"""The number of images used for training""")

tf.app.flags.DEFINE_integer('num_val_images', 10000, # 1200
	"""The number of images used for validation""")

tf.app.flags.DEFINE_integer('num_test_images', 10000, # 1200
	"""The number of images used for test""")


############################################################################
# Transfer learning
tf.app.flags.DEFINE_string('status', 'scratch', # "scratch", "transfer","prune"
	"""Training from teh scatch or use the pretrained model""")

tf.app.flags.DEFINE_string('checkpoint_file', 
	'/home/spc/Dropbox/DeepAcute/Logs_auto/\
autoencoder_test_test_random_lr_0.001_wd_0.001_Aug_31_16_17/model/epoch_95.0_acc_0.405-85500',
	"""Path of the pretrained model if using transfer learning""")

############################################################################
# Utils
# Doesn't need to be changes for most of cases

tf.app.flags.DEFINE_integer('num_labels', 10, #5000
	"""The number of labels""")

tf.app.flags.DEFINE_boolean('seg', False,
	"""Segmentation or classification""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
	"""The number of gpus used for training""")

tf.app.flags.DEFINE_integer('train_epoch', 500,
	"""The number of training epoches""")

tf.app.flags.DEFINE_float('clip_gradients', 5,
	"""Clip gradients""")

tf.app.flags.DEFINE_float('weight_scale', 1,
	"""Only for filter pruning""")

tf.app.flags.DEFINE_float('prob_noise', 0,
	"""Add noised to input images""")
