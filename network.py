import tensorflow as tf
import tflearn
from tflearn import conv_2d, batch_normalization
import numpy as np
from flags import FLAGS
import functools


##################################################################################################################
def resnet(inputs, prob_fc, prob_conv, wd, training_phase=True):
  n = 5
  net = tflearn.conv_2d(inputs, 16, 3, regularizer='L2', weight_decay=0.0001)
  net = tflearn.residual_block(net, n, 16)
  net = tflearn.residual_block(net, 1, 32, downsample=True)
  net = tflearn.residual_block(net, n-1, 32)
  net = tflearn.residual_block(net, 1, 64, downsample=True)
  net = tflearn.residual_block(net, n-1, 64)
  net = tflearn.batch_normalization(net)
  net = tflearn.activation(net, 'relu')
  net = tflearn.global_avg_pool(net)
  # Regression
  net = tflearn.fully_connected(net, 10, activation='linear')
  return net

def baseline(inputs, prob_fc, prob_conv, wd,training_phase=True):
    # First Block
    #print(inputs.shape)
    use_scale=False
    #a = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    a = [64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
    
    #a = [32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    #a = [32, 32,32, 32,32, 32,32, 32,32, 32,32, 32,32]
    conv = inputs
    conv_fn = functools.partial(conv_2d, filter_size=[3,3], activation='relu',
              weights_init='normal',regularizer='L2',weight_decay=wd)
    bn = functools.partial(batch_normalization)

    conv = conv_fn(incoming=conv, nb_filter=a[0], scope="Conv"+'1_1')
    conv = conv_fn(incoming=conv, nb_filter=a[1], scope="Conv"+'1_2')
    conv = bn(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_1')
    conv = tflearn.dropout(conv, prob_conv, name='dropout1')

    # Second Block
    conv = conv_fn(incoming=conv, nb_filter=a[2], scope="Conv"+'2_1')
    conv = conv_fn(incoming=conv, nb_filter=a[3], scope="Conv"+'2_2')         
    conv = bn(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_2')
    conv = tflearn.dropout(conv, prob_conv, name='dropout2')

    # Third Block
    conv = conv_fn(incoming=conv, nb_filter=a[4], scope="Conv"+'3_1')
    conv = conv_fn(incoming=conv, nb_filter=a[5], scope="Conv"+'3_2') 
    conv = conv_fn(incoming=conv, nb_filter=a[6], scope="Conv"+'3_3')
    conv = bn(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_3')
    conv = tflearn.dropout(conv, prob_conv, name='dropout3')

    # Fourth Block
    conv = conv_fn(incoming=conv, nb_filter=a[7], scope="Conv"+'4_1') 
    conv = conv_fn(incoming=conv, nb_filter=a[8], scope="Conv"+'4_2')
    conv = conv_fn(incoming=conv, nb_filter=a[9], scope="Conv"+'4_3')
                      
    conv = bn(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_4')
    conv = tflearn.dropout(conv, prob_conv, name='dropout4')

    # Fifth Block
    conv = conv_fn(incoming=conv, nb_filter=a[10], scope="Conv"+'5_1') 
    conv = conv_fn(incoming=conv, nb_filter=a[11], scope="Conv"+'5_2')
    conv = conv_fn(incoming=conv, nb_filter=a[12], scope="Conv"+'5_3')
                      
    conv = bn(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_6')
    conv = tflearn.dropout(conv, prob_fc, name='dropout6')

    #conv = tf.reduce_mean(conv, [1,2], name='global_pool_1')
    conv = tflearn.fully_connected(conv, 512, activation='relu', scope='fc1')
    #conv = tf.nn.dropout(conv, prob_fc, name='dropout1')
    #conv = tflearn.fully_connected(conv, 4096, activation='relu', scope='fc2')
    conv = tflearn.dropout(conv, prob_fc, name='dropout6')
    conv = tflearn.fully_connected(conv, FLAGS.num_labels, name='fc3') #,activation='softmax'

    return conv

    

def mergeon_fourier_deep(inputs, prob_fc, prob_conv, wd, wd_scale, training_phase):
    first_channel = 32
    drop = prob_fc
    #Inputs = MaxPooling2D(2, padding='same', name='down'+'0')(Inputs)
    inputs = tflearn.batch_normalization(inputs)
    conv1, pool1 = basic_netunit_tf(inputs, '1', first_channel, 3, wd=wd, drop=drop) # 256
    conv2, pool2 = basic_netunit_tf(pool1, '2', first_channel*2, 3, wd=wd,  drop=drop) # 128
    conv3, pool3 = basic_netunit_tf(pool2, '3', first_channel*4, 3, wd=wd, drop=drop) # 64
    conv4, pool4 = basic_netunit_tf(pool3, '4', first_channel*8, 3, wd=wd, drop=drop) # 32

    #pool3=Dropout(drop)(pool3)
    conv5, up5 = basic_netunit_tf(pool4, '5', first_channel*16, 3, wd=wd, drop=drop,down=False) #64
    Merge5 = concatenate([conv4, up5])
    # Merge5=Dropout(drop)(Merge5)
    conv6, up6 = basic_netunit_tf(Merge5, '6', first_channel*8, 3, wd=wd, drop=drop,down=False) # 128
    Merge6 = concatenate([conv3, up6])
    # Merge5=Dropout(drop)(Merge5)
    conv7, up7 = basic_netunit_tf(Merge6, '7', first_channel*4, 3, wd=wd, drop=drop,down=False) # 256
    Merge7 = concatenate([conv2, up7])
    # Merge6=Dropout(drop)(Merge6)
    conv8, up8 = basic_netunit_tf(Merge7, '8', first_channel*2, 3, wd=wd, drop=drop,down=False) # 512
    Merge8 = concatenate([conv1, up8])
    # Merge7=Dropout(drop)(Merge7)
    conv9, _ = basic_netunit_tf(Merge8, '9', first_channel, 3, wd=wd, drop=drop,down=False) # 512

    high_res = tflearn.conv_2d(conv9, FLAGS.num_labels, [1,1],  regularizer='L2',weight_decay=wd, 
            scope="Conv"+'10'+'_1')
    return high_res


def basic_netunit_tf(inputs,No,channel,kernel,wd, drop, down=True):
    network = tflearn.conv_2d(inputs, channel, [3,3], activation='relu', regularizer='L2',weight_decay=wd, 
              scope="Conv"+No+'_1')
    network = tflearn.conv_2d(inputs, channel, [3,3], activation='relu', regularizer='L2',weight_decay=wd,
              scope="Conv"+No+'_2')

    #conv = network
    conv = tflearn.batch_normalization(network)
    #conv = tf.contrib.layers.batch_norm(network, is_training=training_phase, 
    #                  decay=0.99, center=True, scale=True,
    #                  scope='BN'+No, reuse=reuse)

    if down:
      pool = tflearn.max_pool_2d(conv, 2)
      # pool=BatchNormalization(name='BN'+No)(pool)
    else:
      pool = tflearn.upsample_2d(conv, 2)
      # pool= BatchNormalization(name='BN'+No)(pool)
    return [conv,pool]

