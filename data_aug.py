import tensorflow as tf 
import random, math
import cv2
import numpy as np
import scipy.stats as st


def elastic_transform(img, alpha, sigma):
    shape = img.get_shape().as_list()
    print(shape)
    rx = tf.random_uniform(shape[0:2], minval=0, maxval=1) * 2 - 1
    rx = tf.reshape(rx,[1]+shape[0:2]+[1])

    smoother = Smoother({'data':rx}, 15, sigma)
    dx = smoother.get_output()* alpha
    #dx = rx
    dx = tf.squeeze(dx)
    ry = tf.random_uniform(shape[0:2], minval=0, maxval=1) * 2 - 1
    ry = tf.reshape(ry,[1]+shape[0:2]+[1])

    smoother = Smoother({'data':ry}, 15, sigma)
    dy = smoother.get_output()* alpha
    #dy = ry
    dy = tf.squeeze(dy)
    flow = tf.stack([dx, dy], axis=-1)
    img = tf.expand_dims(img, axis=0)
    flow = tf.expand_dims(flow, axis=0)
    new_img = tf.contrib.image.dense_image_warp(img, flow)
    new_img = tf.squeeze(new_img)
    return new_img

    
def random_rotation(img, a=1):
    img = tf.contrib.image.rotate(img, 
            tf.random_uniform([1], minval=-a*math.pi, maxval=a*math.pi),
            interpolation='BILINEAR') 
    return img

def random_crop(img, crop_h, crop_w):
    img_size = img.get_shape().as_list()
    img = tf.random_crop(img, [crop_h, crop_w, img_size[-1]])
    return img

def random_shift(img, max_h, max_w):
    img_size = img.get_shape().as_list()
    height = img_size[0]
    width = img_size[1]
    img = tf.image.resize_image_with_crop_or_pad(img, 
                    target_height=height+2*max_h, target_width=width+2*max_w)
    img = tf.random_crop(img, img_size)
    return img

def random_resize(img, max_r):
    img_size = img.get_shape().as_list()
    rsize = tf.random_uniform([1], minval=img_size[0], maxval=max_r, dtype=tf.int32)

    img = tf.image.resize_images(img,
        size= tf.concat([rsize, rsize], axis=0), #* tf.random_uniform([1], minval=1, maxval=max_r)
    )

    img = tf.random_crop(img, img_size)
    return img


def random_flip_left_right(img):
    return tf.image.random_flip_left_right(img)


def random_contrast(img, l_th, h_th):
    img_size = img.get_shape().as_list()
    l_th = -0.2
    h_th = 0.2
    ls = l_th*tf.ones(img_size)
    hs = h_th*tf.ones(img_size)
    mask_l = tf.cast(tf.greater(img, ls), tf.float32)
    mask_h = tf.cast(tf.less(img, hs), tf.float32)
    img = img*mask_l + (1-mask_l)*l_th
    img = img*mask_h + (1-mask_h)*h_th
    return img



def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed('smoothing')
        # Return self for chained calls.
        return self

    return layer_decorated


class Smoother(object):
    def __init__(self, inputs, filter_size, sigma):
        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.filter_size = filter_size
        self.sigma = sigma
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(name = 'smoothing'))

    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            fed_layer = self.layers[fed_layer]
            self.terminals.append(fed_layer)
        return self

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        out_filter = np.array(kernel, dtype = np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis = 2)
        return out_filter

    def make_gauss_var(self, name, size, sigma, c_i):
        with tf.device("/gpu:0"):
            kernel = self.gauss_kernel(size, sigma, c_i)
            var = tf.convert_to_tensor(kernel)
            #var = tf.Variable(tf.convert_to_tensor(kernel), name = name)
        return var

    def get_output(self):
        '''Returns the smoother output.'''
        return self.terminals[-1]

    @layer
    def conv(self,
             input,
             name,
             padding='SAME'):
        # Get the number of channels in the input
        #print(input)
        c_i = input.get_shape().as_list()[3]

        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1],
                                                             padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_gauss_var('gauss_weight', self.filter_size,
                                                         self.sigma, c_i)
            output = convolve(input, kernel)
            return output

