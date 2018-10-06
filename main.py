
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from scipy.misc import imresize
#import keras.backend as K
import numpy as np
import time, os, cv2
import functools, itertools, six
from operator import is_not
from tensorflow.python.framework import ops

## Import Customized Functions
import network
from flags import FLAGS, TRAIN_RANGE, VAL_RANGE, TEST_RANGE, option, fold, IMG_SIZE
import tflearn
from data_flow import input_fn
from utils import multig, prune, op_utils, train_ops
from medpy.metric.binary import hd, dc


def get_trainable_variables(checkpoint_file, layer=None):
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    saved_shapes = reader.get_variable_to_shape_map()

    checkp_var = [var for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes]

    checkp_name2var = dict(zip(map(lambda x:x.name.split(':')[0], checkp_var), checkp_var))
    all_name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

    if layer==None:
        for name, var in all_name2var.items():
            if name in checkp_name2var:
                tf.add_to_collection('restore_vars', all_name2var[name])
            else:
                print(name)
                tf.add_to_collection('my_new_vars', all_name2var[name])
    else:
        for name, var in all_name2var.items():
            if name in checkp_name2var and 'Block8' not in name:
                tf.add_to_collection('restore_vars', all_name2var[name])
                if 'Block8' in name:
                    tf.add_to_collection('my_new_vars', all_name2var[name])
            else:
                print(name)
                tf.add_to_collection('my_new_vars', all_name2var[name])

    my_trainable_vars = tf.get_collection('my_new_vars')
    restore_vars = tf.get_collection('restore_vars')
    return my_trainable_vars, restore_vars


def weighted_jaccard_loss(y_preds, y_trues):
    dice = 0
    weights = [FLAGS.pscale, 1]
    for i in range(2):
        y_pred = y_preds[:,:,:,i]
        y_true = y_trues[:,:,:,i]
        union = y_pred+y_true - y_pred*y_true
        dice_list = -1*(tf.reduce_sum(y_pred*y_true,axis=(0,1,2))+1e-7)/(tf.reduce_sum(union,axis=(0,1,2))+2e-7)
        dice = dice+tf.reduce_mean(dice_list)*weights[i]
    dice = dice/(1+FLAGS.pscale)
    return dice


class Train():
    def __init__(self, run_id, config):
        self.img_size = IMG_SIZE
        self.run_id = run_id
        self.dict_widx = None
        self.tf_config = config

        # From FLAGS
        self.train_range = TRAIN_RANGE
        self.vali_range = VAL_RANGE
        self.wd = FLAGS.weight_decay
        self.wd_scale = FLAGS.weight_scale
        self.set_id = FLAGS.set_id
        self.am_training = True
    
    
    def _dice_loss(self, logits, index):
        y_pred = tf.nn.softmax(logits)[...,1:]
        y_true = self.batch_labels[index][...,1:]
        
        """
        temp_labels = tf.reshape(y_true, [-1, y_true.shape[-1].value])
        class_weights = tf.constant([[FLAGS.pscale, 1.0]])
        temp_labels = class_weights*temp_labels
        y_true = tf.reshape(temp_labels, (-1, y_true.shape[-3].value, y_true.shape[-2].value, y_true.shape[-1].value))

        union = y_pred+y_true - y_pred*y_true
        dice_list = -1*(tf.reduce_sum(y_pred*y_true,axis=(1,2))+1e-7)/(tf.reduce_sum(union,axis=(1,2))+2e-7)
        """
        dice = weighted_jaccard_loss(y_pred, y_true)
        return tf.reduce_mean(dice)

        
    def _tower_fn(self, index):
        """
        Build computation tower 
        """
        
        logits = getattr(network, FLAGS.net_name)(inputs=self.batch_data[index], 
            prob_fc=self.prob_fc, prob_conv=self.prob_conv, wd=self.wd,
            training_phase=self.am_training)
        labels = self.batch_labels[index]
        if FLAGS.seg:
            argm = tf.reduce_max(input_tensor=logits[0], axis=-1)
            #argm = tf.argmax(logits[0], axis=-1)
            #print(argm.shape)
            temp = tf.stack([argm, argm, argm], axis=-1)
            pred_class = tf.equal(logits[0],temp)
            pred_class = tf.cast(pred_class, tf.float32)
            tower_pred = {
                'classes': pred_class,
                'probabilities': tf.nn.softmax(logits[0])
                }

            tower_loss = self._dice_loss(logits[0],index) + self._dice_loss(logits[1],index) + \
                                    self._dice_loss(logits[2],index) + self._dice_loss(logits[3],index) +\
                                    self._dice_loss(logits[4],index) + self._dice_loss(logits[5],index)
        else:
            temp_labels = tf.reshape(labels, [-1, labels.shape[-1].value])
            temp_logits = tf.reshape(logits, [-1,logits.shape[-1].value])
            """
            class_weights = tf.constant([[1.0, FLAGS.pscale]]) # Need to be customized
            tower_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=class_weights *temp_labels, logits=temp_logits))
            """
            tower_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=temp_labels, logits=temp_logits, name='cross_entropy'))
            tower_pred = {
                'classes': tf.argmax(logits, axis=-1),
                'probabilities': tf.nn.softmax(logits)
                }

        tower_total_loss = train_ops.add_all_losses(tower_loss)

        if FLAGS.status=='transfer':
            model_params, restore_vars = get_trainable_variables(FLAGS.checkpoint_path)
            model_params = model_params+restore_vars
        else:
            model_params = tf.trainable_variables()
            
        tower_grad = tf.gradients(tower_total_loss, model_params)

        return tower_loss, tower_total_loss, zip(tower_grad, model_params), tower_pred


    def _build_graph(self):
        """Resnet model body.
        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
           manages gradient updates.
        """
        global_step = tf.contrib.framework.get_or_create_global_step()
        tower_losses = []
        tower_total_losses = []
        tower_gradvars = []
        tower_preds = []
        num_gpus = FLAGS.num_gpus
        variable_strategy = 'GPU'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = multig.local_device_setter(
                  worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = multig.local_device_setter(
                  ps_device_type='gpu',
                  worker_device=worker_device,
                  ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                      num_gpus, tf.contrib.training.byte_size_load_fn))

            with tf.name_scope('tower_%d' % i) as name_scope:
                with tf.device(device_setter):
                    loss, total_loss, gradvars, preds = self._tower_fn(i)
                    tower_losses.append(loss)
                    tower_total_losses.append(total_loss)
                    tower_gradvars.append(gradvars)
                    tower_preds.append(preds)
                    if i == 0:
                    # Only trigger batch_norm moving mean and variance update from
                    # the 1st tower. Ideally, we should grab the updates from all
                    # towers but these stats accumulate extremely fast so we can
                    # ignore the other stats from the other towers without
                    # significant detriment.
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                   name_scope)
        # Now compute global loss and gradients.
        gradlst = []
        varlst = []

        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradlst.append(avg_grad)
                varlst.append(var)

                if FLAGS.clip_gradients > 0.0:
                    gradlst, grad_norm = tf.clip_by_global_norm(gradlst, FLAGS.clip_gradients)
                gradvars = list(zip(gradlst, varlst))

        if FLAGS.status=='prune' and self.dict_widx:
            print('prune')
            gradvars = prune.apply_prune_on_grads(gradvars, self.dict_widx)

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        

        with tf.device(consolidation_device):
          # Suggested learning rate scheduling from

            self.loss = tf.reduce_mean(tower_losses, name='loss')
            self.total_loss = tf.reduce_mean(tower_total_losses, name='total_loss')
            
            
            #learning_rate = tf.train.exponential_decay(FLAGS.ad_learning_rate, global_step, 
            #    FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
            #self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)
            
            _, apply_gradients_op, learning_rate = train_ops.train_operation(lr=FLAGS.learning_rate, global_step=global_step, 
                            decay_rate=FLAGS.decay_rate, decay_steps=FLAGS.decay_steps, optimizer=FLAGS.optimizer, clip_gradients=FLAGS.clip_gradients,
                            loss=self.total_loss, var_list=tf.trainable_variables(), grads_and_vars=gradvars)
            
            predictions = {
                'classes':
                    tf.concat([p['classes'] for p in tower_preds], axis=0),
                
                #'probabilities':
                #    tf.concat([p['probabilities'] for p in tower_preds], axis=0)
            }
            stacked_labels = tf.concat(self.batch_labels, axis=0)
            #stacked_labels = tf.argmax(input=stacked_labels, axis=1),
            

            if FLAGS.seg:
                y_pred = tf.cast(predictions['classes'], tf.float32)
                y_pred_wall = y_pred[...,1]
                y_true_wall = tf.cast(stacked_labels[...,1], tf.float32)

                union = y_pred_wall + y_true_wall - y_pred_wall * y_true_wall
                dice_list = -1*tf.reduce_sum(y_pred_wall*y_true_wall,axis=(1,2))/(tf.reduce_sum(union,axis=(1,2))+0.001)
                metrics = {
                    'accuracy': tf.reduce_mean(dice_list)
                    }                          
            else:
                correct_prediction = tf.equal(predictions['classes'], tf.argmax(stacked_labels,-1))
                metrics = {
                    'accuracy': tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        #tf.metrics.accuracy(stacked_labels, predictions['classes'])
                }

            # Create single grouped train op
            
            train_op = [apply_gradients_op]
            train_op.extend(update_ops)
            self.train_op = tf.group(*train_op)
            
            self.prediction = predictions['classes']
            self.accuracy = metrics['accuracy']
            

        """
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in gradvars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        

        for var in tf.get_collection('scale_vars'):
            tf.summary.histogram(var.op.name, var)
        """
        tf.summary.image('images', self.batch_data[0], max_outputs=10)
        
        #tf.summary.image('images', tf.cast(self.batch_data[0]*255, tf.uint8), max_outputs=10)
        self.summary_op = tf.summary.merge_all()


    def train(self, **kwargs):
        #with tf.Graph().as_default():
        ops.reset_default_graph()
        sess = tf.Session(config=self.tf_config)

        with sess.as_default():
            # Data Reading objects
            tflearn.is_training(True, session=sess)

            self.am_training = tf.placeholder(dtype=bool, shape=())
            self.prob_fc = tf.placeholder_with_default(0.5, shape=())
            self.prob_conv = tf.placeholder_with_default(0.5, shape=())


            data_fn = functools.partial(input_fn, data_dir=os.path.join(FLAGS.data_dir, FLAGS.set_id), 
                num_shards=FLAGS.num_gpus, batch_size=FLAGS.batch_size, use_distortion_for_training=True)

            self.batch_data, self.batch_labels, _, _ = tf.cond(self.am_training, 
                lambda: data_fn(data_range=self.train_range, subset='train'),
                lambda: data_fn(data_range=self.vali_range, subset='test'))

            #self.batch_data = self.batch_data[0]
            #self.batch_labels = self.batch_labels[0]
            if FLAGS.status=='scratch':
                self.dict_widx = None
                self._build_graph()
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                # Build an initialization operation to run below
                init = tf.global_variables_initializer()
                sess.run(init)

            elif FLAGS.status=='prune':
                self.dict_widx = kwargs['dict_widx']
                pruned_model = kwargs['pruned_model_path']

                self._build_graph()             
                init = tf.global_variables_initializer()
                sess.run(init)

                all_name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
                v_sel = []

                for name, var in all_name2var.items():
                    if 'Adam' not in name:
                        v_sel.append(all_name2var[name])
                self.saver = tf.train.Saver(v_sel)
                #self.saver = tf.train.Saver(tf.global_variables())
                self.saver.restore(sess, pruned_model)
                print('Pruned model restored from ', pruned_model)

            elif FLAGS.status=='transfer':  
                self._build_graph()             
                #tflearn.config.init_training_mode()
                init = tf.global_variables_initializer()
                sess.run(init)
                v1, v2 = get_trainable_variables(FLAGS.checkpoint_path)
                self.saver = tf.train.Saver(tf.global_variables())
                saver =  tf.train.Saver(v2)
                saver.restore(sess, FLAGS.checkpoint_path)
                print('Model restored.')


            #coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(coord=coord)

            # This summary writer object helps write summaries on tensorboard
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir+self.run_id)
            summary_writer.add_graph(sess.graph)

            print('Start training...')
            print('----------------------------------')


            train_steps_per_epoch = FLAGS.num_train_images//FLAGS.batch_size
            report_freq = train_steps_per_epoch

            train_steps = FLAGS.train_epoch * train_steps_per_epoch

            durations = []
            train_loss_list = []
            train_total_loss_list = []
            train_accuracy_list = []

            nparams = op_utils.calculate_number_of_parameters(tf.trainable_variables())
            print(nparams)

            for step in range(train_steps):

                #print('{} step starts'.format(step))
                start_time = time.time()
                tflearn.is_training(True, session=sess)

                data, labels, _, summary_str, loss_value, total_loss, accuracy = sess.run(
                    [self.batch_data, self.batch_labels, self.train_op, self.summary_op, self.loss, self.total_loss, self.accuracy], 
                    feed_dict={self.am_training: True, self.prob_fc: FLAGS.keep_prob_fc, self.prob_conv: FLAGS.keep_prob_conv})
                
                tflearn.is_training(False, session=sess)
                duration = time.time() - start_time
                durations.append(duration)
                train_loss_list.append(loss_value)
                train_total_loss_list.append(total_loss)
                train_accuracy_list.append(accuracy)

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                
                if step%report_freq == 0:
                    start_time = time.time()

                    summary_writer.add_summary(summary_str, step)

                    sec_per_report = np.sum(np.array(durations))
                    train_loss = np.mean(np.array(train_loss_list))
                    train_total_loss = np.mean(np.array(train_total_loss_list))
                    train_accuracy_value = np.mean(np.array(train_accuracy_list))

                    train_loss_list = []
                    train_total_loss_list = []
                    train_accuracy_list = []
                    durations = []

                    train_summ = tf.Summary()
                    train_summ.value.add(tag="train_loss", simple_value=train_loss.astype(np.float))
                    train_summ.value.add(tag="train_total_loss", simple_value=train_total_loss.astype(np.float))
                    train_summ.value.add(tag="train_accuracy", simple_value=train_accuracy_value.astype(np.float))

                    summary_writer.add_summary(train_summ, step)
                                                      
                    vali_loss_value, vali_accuracy_value = self._full_validation(sess)
                    
                    if step%(report_freq*FLAGS.save_epoch)==0:
                        epoch = step/(report_freq*FLAGS.save_epoch)
                        model_dir = os.path.join(FLAGS.log_dir, self.run_id, 'model')
                        if not os.path.isdir(model_dir):
                            os.mkdir(model_dir)
                        checkpoint_path = os.path.join(model_dir, 'epoch_{}_acc_{:.3f}'.format(epoch, vali_accuracy_value))

                        self.saver.save(sess, checkpoint_path, global_step=step)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag="vali_loss", simple_value=vali_loss_value.astype(np.float))
                    vali_summ.value.add(tag="vali_accuracy", simple_value=vali_accuracy_value.astype(np.float))


                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                    vali_duration = time.time() - start_time

                    format_str = ('Epoch %d, loss = %.4f, total_loss = %.4f, acc = %.4f, vali_loss = %.4f, val_acc = %.4f (%.3f ' 'sec/report)')
                    print(format_str % (step//report_freq, train_loss, train_total_loss, train_accuracy_value, vali_loss_value, vali_accuracy_value, sec_per_report+vali_duration))


    def _full_validation(self, sess):
        tflearn.is_training(False, session=sess)
        num_batches_vali = FLAGS.num_val_images // FLAGS.batch_size

        loss_list = []
        accuracy_list = []

        for step_vali in range(num_batches_vali):
            _, _, loss, accuracy = sess.run([self.batch_data, self.batch_labels,self.loss, self.accuracy], 
                feed_dict={self.am_training:False, self.prob_fc: 1, self.prob_conv: 1})
                                            #feed_dict={self.am_training: False, self.prob_fc: FLAGS.keep_prob_fc, self.prob_conv: 1})
            #accuracy = 0
            loss_list.append(loss)
            accuracy_list.append(accuracy)

        vali_loss_value = np.mean(np.array(loss_list))
        vali_accuracy_value = np.mean(np.array(accuracy_list))

        return vali_loss_value, vali_accuracy_value


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("gi", nargs='?', help="index of the gpu",
                        type=int)
    
    gi = parser.parse_args().gi

    if gi==None:
        os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"]= str(gi)
    
    tf_config=tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True 


    if option == 1:
        run_id = '{}_{}_wd_{}_{}'.format(FLAGS.net_name, FLAGS.run_name, FLAGS.weight_scale, time.strftime("%b_%d_%H_%M", time.localtime()))

        # First Training
        train = Train(run_id, tf_config)
        train.train()

    elif option == 2:
        test_path = '/home/spc/Dropbox/ACDC/Logs/vgg_full_test_c1_wd_1.0_Oct_04_15_03_29/model/epoch_7.0_acc_0.829-10920'

        save_root = os.path.join(FLAGS.save_root_for_prediction, 'Train_again')
        if not os.path.isdir(save_root):
            os.mkdir(save_root) 
        test(tf_config, test_path, save_root=save_root, draw=True)


if __name__ == '__main__':
    tf.app.run()



