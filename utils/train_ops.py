import tensorflow as tf
from utils import *

def add_all_losses(loss):
	"""
	Description: Add loss from weight decay and scale variables

	Arguments:
	loss: defined loss calculated from logits

	Output:
	total loss
	"""
	weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	#weight_loss = tf.get_collection('loss_weights')
	scale_loss = tf.get_collection('losses')

	w1 = 0; w2 = 0
	if len(weight_loss)>0:
		w1 = tf.add_n(weight_loss)
	
	if len(scale_loss)>0:
		w2 = tf.add_n(scale_loss)

	total_loss = loss + w1 + w2
	return total_loss


def train_operation(loss, var_list, global_step, decay_rate, decay_steps, grads_and_vars, 
	lr=0.001, optimizer='Adam',dict_widx=None, clip_gradients=0):
	"""
	Description: To calculate the gradient and apply the gradient to target variables

	Auguments:
	loss: Target loss. A tenosr.
	var_list: Target variables. A list of tensors.
	global_step: A scalar int32 or int64 Tensor or a Python number. Global step to use for the decay computation. Must not be negative.
	decay_rate: A scalar float32 or float64 Tensor or a Python number. 
	decay_steps: A scalar int32 or int64 Tensor or a Python number. 
	lr: learning rate
	optimizer: Options: 'Adam', 'GSD', 'Momentum', 'RMSProp'
	dice_widx: For filter pruning

	Outputs:
	grads_and_vars
	"""
	learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase=True) # 10000, 0.9

	if optimizer=='Adam':
		opt = tf.train.AdamOptimizer(learning_rate)
	elif optimizer=='SGD':
		opt = tf.train.GradientDescentOptimizer(learning_rate)
	elif optimizer=='Momentum':
		opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=False)
	elif optimizer=='RMSProp':
		opt = tf.train.RMSPropOptimizer(learning_rate)

	#grad = tf.gradients(loss, tf.trainable_variables())
	if grads_and_vars is None:
		grad = tf.gradients(loss, tf.trainable_variables())

		if clip_gradients > 0.0:
			grad, grad_norm = tf.clip_by_global_norm(grad, clip_gradients)
		grads_and_vars = list(zip(grad, tf.trainable_variables()))

	if dict_widx is None:
		apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
	else:
		grads_and_vars = apply_prune_on_grads(grads_and_vars, dict_widx)
		apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

	return grads_and_vars, apply_gradient_op, learning_rate


def prediction_and_accuracy(logits, labels):
	"""
	Description: Calcualte the prediction based on logits and compare it with the ground truth

	"""
	preclass = tf.nn.softmax(logits)
	prediction = tf.argmax(preclass,1)
	correct_prediction = tf.equal(prediction, tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return prediction, accuracy
