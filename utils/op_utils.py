import tensorflow as tf
import numpy as np
import sys, cv2, os, pickle
import scipy.io
import glob

_EPSILON = 1e-8


def weighted_cross_entropy(y_preds, y_trues):
	loss = 0
	weights = [1, 5, 5]
	for i in range(3):
		y_pred = y_preds[:,:,:,i]
		y_true = y_trues[:,:,:,i]

		y_pred = tf.contrib.layers.flatten(y_pred)
		y_true = tf.contrib.layers.flatten(y_true)
		loss = loss + tf.reduce_mean(-1*y_true*tf.map_fn(tf.log, y_pred+1e-3))*weights[i]
	return loss


def weighted_jaccard_loss(y_preds, y_trues):
	dice = 0
	weights = [1, 1]
	for i in range(2):
		y_pred = y_preds[:,:,:,i+1]
		y_true = y_trues[:,:,:,i+1]
		union = y_pred+y_true - y_pred*y_true
		dice_list = -1*tf.reduce_sum(y_pred*y_true,(1,2))/(tf.reduce_sum(union,(1,2))+1)
		dice = dice+tf.reduce_mean(dice_list)
	return dice


def weighted_dice_loss(y_preds, y_trues):
	dice = 0
	weights = [1, 1]
	for i in range(2):
		y_pred = y_preds[:,:,:,i+1]
		y_true = y_trues[:,:,:,i+1]
		union = y_pred+y_true
		dice_list = -2*(tf.reduce_sum(y_pred*y_true,(1,2))+0.5)/(tf.reduce_sum(union,(1,2))+1)
		#dice_list = -2*(tf.reduce_sum(y_pred*y_true,(1,2))+0.5)/(tf.reduce_sum(union,(1,2))+1)
		dice = dice+tf.reduce_mean(dice_list)*weights[i]
	return dice


def save_tofile(Patient_No, epi, wall, Input, Label, Prediction, Location,
                                SliceSpacing, platform='python'):
    #filename = '/home/spc/Documents/Prediction/' + '{}_{:.3f}_{:.3f}'.format(
    #    			Patient_No, epi, wall)
    dirname = '/media/DensoML/DENSO ML/LVData/Prediction/LV2011/'
    if not os.path.isdir(dirname):
    	os.mkdir(dirname)

    filename = dirname + '{}_{:.3f}_{:.3f}'.format(
        			Patient_No, epi, wall)
    
    result = {'Input': Input,
			 'Label': Label,
			 'Prediction': Prediction,
              'Location': Location,
              'SliceSpacing': SliceSpacing}
    if platform=='python':
        with open(filename+'.pkl', 'wb') as handle:
            pickle.dump(result, handle)
    elif platform=='matlab':
        	scipy.io.savemat(filename, result)



def dice(y_pred, y_true, whole=False):
	y_true = y_true.astype(np.float)
	union = y_pred+y_true
	if whole:
		dice = 2*np.sum(y_pred*y_true, (0, 1, 2))/(np.sum(union, (0, 1,2))+0.00001)
	else:
		dice = 2*np.sum(y_pred*y_true, (1, 2))/(np.sum(union, (1, 2))+0.00001)
	return dice


def jaccard(y_pred, y_true, all=False):
	y_true = y_true.astype(np.float)
	union = y_pred+y_true - y_pred*y_true
	if all:
		j = np.sum(y_pred*y_true, (0,1,2))/(np.sum(union, (0,1,2))+0.00001)
	else:
		j = np.sum(y_pred*y_true, (1,2))/(np.sum(union, (1,2))+0.00001)
	return j



def draw_wall_compare(Patient_No, wall, Input, Label, Prediction):
	num = Input.shape[0]			
	for i in range(num):
		path = '/home/spc/Documents/Prediction_cross_entro/' + '{}_{:.3f}'.format(
			Patient_No, wall)
		if not os.path.isdir(path):
			os.mkdir(path)
		print(path)

		#print(Input.shape)
		j = jaccard(Prediction[i:i+1,:,:,1], Label[i:i+1,:,:,1])
		concat_img= np.hstack((Input[i,:,:]*255, 
			Label[i,:,:,1]*255, 
			Prediction[i,:,:,1]*255))

		name =  path + '/' + str(i)
		cv2.imwrite('{}_{:.3f}.png'.format(name, j), concat_img)


def draw_prediction_compare(Patient_No, epi, wall, Input, Label, Prediction, Location=None):
	num = Input.shape[0]			
	root = '/home/spc/Documents/Prediction_Sunny/'
	if not os.path.isdir(root):
		os.mkdir(root)
	for i in range(num):
		path = root + '{}_{:.3f}_{:.3f}'.format(
			Patient_No, epi, wall)
		if not os.path.isdir(path):
			os.mkdir(path)

		#print(Input.shape)
		j = jaccard(Prediction[i:i+1,:,:,1], Label[i:i+1,:,:,1])
		concat_img= np.hstack((Input[i,:,:]*255, 
			(Label[i,:,:,1]*0.5+Label[i,:,:,2])*255, 
			(Prediction[i,:,:,1]*0.5+Prediction[i,:,:,2])*255))

		name =  path + '/' + str(i)
		cv2.imwrite('{}_{:.3f}_{}_{}.png'.format(name, j, int(Location[i][1]), int(Location[i][2])), concat_img)


def draw_prediction_compare_multi(Patient_No, Input, Label, Prediction):
	num = Input.shape[0]			
	root = '/home/spc/Documents/Multi-output/'
	if not os.path.isdir(root):
		os.mkdir(root)
	for i in range(num):
		path = root + '{}'.format(
			Patient_No)
		if not os.path.isdir(path):
			os.mkdir(path)


		j_1 = jaccard(Prediction[i:i+1,:,0:256,1], Label[i:i+1,:,:,1])
		j_2 = jaccard(Prediction[i:i+1,:,256*1:256*2,1], Label[i:i+1,:,:,1])
		j_3 = jaccard(Prediction[i:i+1,:,256*2:256*3,1], Label[i:i+1,:,:,1])
		j_4 = jaccard(Prediction[i:i+1,:,256*3:256*4,1], Label[i:i+1,:,:,1])
		j_5 = jaccard(Prediction[i:i+1,:,256*4:256*5,1], Label[i:i+1,:,:,1])

		concat_img= np.hstack((Input[i,:,:]*255, 
			(Label[i,:,:,1]*0.5+Label[i,:,:,2])*255, 
			(Prediction[i,:,:,1]*0.5+Prediction[i,:,:,2])*255))

		name =  path + '/' + str(i)
		cv2.imwrite('{}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}.png'.format(name, j_1, j_2, j_3, j_4, j_5), concat_img)



def samplewise_zero_center(batch, per_channel=False):
	for i in range(len(batch)):
		if not per_channel:
			batch[i] -= np.mean(batch[i])
		else:
			batch[i] -= np.mean(batch[i], axis=(0, 1, 2), keepdims=True)
	return batch


def samplewise_stdnorm(batch):
	for i in range(len(batch)):
		batch[i] /= (np.std(batch[i], axis=0) + _EPSILON)
	return batch


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 



def shuffle(*arrs):
	""" shuffle.
	Shuffle given arrays at unison, along first axis.
	Arguments:
	    *arrs: Each array to shuffle at unison.
	Returns:
	    Tuple of shuffled arrays.
	"""
	arrs = list(arrs)
	for i, arr in enumerate(arrs):
		assert len(arrs[0]) == len(arrs[i])
		arrs[i] = np.array(arr)
	p = np.random.permutation(len(arrs[0]))
	return tuple(arr[p] for arr in arrs)


def to_categorical(y, nb_classes):
	
	y = np.asarray(y, dtype='int32')
	# high dimensional array warning
	if len(y.shape) > 2:
		warnings.warn('{}-dimensional array is used as input array.'.format(len(y.shape)), stacklevel=2)
	# flatten high dimensional array
	if len(y.shape) > 1:
		y = y.reshape(-1)
	if not nb_classes:
		nb_classes = np.max(y)+1
	Y = np.zeros((len(y), nb_classes))
	Y[np.arange(len(y)),y] = 1.
	return Y


def calculate_number_of_parameters(variables, show=False):
	total_parameters = 0
	for variable in variables:
		#print(variable)
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
		#print(variable_parameters)
	return total_parameters


def add_scaled_noise_to_gradients(self, grads_and_vars, gradient_noise_scale):
	"""Adds scaled noise from a 0-mean normal distribution to gradients."""
	gradients, variables = zip(*grads_and_vars)
	noisy_gradients = []
	for gradient in gradients:
		if gradient is None:
			noisy_gradients.append(None)
			continue
		if isinstance(gradient, ops.IndexedSlices):
			gradient_shape = gradient.dense_shape
		else:
			gradient_shape = gradient.get_shape()
		noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
		noisy_gradients.append(gradient + noise)
	return list(zip(noisy_gradients, variables))


def draw_images(save_root, batch_data, label, subject, index, prediction, accuracy):
	if not os.path.isdir(save_root):
		os.mkdir(save_root)

	num = batch_data.shape[0]		
	for i in range(num):
		subject_path = os.path.join(save_root, str(subject[i]))
		if not os.path.isdir(subject_path):
			os.mkdir(subject_path)

		imgname = os.path.join(subject_path, '{}_{:.3f}.png'.format(index[i], accuracy[i]))

		concat_img= np.hstack((batch_data[i,:,:,0], label[i,:,:]*255, prediction[i,:,:]*255))

		cv2.imwrite(imgname, concat_img)


def calcuate_dice_per_subject(save_root, img_size):
	folders = glob.glob(os.path.join(save_root, '*'))
	dice_list = []
	for subject in folders:
		files = glob.glob(os.path.join(subject, '*png'))
		labels = []
		predictions = []
		for each in files:
			img = cv2.imread(each)
			labels.append(img[:,img_size:img_size*2,0]/255)
			predictions.append(img[:,img_size*2:img_size*3,0]/255)

		labels = np.stack(labels, 0)
		predictions = np.stack(predictions, 0)

		union = labels + predictions
		dice = -2*(np.sum(labels*predictions,(0,1,2))+1e-7)/(np.sum(union, (0,1,2))+1e-7)
		dice_list.append(dice)
	return dice_list
