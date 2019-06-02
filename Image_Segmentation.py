import numpy as np
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io as io
from tensorflow.python import pywrap_tensorflow


train_data = '#traindata'
test_data = '#testdata'
segment_label = '#segmentedimages ( Done using Labelme online tool )'

tf.random.set_random_seed(1234)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

size = 224
batch_size = 5
X = tf.placeholder(tf.float32 , [None,size,size,3])
Y = tf.placeholder(tf.float32 , [None , 10])
droupout_prob = tf.placeholder(tf.float32)
X_pred = tf.placeholder(tf.float32 , [None,size,size,3])
Y_true = tf.placeholder(tf.int32 , [None,size,size])
Y_x = tf.placeholder(tf.float32, [None, size, size, 3])


def one_hot_label(img):
    label=str(img.split('.')[0])
    if 'gossiping' in label:
        ohl=[1,0,0,0,0,0,0,0,0,0]
        return ohl
    elif 'isolation' in label:
        ohl=[0,1,0,0,0,0,0,0,0,0]
        return ohl
    elif 'laughing' in label:
        ohl=[0,0,1,0,0,0,0,0,0,0]
        return ohl
    elif 'lp' in label or 'pullinghair' in label:
        ohl=[0,0,0,1,0,0,0,0,0,0]
        return ohl
    elif 'punching' in label:
        ohl=[0,0,0,0,1,0,0,0,0,0]
        return ohl
    elif 'slapping' in label:
        ohl=[0,0,0,0,0,1,0,0,0,0]
        return ohl
    elif 'stabbing' in label:
        ohl=[0,0,0,0,0,0,1,0,0,0]
        return ohl
    elif 'strangle' in label:
        ohl=[0,0,0,0,0,0,0,1,0,0]
        return ohl
    elif '00' in label:
        ohl=[0,0,0,0,0,0,0,0,1,0]
        return ohl
    else:
        ohl=[0,0,0,0,0,0,0,0,0,1]
        return ohl


def train_data_with_label():
    train_images=[]
    for i in tqdm(os.listdir(train_data)):
        path=os.path.join(train_data,i)
        img=cv2.imread(path)
        img=cv2.resize(img,(size,size))
        train_images.append([np.array(img),one_hot_label(i)])
    shuffle(train_images)
    return train_images


def important_function(c):
	y = np.empty(c.shape[:-1])
	print(c.shape[2])
	sing = np.zeros(c.shape[:-1])
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			flat = c[i, j, :].flatten()
			k = np.max(flat)
			for g in range(c.shape[2]):
				if k == c[i,j,g]:
					y[i,j] = g
	return y


def segmented_train_data_with_label():
	segment_images = []
	for i in tqdm(os.listdir(segment_label)):
		path = os.path.join(segment_label, i)
		img_ = io.imread(path)
		img_ = cv2.resize(img_, (size, size))
		img_classes = important_function(img_)
		j = i[:-4] + '.jpg'
		path_ = os.path.join(train_data+"/", j)
		print(path_)
		img = cv2.imread(path_)
		if img is None:
			continue
		else:
			img = cv2.resize(img, (size, size))
			segment_images.append([np.array(img), np.array(img_classes), np.array(img_)])
	shuffle(segment_images)
	return segment_images


training_images = train_data_with_label()
segment_images = segmented_train_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,size,size,3)
tr_lbl_data = np.array([i[1] for i in training_images])


tr_segment_data = np.array([i[0] for i in segment_images]).reshape(-1,size,size,3)
tr_segment_lbl = np.array([i[1] for i in segment_images]).reshape(-1,size,size)
tr_segment = np.array([i[2] for i in segment_images]).reshape(-1,size,size,3)

var_list = []

kernel = [3,3,3,64]
kernel_ = [3,3,64,64]
kernel2 = [3,3,64,128]
kernel2_ = [3,3,128,128]
kernel3 = [3,3,128,256]
kernel3_ = [3,3,256,256]
kernel4 = [3,3,256,512]
kernel5 = [3,3,512,512]
kernel6 = [7 * 7 * 512,4096]
kernel7 = [4096,4096]
kernel8 = [4096 , 10]


def vgg(X,kernel,kernel_,kernel2,kernel2_,kernel3,kernel3_,kernel4,kernel5,kernel6,kernel7,kernel8,droupout_prob, train, circ = False):

	ksize = [1,2,2,1]
	strides = [1,1,1,1]
	strides2 = [1,2,2,1]
	initial = tf.contrib.layers.xavier_initializer()


	#CONVOLUTION BLOCK - 1
	filter1 = tf.Variable(initial(kernel))
	layer1_conv_1 = tf.layers.batch_normalization(tf.nn.relu(tf.nn.conv2d(X ,filter1 , strides = strides , padding = 'SAME')), training = train)

	filter2 = tf.Variable(initial(kernel_))
	layer1_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer1_conv_1 ,filter2 , strides = strides , padding = 'SAME')), droupout_prob), training = train)

	layer1_maxpool = tf.nn.relu(tf.nn.max_pool(layer1_conv_2 , ksize = ksize , strides = [1,2,2,1] , padding = 'SAME'))

	#CONVOLUTION BLOCK - 2
	filter3 = tf.Variable(initial(kernel2))
	layer2_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer1_maxpool ,filter3 ,strides = strides , padding = 'SAME')), droupout_prob), training = train)

	filter4 = tf.Variable(initial(kernel2_))
	layer2_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer2_conv_1 ,filter4 ,strides = strides , padding = 'SAME')), droupout_prob), training = train)

	layer2_maxpool = tf.nn.relu(tf.nn.max_pool(layer2_conv_2 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer2 = tf.nn.dropout(layer2_maxpool , droupout_prob)

	#CONVOLUTION BLOCK - 3
	filter5 = tf.Variable(initial(kernel3))
	layer3_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer2 , filter5 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter6 = tf.Variable(initial(kernel3_))
	layer3_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3_conv_1 , filter6 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter7 = tf.Variable(initial(kernel3_))
	layer3_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3_conv_2 , filter7 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer3_maxpool = tf.nn.relu(tf.nn.max_pool(layer3_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer3 = tf.nn.dropout(layer3_maxpool , droupout_prob)
	shape = layer3.get_shape().as_list()

	#CONVOLUTION BLOCK - 4
	filter8 = tf.Variable(initial(kernel4))
	layer4_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3 , filter8 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter9 = tf.Variable(initial(kernel5))
	layer4_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4_conv_1 , filter9 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter10 = tf.Variable(initial(kernel5))
	layer4_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4_conv_2 , filter10 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer4_maxpool = tf.nn.relu(tf.nn.max_pool(layer4_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer4 = tf.nn.dropout(layer4_maxpool , droupout_prob)
	shape = layer4.get_shape().as_list()

	if circ:
		return layer4

	#CONVOLUTION BLOCK - 5
	filter11 = tf.Variable(initial(kernel5))
	layer5_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4 , filter11 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter12 = tf.Variable(initial(kernel5))
	layer5_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer5_conv_1 , filter12 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter13 = tf.Variable(initial(kernel5))
	layer5_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer5_conv_2 , filter13 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer5_maxpool = tf.nn.relu(tf.nn.max_pool(layer5_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer5 = tf.layers.batch_normalization(tf.nn.dropout(layer5_maxpool , droupout_prob), training = train)
	shape = layer5.get_shape().as_list()

	#FIRST DENSE LAYER
	filter14 = tf.Variable(initial(kernel6))
	shapestraight = [-1 , shape[1] * shape[2] * shape[3]]
	layer6_shaping = tf.reshape(layer5 , shapestraight)
	layer6drop = tf.nn.dropout(layer6_shaping , droupout_prob)
	layer6 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(layer6drop , filter14)), training = train)

	#SECOND DENSE LAYER
	filter15 = tf.Variable(initial(kernel7))
	layer7 = tf.nn.relu(tf.matmul(layer6, filter15))
	layer7 = tf.layers.batch_normalization(tf.nn.dropout(layer7 , droupout_prob), training = train)

	#OUTPUT LAYER
	filter16 = tf.Variable(initial(kernel8))
	output = tf.matmul(layer7 , filter16)
	print("Output",output.get_shape().as_list())
	return output

def DeepLab(X, rates = [1, 2, 4], mrates = [], mgrid = False, apply_batchnorm = False, depth = 256):

	initial = tf.contrib.layers.xavier_initializer()
	res = tf.reduce_mean(X, [1,2], name = 'global_pool', keepdims = True)
	print("RES", res.get_shape().as_list())
	k = tf.Variable(initial([1, 1, 512, depth]))
	var_list.append(k)
	image_level_features = tf.nn.conv2d(res, k, strides = [1, 1, 1, 1], padding = 'SAME', name = 'resize')
	res_ = tf.image.resize_bilinear(image_level_features, (tf.shape(X)[1], tf.shape(X)[2]))
	filter_1 = tf.Variable(initial([1, 1, 512, depth]))
	var_list.append(filter_1)
	res = tf.nn.conv2d(X, filter_1, strides = [1, 1, 1, 1], padding = 'SAME', name = 'conv_1_1')

	filter_2_1 = tf.Variable(initial([3, 3, 512, depth]))
	filter_2_2 = tf.Variable(initial([3, 3, 512, depth]))
	filter_2_3 = tf.Variable(initial([3, 3, 512, depth]))
	var_list.append(filter_2_1)
	var_list.append(filter_2_2)
	var_list.append(filter_2_3)
	
	res1 = tf.nn.conv2d(X, filter_2_1, strides = [1, 1, 1, 1], dilations = [1, rates[0], rates[0], 1], padding = 'SAME', name = 'conv_3_3_1')
	res2 = tf.nn.conv2d(X, filter_2_2, strides = [1, 1, 1, 1], dilations = [1, rates[1], rates[1], 1], padding = 'SAME', name = 'conv_3_3_2')
	res3 = tf.nn.conv2d(X, filter_2_3, strides = [1, 1, 1, 1], dilations = [1, rates[2], rates[2], 1], padding = 'SAME', name = 'conv_3_3_3')

	final = tf.concat((res_, res, res1, res2, res3), axis = 3, name = 'concat')
	final_w = tf.Variable(initial([1, 1, 5 * depth, depth]))
	var_list.append(final_w)
	final_ = tf.nn.conv2d(final, final_w, strides = [1, 1, 1, 1], padding = 'SAME', name = 'final_conv')

	return final_


def complete_DeepLab(input_img, dropout, N = 3, bn = False):

	initial = tf.contrib.layers.xavier_initializer()
	X = vgg(X_pred, kernel, kernel_, kernel2, kernel2_, kernel3, kernel3_, kernel4, kernel5, kernel6, kernel7, kernel8, droupout_prob, train = is_train, circ = True)

	result = DeepLab(X)
	filter_final= tf.Variable(initial([1, 1, 256, N]))
	var_list.append(filter_final)
	res_ = tf.nn.conv2d(result, filter_final, strides = [1, 1, 1, 1], padding = 'SAME', name = 'conv_final')

	return tf.image.resize_bilinear(res_, input_img.shape[1:-1])

def compute_iou(original, prediction):

	H, W, N = original.get_shape().as_list()[1:]
	pred = tf.reshape(prediction, [-1, H * W, N])
	orig = tf.reshape(original, [-1, H * W, N])
	intersection = tf.reduce_sum(pred * orig, axis = 2) + 1e-7
	denominator = tf.reduce_sum(pred, axis = 2) + tf.reduce_sum(orig, axis = 2) + 1e-7
	iou = tf.reduce_mean(intersection / denominator)

	return iou

is_train = tf.placeholder(tf.bool)
train_layer = vgg(X, kernel, kernel_, kernel2, kernel2_, kernel3, kernel3_, kernel4, kernel5, kernel6, kernel7, kernel8, droupout_prob, is_train)

Y_ = tf.nn.softmax(train_layer)
correct = tf.equal(tf.argmax(Y,1) , tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = train_layer)
feature_map_16 = complete_DeepLab(X_pred, dropout = droupout_prob, bn = is_train)
iou = compute_iou(feature_map_16, Y_x)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y_true, logits = feature_map_16)
with tf.control_dependencies(update_ops):
    optimize = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
    optimize_segment =  tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_cls, var_list = var_list)

init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as session:
	session.run(init)
	# TRAINING PHASE
	'''
	for ep in range(40):
		batch_size = 20
		print("Epoch number %d" %(ep))
		x = 0
		ls = []
		count = 0
		for i in range(int(len(tr_img_data)/batch_size)):
			testing = []
			trainimg = tr_img_data[batch_size * i : batch_size * (i+1)]
			trainlbl = tr_lbl_data[batch_size * i : batch_size * (i+1)]
			for i in trainlbl:
				testing.append(np.reshape(i, (-1, 10)))
			testing = np.reshape(testing, (batch_size, 10))
			data = {X : trainimg , Y: testing , X_pred : np.zeros((batch_size, size, size, 3)), Y_x : np.zeros((batch_size, size, size, 3)), Y_true : np.zeros((batch_size, size, size)), is_train : True, droupout_prob: 1.0}
			session.run(optimize , feed_dict = data)
			ls.append(session.run(accuracy , feed_dict = data))
		print("Highest : ", max(ls))
		print("Minimum : ",min(ls))

	# SEGMENTATION CODE
	batch_size = 5
	print("SEGMENTATION TRAINING PHASE")
	for ep in range(15):
		print("Epoch number %d" %(ep))
		ls = []
		for i in range(int(len(tr_segment_data)/batch_size)):
			testing = []
			trainimg = tr_segment_data[batch_size * i : batch_size * (i+1)]
			trainlbl = tr_segment_lbl[batch_size * i : batch_size * (i+1)]
			train_ = tr_segment[batch_size * i : batch_size * (i+1)]
			data = {X : np.zeros((batch_size, size, size, 3)), Y : np.zeros((batch_size, 10)) ,X_pred : trainimg , Y_true : trainlbl , Y_x : train_ ,  is_train : False, droupout_prob: 1.0}
			session.run(optimize_segment , feed_dict = data)
			ls.append(session.run(iou , feed_dict = data))
		print("Highest : ", max(ls))
		print("Minimum : ",min(ls))'''
	
	print("SEGMENTATION TESTING PHASE")
	saver.restore(session, 'C:/Users/vperugu/segwgts.ckpt')
	for _ in range(1):
		img = io.imread('000045.jpg') 
		img = cv2.resize(img,(size,size))
		img = np.reshape(img, (1, size, size, 3))
		o = session.run(feature_map_16, feed_dict = {X : np.zeros((batch_size, size, size, 3)), Y : np.zeros((batch_size, 10)), X_pred : img, Y_x : np.zeros((batch_size, size, size, 3)), Y_true : np.zeros((batch_size, size, size)), droupout_prob : 1.0, is_train : False})
		o = np.reshape(o, (o.shape[1], o.shape[2], o.shape[3]))
		print(o.shape)
		io.imshow(o)
		plt.show() 
