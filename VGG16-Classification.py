#Authors : Sneha Perugupalli 
#Title : Cyberbullying vs NoCyberbullying Classification
#Date : 04-10-2019

import numpy as np
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt


train_data= '/home/vperugu/DL_new/Final_train'
test_data=  '/home/vperugu/DL_new/Final_test'



tf.random.set_random_seed(1234)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

size = 224
batch_size = 16
X = tf.placeholder(tf.float32 , [None,size,size,1])
Y = tf.placeholder(tf.float32 , [None , 10])
droupout_prob = tf.placeholder(tf.float32)


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


def rotate_images(X_imgs):
    X_rotate = []
    X = tf.placeholder(tf.float32, shape = (size, size, 1))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        for i in range(3):  # Rotation at 90, 180 and 270 degrees
            rotated_img = sess.run(tf_img, feed_dict = {X: X_imgs, k: i + 1})
            X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    print("Rotate", X_rotate.shape)
    return X_rotate


def flip_images(X_imgs):
    X_flip = []
    X = tf.placeholder(tf.float32, shape = (size, size, 1))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        flipped_imgs = sess.run([tf_img1,tf_img2,tf_img3], feed_dict = {X:X_imgs})
        X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    print("Flip", X_flip.shape)
    return X_flip


def central_scale_images(X_imgs, scales):
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale 
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([size, size], dtype = np.int32)
    
    X_scale_data = []
    X = tf.placeholder(tf.float32, shape = (1, size, size, 1))
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_img = np.expand_dims(X_imgs, axis = 0)
        scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
        X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    print("Scale", X_scale_data.shape)
    return X_scale_data

def test_data_with_label():
    test_images=[]
    for i in tqdm(os.listdir(test_data)):
        path=os.path.join(test_data,i)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(size,size))
        test_images.append([np.array(img),one_hot_label(i)])
        shuffle(test_images)
    return test_images

def train_data_with_label():
	seg_images=[]
	for i in tqdm(os.listdir(train_data)):
		path = os.path.join(train_data,i)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		plt.imshow(img)
		plt.show()
		img=cv2.resize(img,(size,size))
		img=np.reshape(img,(224,224,1))
		seg_images.append([np.array(img),one_hot_label(i)])
		img1 = rotate_images(img)
		for j in img1:
			seg_images.append([np.array(j),one_hot_label(i)])
		img2 = flip_images(img)
		print("this",img2.shape)
		for j in img2:
			seg_images.append([np.array(j), one_hot_label(i)])
		scaled_imgs = central_scale_images(img, [0.90, 0.75, 0.60])
		for j in scaled_imgs:
			seg_images.append([np.array(j),one_hot_label(i)])
	shuffle(seg_images)
	return seg_images

training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,size,size,1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,size,size,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

def vgg(X,kernel,kernel_,kernel2,kernel2_,kernel3,kernel3_,kernel4,kernel5,kernel6,kernel7,kernel8,droupout_prob, train):

	ksize = [1,2,2,1]
	strides = [1,1,1,1]
	strides2 = [1,2,2,1]
	X /= 255
	initial = tf.contrib.layers.xavier_initializer()


	#CONVOLUTION BLOCK - 1
	filter1 = tf.Variable(initial(kernel))
	layer1_conv_1 = tf.layers.batch_normalization(tf.nn.relu(tf.nn.conv2d(X ,filter1 , strides = strides , padding = 'SAME')), training = train)

	filter2 = tf.Variable(initial(kernel_))
	layer1_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer1_conv_1 ,filter2 , strides = strides , padding = 'SAME')), droupout_prob), training = train)
	layer1_maxpool = tf.nn.max_pool(layer1_conv_2 , ksize = ksize , strides = [1,2,2,1] , padding = 'SAME')

	#CONVOLUTION BLOCK - 2
	filter3 = tf.Variable(initial(kernel2))
	layer2_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer1_maxpool ,filter3 ,strides = strides , padding = 'SAME')), droupout_prob), training = train)

	filter4 = tf.Variable(initial(kernel2_))
	layer2_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer2_conv_1 ,filter4 ,strides = strides , padding = 'SAME')), droupout_prob), training = train)

	layer2_maxpool = tf.nn.max_pool(layer2_conv_2 , ksize = ksize , strides = strides2 , padding = 'SAME')
	layer2 = tf.nn.dropout(layer2_maxpool , droupout_prob)

	#CONVOLUTION BLOCK - 3
	filter5 = tf.Variable(initial(kernel3))
	layer3_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer2 , filter5 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter6 = tf.Variable(initial(kernel3_))
	layer3_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3_conv_1 , filter6 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter7 = tf.Variable(initial(kernel3_))
	layer3_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3_conv_2 , filter7 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer3_maxpool = tf.nn.max_pool(layer3_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME')
	layer3 = tf.nn.dropout(layer3_maxpool , droupout_prob)
	shape = layer3.get_shape().as_list()

	#CONVOLUTION BLOCK - 4
	filter8 = tf.Variable(initial(kernel4))
	layer4_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3 , filter8 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter9 = tf.Variable(initial(kernel5))
	layer4_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4_conv_1 , filter9 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter10 = tf.Variable(initial(kernel5))
	layer4_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4_conv_2 , filter10 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer4_maxpool = tf.nn.max_pool(layer4_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME')
	layer4 = tf.nn.dropout(layer4_maxpool , droupout_prob)
	shape = layer4.get_shape().as_list()

	#CONVOLUTION BLOCK - 5
	filter11 = tf.Variable(initial(kernel5))
	layer5_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4 , filter11 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter12 = tf.Variable(initial(kernel5))
	layer5_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer5_conv_1 , filter12 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	filter13 = tf.Variable(initial(kernel5))
	layer5_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer5_conv_2 , filter13 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer5_maxpool = tf.nn.max_pool(layer5_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME')
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
	return output

kernel = [3,3,1,64]
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

is_train = tf.placeholder(tf.bool)
train_layer = vgg(X, kernel, kernel_, kernel2, kernel2_, kernel3, kernel3_, kernel4, kernel5, kernel6, kernel7, kernel8, droupout_prob, is_train)

Y_ = tf.nn.softmax(train_layer)
correct = tf.equal(tf.argmax(Y,1) , tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = train_layer)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimize = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)

init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as session:
	session.run(init)

	# TRAINING PHASE
	for ep in range(40):
		print("Epoch number %d" %(ep))
		x = 0
		ls = []
		count = 0
		print(tr_img_data.shape)
		for i in range(int(len(tr_img_data)/batch_size)):
			testing = []
			trainimg = tr_img_data[batch_size * i : batch_size * (i+1)]
			trainlbl = tr_lbl_data[batch_size * i : batch_size * (i+1)]
			for i in trainlbl:
				testing.append(np.reshape(i, (-1, 10)))
			testing = np.reshape(testing, (batch_size, 10))
			data = {X : trainimg , Y: testing ,is_train : True, droupout_prob: 1.0}
			session.run(optimize , feed_dict = data)
			print("Accuracy:",session.run(accuracy , feed_dict = data))
			ls.append(session.run(accuracy , feed_dict = data))
	
	# TESTING PHASE
	print("TESTING PHASE!!")
	testing = []
	batch_size = 30
	for i in range(int(len(tst_img_data)/batch_size)):
		testing = []
		testimg = tst_img_data[batch_size * i : batch_size * (i+1)]
		testlbl = tst_lbl_data[batch_size * i : batch_size * (i+1)]
		for i in testlbl:
			testing.append(np.reshape(i, (-1, 10)))
		testing = np.reshape(testing, (batch_size, 10))
		data = {X : testimg , Y: testing , is_train : False, droupout_prob: 1.0}
		print(session.run(accuracy , feed_dict = data))
	
