import keras
from mobilenet import mobilenet
from keras.models import Model
from keras.layers import Input, ReLU, ZeroPadding2D, DepthwiseConv2D, BatchNormalization, Conv2D, Reshape, Concatenate, Activation
from anchorbox import AnchorBox
import numpy as np


def ssd300(img_size,n_classes,min_scale=0.2,max_scale=0.9):

	n_classes=n_classes+1
	aspect_ratio_per_layer=[[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
	[1.0/3.0, 0.5, 1.0, 2.0, 3.0],[1.0/3.0, 0.5, 1.0, 2.0, 3.0],[0.5, 1.0, 2.0],[0.5, 1.0, 2.0]]
	n_boxes= [6,6,6,4,4]
	n_prediction_layers = 5
	scales = np.linspace(min_scale, max_scale, n_prediction_layers+1)
	# steps
	############ regularizer can be added to reduce over fitting (kernel_regularizer)
	###########  5 layers instead of 6

	x = Input(shape=img_size)

	

	fc7,conv4_3=mobilenet(x)

	#### fc7 = (10,10,1024)   #### conv4_7 = (19,19,512)

	###### Blocks
	block6 = downsample_block(256, 512, fc7)  ###  (5,5,512)
	block7 = downsample_block(128, 256, block6) ### (3,3,256)
	block8 = downsample_block(128, 256, block7, (1,1),'valid') ### (2,2,256)


	################## class confindence values for each layer

	conf_class1 = Conv2D(n_boxes[0]*n_classes, (3,3), padding='same',use_bias=False)(conv4_3)
	conf_class2 = Conv2D(n_boxes[1]*n_classes, (3,3), padding='same',use_bias=False)(fc7)
	conf_class3 = Conv2D(n_boxes[2]*n_classes, (3,3), padding='same',use_bias=False)(block6)
	conf_class4 = Conv2D(n_boxes[3]*n_classes, (3,3), padding='same',use_bias=False)(block7)
	conf_class5 = Conv2D(n_boxes[4]*n_classes, (3,3), padding='same',use_bias=False)(block8)

	#################### bounding box localization

	loc_box1 = Conv2D(n_boxes[0]*4, (3,3), padding='same',use_bias=False)(conv4_3)
	loc_box2 = Conv2D(n_boxes[1]*4, (3,3), padding='same',use_bias=False)(fc7)
	loc_box3 = Conv2D(n_boxes[2]*4, (3,3), padding='same',use_bias=False)(block6)
	loc_box4 = Conv2D(n_boxes[3]*4, (3,3), padding='same',use_bias=False)(block7)
	loc_box5 = Conv2D(n_boxes[4]*4, (3,3), padding='same',use_bias=False)(block8)

	############ prior box

	prior_box1 = AnchorBox(img_size, scale=scales[0], scale_next=scales[1], aspect_ratios=aspect_ratio_per_layer[0],n_boxes=n_boxes[0])(loc_box1)
	prior_box2 = AnchorBox(img_size, scale=scales[1], scale_next=scales[2], aspect_ratios=aspect_ratio_per_layer[1],n_boxes=n_boxes[1])(loc_box2)
	prior_box3 = AnchorBox(img_size, scale=scales[2], scale_next=scales[3], aspect_ratios=aspect_ratio_per_layer[2],n_boxes=n_boxes[2])(loc_box3)
	prior_box4 = AnchorBox(img_size, scale=scales[3], scale_next=scales[4], aspect_ratios=aspect_ratio_per_layer[3],n_boxes=n_boxes[3])(loc_box4)
	prior_box5 = AnchorBox(img_size, scale=scales[4], scale_next=scales[5], aspect_ratios=aspect_ratio_per_layer[4],n_boxes=n_boxes[4])(loc_box5)

	########## Reshape layer
	conf_class1_reshape = Reshape((-1,n_classes))(conf_class1)
	conf_class2_reshape = Reshape((-1,n_classes))(conf_class2)
	conf_class3_reshape = Reshape((-1,n_classes))(conf_class3)
	conf_class4_reshape = Reshape((-1,n_classes))(conf_class4)
	conf_class5_reshape = Reshape((-1,n_classes))(conf_class5)

	loc_box1_reshape = Reshape((-1,4))(loc_box1)
	loc_box2_reshape = Reshape((-1,4))(loc_box2)
	loc_box3_reshape = Reshape((-1,4))(loc_box3)
	loc_box4_reshape = Reshape((-1,4))(loc_box4)
	loc_box5_reshape = Reshape((-1,4))(loc_box5)

	prior_box1_reshape = Reshape((-1,8))(prior_box1)
	prior_box2_reshape = Reshape((-1,8))(prior_box2)
	prior_box3_reshape = Reshape((-1,8))(prior_box3)
	prior_box4_reshape = Reshape((-1,8))(prior_box4)
	prior_box5_reshape = Reshape((-1,8))(prior_box5)

	######## Concatenate

	mbox_conf = Concatenate(axis=1)([conf_class1_reshape,conf_class2_reshape,
		conf_class3_reshape,conf_class4_reshape,conf_class5_reshape])

	mbox_loc = Concatenate(axis=1)([loc_box1_reshape,loc_box2_reshape,
		loc_box3_reshape,loc_box4_reshape,loc_box5_reshape])

	mbox_prior = Concatenate(axis=1)([prior_box1_reshape,prior_box2_reshape,
		prior_box3_reshape,prior_box4_reshape,prior_box5_reshape])


	mbox_conf_softmax = Activation('softmax')(mbox_conf)

	predictions = Concatenate(axis=2)([mbox_conf_softmax,mbox_loc,mbox_prior])

	model = Model(inputs=x, outputs=predictions)

	predictor_sizes = np.array([conf_class1._keras_shape[1:3],
		conf_class2._keras_shape[1:3],conf_class3._keras_shape[1:3],
		conf_class4._keras_shape[1:3],conf_class5._keras_shape[1:3]])

	return model, predictor_sizes



def downsample_block(conv_filter1, conv_filter2, previous_block, strides=(2,2),padding='same'):

	x = Conv2D(conv_filter1, (1,1), padding='same',use_bias=False)(previous_block)
	x = BatchNormalization()(x)
	x = ReLU(6.)(x) ### can use Activation('relu') instead
	x = Conv2D(conv_filter2, (3,3), padding=padding, strides=strides, use_bias=False)(x)
	x = BatchNormalization()(x)
	x = ReLU(6.)(x)

	return x
