import keras
from keras.layers import Input, ReLU, ZeroPadding2D, DepthwiseConv2D, BatchNormalization, Conv2D




def mobilenet(input_tensor):

	default_shape=(300,300,3)
	depth_multiplier=1  ######### need to learn more about this 

	if input_tensor is None:
		input_tensor = Input(shape=default_shape)


############ regularizer can be added to reduce over fitting

####### block 1
	x = ZeroPadding2D(padding=((1,1), (1,1)), name='conv1_pad')(input_tensor) #### ((0,1),(0,1))
	x = Conv2D(32, (3,3), strides=(2,2), padding='valid', use_bias=False, name='conv1')(x)
	x = BatchNormalization()(x) #### need to check axis
	x = ReLU(6.)(x)  ### can use Activation('relu') instead

######### depthwise blocks
	
	x = depthwise_conv_block(x,64,depth_multiplier)
	x = depthwise_conv_block(x,128,depth_multiplier,strides=(2,2))
	x = depthwise_conv_block(x,128,depth_multiplier)
	x = depthwise_conv_block(x,256,depth_multiplier,strides=(2,2))
	x = depthwise_conv_block(x,256,depth_multiplier)
	x = depthwise_conv_block(x,512,depth_multiplier,strides=(2,2))
	x = depthwise_conv_block(x,512,depth_multiplier)
	x = depthwise_conv_block(x,512,depth_multiplier)
	x = depthwise_conv_block(x,512,depth_multiplier)
	x = depthwise_conv_block(x,512,depth_multiplier)
	x = depthwise_conv_block(x,512,depth_multiplier)
	cov4_3=x #### dimension (19,19,512) 
	x = depthwise_conv_block(x,1024,depth_multiplier,strides=(2,2))
	x = depthwise_conv_block(x,1024,depth_multiplier)
	
	## dimension of x (10*10*1024) 
	return x, cov4_3




def depthwise_conv_block(input,conv_filter,depth_multiplier,strides=(1,1)):

	if strides == (1,1):
		x = input
		padding_type = 'same'
	else:
		x = ZeroPadding2D(padding=((1, 1), (1, 1)))(input)
		padding_type = 'valid'

	x = DepthwiseConv2D((3,3), padding=padding_type,
		depth_multiplier=depth_multiplier, strides=strides,
		use_bias=False)(x)
	x = BatchNormalization()(x)
	x = ReLU(6.)(x)
	x = Conv2D(conv_filter, (1,1), padding='same', use_bias=False,
		strides=(1,1))(x)

	x= BatchNormalization()(x)

	return ReLU(6.)(x)








