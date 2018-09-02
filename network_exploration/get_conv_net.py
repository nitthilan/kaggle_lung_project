from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.regularizers import l2

def get_conv_net(input_shape, num_classes, 
	num_extra_conv_layers, wgt_fname=None):
	model = Sequential()

	model.add(Conv2D(64, (3, 3), padding='same',
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	for i in range(num_extra_conv_layers):
		model.add(Conv2D((i+2)*64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D((i+2)*64, (3, 3), padding='same'))
		model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))


	# If a already trained model is present
	# copy weights from one model to another
	if(wgt_fname):
		model1 = load_model(wgt_fname)

		model1.summary()
		model.summary()
		model.load_weights(wgt_fname, by_name=True)


		for layer in model.layers:
			layer.trainable = False

	    # top_model.load_weights(wgt_fname, by_name=True)

	# if(num_extra_conv_layers == 1):
	# 	model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(4, 4)))
	# elif(num_extra_conv_layers == 0):
	# 	model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(8, 8)))

	if(num_extra_conv_layers == 1):
		model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(2, 2)))
	elif(num_extra_conv_layers == 0):
		model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(4, 4)))


	model.add(Flatten())
	# model.add(Dense(512))
	# if(num_extra_conv_layers == 0):
	# 	model.add(Dense(16))
	# else:
	# 	model.add(Dense(32))
	
	model.add(Dense(16, kernel_regularizer=l2(0.01))) #- Reduces the error to small values
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])
	model.summary()

	return model


def get_conv_net_v3(input_shape, num_classes, 
	num_extra_conv_layers):
	model = Sequential()

	model.add(Conv2D(128, (3, 3), padding='same',
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	
	# model.add(Conv2D(64, (3, 3), padding='same'))
	# model.add(Activation('relu'))
	# model.add(Conv2D(64, (3, 3)))
	# model.add(Activation('relu'))
	
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	for i in range(num_extra_conv_layers):
		model.add(Conv2D(256, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(256, (3, 3), padding='same'))
		model.add(Activation('relu'))

		# model.add(Conv2D(128, (3, 3), padding='same'))
		# model.add(Activation('relu'))
		# model.add(Conv2D(128, (3, 3)))
		# model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	return model




# saved_model_v2
def get_conv_net_v2(input_shape, num_classes, num_extra_conv_layers):
	model = Sequential()

	model.add(Conv2D(64, (3, 3), padding='same',
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	
	# model.add(Conv2D(32, (3, 3), padding='same'))
	# model.add(Activation('relu'))
	# model.add(Conv2D(32, (3, 3)))
	# model.add(Activation('relu'))
	
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	for i in range(num_extra_conv_layers):
		model.add(Conv2D(128, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(128, (3, 3), padding='same'))
		model.add(Activation('relu'))

		# model.add(Conv2D(64, (3, 3), padding='same'))
		# model.add(Activation('relu'))
		# model.add(Conv2D(64, (3, 3)))
		# model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	model.summary()
	return model

# saved_model_v1
def get_conv_net_small(input_shape, num_classes, num_extra_conv_layers):
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same',
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	for i in range(num_extra_conv_layers):
		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	return model

