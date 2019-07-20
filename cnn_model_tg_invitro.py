# import tensorflow as tf_
import os

print('aaaaaaaaaaaaaaaaaaaa')
import keras

print('aaaaaaaaaaaaaaaaaa')
import datetime
import numpy as np
from PIL import Image
from numpy import *
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, AveragePooling2D, \
    ZeroPadding2D, add, Flatten, concatenate, Dropout
from keras.optimizers import SGD, Adam, Adamax, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping

now_time = str(datetime.datetime.now())
date = now_time.split(' ')[0]
clork = now_time.split(' ')[-1]
hour = clork.split(':')[0]
min = clork.split(':')[1]


# def load_data():
#     path0 = './Data/0'
#     path1 = './Data/1'
#     filelist0 = [os.path.join(path0,f) for f in os.listdir(path0)]
#     filelist1 = [os.path.join(path1,f) for f in os.listdir(path1)]
#     X_data = []
#     label = []
#     n0 = len(filelist0)
#     n1 =  len(filelist1)
#     for imagePath0 in filelist0:
#         image0 = array(Image.open(imagePath0))
#         X_data.append(image0)
#     for imagePath1 in filelist1:
#         image1 = array(Image.open(imagePath1))
#         X_data.append(image1)
#     for i in range(n0):
#         label.append(0)
#     for i in range(n1):
#         label.append(1)
#     return X_data,label

def lenetModel():
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)

    return x


def Conv_Block(input, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(input, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, input])
        return x


def resnetModel():
    input = Input(shape=(224, 224,3))
    x = ZeroPadding2D((3, 3))(input)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=x)
    # sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9)
    adam = Adam(lr=0.0001)
    # rmsprop = RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def Inception(x, nb_filter):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

    return x


# #def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY):
#     #l2 normalization
#     if weight_decay:
#         kernel_regularizer=regularizers.l2(weight_decay)
#         bias_regularizer=regularizers.l2(weight_decay)
#     else:
#         kernel_regularizer=None
#         bias_regularizer=None
#
#     x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
#
#     if lrn2d_norm:
#         #batch normalization
#         x=BatchNormalization()(x)
#
#     return x


# def inception_module(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=None):
#     (branch1,branch2,branch3,branch4)=params
#     if weight_decay:
#         kernel_regularizer=regularizers.l2(weight_decay)
#         bias_regularizer=regularizers.l2(weight_decay)
#     else:
#         kernel_regularizer=None
#         bias_regularizer=None
#     #1x1
#     pathway1=Conv2D(filters=branch1[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
#
#     #1x1->3x3
#     pathway2=Conv2D(filters=branch2[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
#     pathway2=Conv2D(filters=branch2[1],kernel_size=(3,3),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway2)
#
#     #1x1->5x5
#     pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
#     pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)
#
#     #3x3->1x1
#     pathway4=MaxPooling2D(pool_size=(3,3),strides=1,padding=padding,data_format=DATA_FORMAT)(x)
#     pathway4=Conv2D(filters=branch4[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway4)
#
#     return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)


# def googlenetv4_model():
#     #Data format:tensorflow,channels_last;theano,channels_last
#     if DATA_FORMAT=='channels_first':
#         INP_SHAPE=(3,224,224)
#         img_input=Input(shape=INP_SHAPE)
#         CONCAT_AXIS=1
#     elif DATA_FORMAT=='channels_last':
#         INP_SHAPE=(224,224,3)
#         img_input=Input(shape=INP_SHAPE)
#         CONCAT_AXIS=3
#     else:
#         raise Exception('Invalid Dim Ordering')
#
#     x=conv2D_lrn2d(img_input,64,(7,7),2,padding='same',lrn2d_norm=False)
#     x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
#     x=BatchNormalization()(x)
#
#     x=conv2D_lrn2d(x,64,(1,1),1,padding='same',lrn2d_norm=False)
#
#     x=conv2D_lrn2d(x,192,(3,3),1,padding='same',lrn2d_norm=True)
#     x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
#
#     x=inception_module(x,params=[(64,),(96,128),(16,32),(32,)],concat_axis=CONCAT_AXIS) #3a
#     x=inception_module(x,params=[(128,),(128,192),(32,96),(64,)],concat_axis=CONCAT_AXIS) #3b
#     x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
#
#     x=inception_module(x,params=[(192,),(96,208),(16,48),(64,)],concat_axis=CONCAT_AXIS) #4a
#     x=inception_module(x,params=[(160,),(112,224),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4b
#     x=inception_module(x,params=[(128,),(128,256),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4c
#     x=inception_module(x,params=[(112,),(144,288),(32,64),(64,)],concat_axis=CONCAT_AXIS) #4d
#     x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #4e
#     x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
#
#     x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #5a
#     x=inception_module(x,params=[(384,),(192,384),(48,128),(128,)],concat_axis=CONCAT_AXIS) #5b
#     x=AveragePooling2D(pool_size=(7,7),strides=1,padding='valid',data_format=DATA_FORMAT)(x)
#
#     x=Flatten()(x)
#     x=Dropout(0.4)(x)
#     x=Dense(output_dim=1000,activation='relu')(x)
#     x=Dense(output_dim=1,activation='sigmoid')(x)
#
#
#     model = Model(input, x, name='Inception')
#     model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
#     return model
def googlenetModel():
    input = Input(shape=(224, 224, 3))
    x = Conv2d_BN(input, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 64)
    x = Inception(x, 120)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)
    x = Inception(x, 208)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 208)
    x = Inception(x, 256)
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(input, x, name='Inception')
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


def finetuning_resnet():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # x=Dropout(0.4)(base_model)
    x = Dense(1024, activation='relu')(base_model.output)
    dense_256 = Dense(256, activation='relu', name='dense_256')(x)
    # x = Dense(64, activation='relu')(base_model.output)
    x = Dense(1, activation='sigmoid')(dense_256)
    model = Model(input=base_model.input, outputs=x)
    # sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def AlexNet():
    model = Sequential()
    # input=input(shape=(227,227,1))
    # x=Conv2d_BN(input,96,(11,11),strides=(4,4),padding='valid',)
    
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                     strides=(4, 4), padding='valid',
                     input_shape=(227, 227, 3),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))
    
    model.add(Conv2D(filters=256, kernel_size=(5, 5),
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))
    
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2), padding='valid'))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


def batchTrain(data_path, new_path, model, start, end):
    early_stoping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    res = []
    res_path = new_path + '/'
    if model == 'resNet':
        # os.makedirs(new_path+'/resNet')
        res_path = res_path + 'resNet' + str(start) + '_' + str(end)
        # os.makedirs(res_path)
        for i in range(start, end):
            lo_path = data_path + '/' + str(i) + 'lo'
            train_path = lo_path + '/' + 'train'
            val_path = lo_path + '/val'
            train_datagen = ImageDataGenerator(rescale=1.0 / 255)
            val_datagen = ImageDataGenerator(rescale=1.0 / 255)
            train_generator = train_datagen.flow_from_directory(train_path, shuffle=True, target_size=(224, 224),
                                                                batch_size=32, class_mode='binary')
            val_generator = val_datagen.flow_from_directory(val_path, shuffle=True, target_size=(224, 224),
                                                            batch_size=32, class_mode='binary')
            net = resnetModel()
            net.fit_generator(
                train_generator,
                verbose=2,
                steps_per_epoch=2,
                epochs=100,
                validation_data=val_generator,
                validation_steps=2,
                callbacks=[early_stoping]

            )
            test_path = lo_path + '/test'
            test_datagen = ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(test_path, shuffle=True, target_size=(224, 224),
                                                              batch_size=32, class_mode='binary')
            a = net.predict_generator(test_generator, steps=2, verbose=2)
            # print(a.shape)
            tmp = np.mean(a[:, 0])
            np.save(new_path + '/resNet/' + str(i) + '.npy', tmp)
            res.append(tmp)
            print(i, 'lo finished:', tmp, '\n')
            # print(a)
    if model == 'alexNet':
        # os.makedirs(new_path+'/resNet')
        res_path = res_path + 'resNet' + str(start) + '_' + str(end)
        # os.makedirs(res_path)
        for i in range(start, end):
            lo_path = data_path + '/' + str(i) + 'lo'
            train_path = lo_path + '/' + 'train'
            val_path = lo_path + '/val'
            train_datagen = ImageDataGenerator(rescale=1.0 / 255)
            val_datagen = ImageDataGenerator(rescale=1.0 / 255)
            train_generator = train_datagen.flow_from_directory(train_path, shuffle=True, target_size=(227, 227),
                                                                batch_size=32, class_mode='binary')
            val_generator = val_datagen.flow_from_directory(val_path, shuffle=True, target_size=(227, 227),
                                                            batch_size=32, class_mode='binary')
            net = AlexNet()
            net.fit_generator(
                train_generator,
                verbose=2,
                steps_per_epoch=2,
                epochs=100,
                validation_data=val_generator,
                validation_steps=2,
                callbacks=[early_stoping]

            )
            test_path = lo_path + '/test'
            test_datagen = ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(test_path, shuffle=True, target_size=(227, 227),
                                                              batch_size=32, class_mode='binary')
            a = net.predict_generator(test_generator, steps=2, verbose=2)
            # print(a.shape)
            tmp = np.mean(a[:, 0])
            np.save(new_path + '/alexnet/' + str(i) + '.npy', tmp)
            res.append(tmp)
            print(i, 'lo finished:', tmp, '\n')
    if model == 'lenet':
        res_path = res_path + 'lenet' + str(start) + '_' + str(end)
        os.makedirs(res_path)
        for i in range(start, end):
            lo_path = data_path + '/' + str(i) + 'lo'
            train_path = lo_path + '/' + 'train'
            val_path = lo_path + '/val'
            train_datagen = ImageDataGenerator(rescale=1.0 / 255)
            val_datagen = ImageDataGenerator(rescale=1.0 / 255)
            train_generator = train_datagen.flow_from_directory(train_path, shuffle=True, target_size=(28, 28),
                                                                batch_size=32, class_mode='binary')
            val_generator = val_datagen.flow_from_directory(val_path, shuffle=True, target_size=(28, 28),
                                                            batch_size=32, class_mode='binary')
            net = lenetModel()
            net.fit_generator(
                train_generator,
                verbose=2,
                steps_per_epoch=2,
                epochs=100,
                validation_data=val_generator,
                validation_steps=2,
                callbacks=[early_stoping]

            )
            test_path = lo_path + '/test'
            test_datagen = ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(test_path, shuffle=True, target_size=(28, 28),
                                                              batch_size=32, class_mode='binary')
            a = net.predict_generator(test_generator, steps=2, verbose=2)
            tmp = np.mean(a[:, 0])
            res.append(tmp)
    if model == 'googlenet':
        # os.makedirs(new_path+'/googleNet')
        res_path = res_path + 'googlenet' + str(start) + '_' + str(end)
        # os.makedirs(res_path)
        for i in range(start, end):
            lo_path = data_path + '/' + str(i) + 'lo'
            train_path = lo_path + '/' + 'train'
            val_path = lo_path + '/val'
            train_datagen = ImageDataGenerator(rescale=1.0 / 255)
            val_datagen = ImageDataGenerator(rescale=1.0 / 255)
            train_generator = train_datagen.flow_from_directory(train_path, shuffle=True, target_size=(224, 224),
                                                                batch_size=32, class_mode='binary')
            val_generator = val_datagen.flow_from_directory(val_path, shuffle=True, target_size=(224, 224),
                                                            batch_size=32, class_mode='binary')
            net = googlenetModel()
            net.fit_generator(
                train_generator,
                verbose=2,
                steps_per_epoch=2,
                epochs=100,
                validation_data=val_generator,
                validation_steps=2,
                callbacks=[early_stoping]

            )
            test_path = lo_path + '/test'
            test_datagen = ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(test_path, shuffle=True, target_size=(224, 224),
                                                              batch_size=32, class_mode='binary')
            a = net.predict_generator(test_generator, steps=2, verbose=2)
            # print(a.shape)
            tmp = np.mean(a[:, 0])
            np.save(new_path + '/googlenet/' + str(i) + '.npy', tmp)
            res.append(tmp)
            print(i, 'lo finished:', tmp, '\n')

    res = np.array(res)
    np.save(res_path + '/prop.npy', res)


if __name__ == '__main__':
    
   batchTrain("./resnet", "./tg_invitro/cnn/res", 'resNet',0, 108)
    batchTrain("./googlenet", "./tg_invitro/cnn/res", 'googlenet',0, 108)
    batchTrain("./lenet", "./tg_invitro/cnn/res", 'lenet',0, 108)
	batchTrain("./alexnet", "./tg_invitro/cnn/res", 'alexNet',0, 108)
  
