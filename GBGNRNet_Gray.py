# import packages

from conf import myConfig as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,AveragePooling2D,Multiply,Concatenate,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from numpy import *
import random
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus: 
	tf.config.experimental.set_memory_growth(gpu, True)
# custom filter
def my_Hfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[0]], [[1]]],
            [[[-2]], [[0]], [[2]]],
            [[[-1]], [[0]], [[1]]]
        ])
    assert f.shape == shape
    return K.variable(f, dtype='float32')
    
def my_Vfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[-2]], [[-1]]],
            [[[0]], [[0]], [[0]]],
            [[[1]], [[2]], [[1]]]
        ])
    assert f.shape == shape
    return K.variable(f, dtype='float32')

# create CNN model
input_img=Input(shape=(None,None,1))
x=Activation('relu')(input_img)
x1=Conv2D(64,(3,3), dilation_rate=1,padding="same")(x)

x=Activation('relu')(x1)
x=Conv2D(64,(3,3), dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)

x=Conv2D(64,(3,3), padding="same")(x)
x_temp2=Activation('relu')(x)
x=AveragePooling2D(pool_size=(1, 1))(x_temp2)
x=Conv2D(64,(1,1), padding="same")(x)
x_temp3 = Add()([x, x_temp2])

x=Activation('relu')(x_temp3)
x=Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(64,(3,3), dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x2=Conv2D(64,(3,3), dilation_rate=1,padding="same")(x)

x = Subtract()([x2, x1])
x = Conv2D(1,(3,3), padding="same")(x)
x3 = Add()([input_img,x])

x=Activation('relu')(x3)
x=Conv2D(64,(3,3), dilation_rate=1,padding="same")(x)

Gh=Conv2D(filters=1, kernel_size = 3, kernel_initializer=my_Hfilter, padding='same')(x3)
Gv=Conv2D(filters=1, kernel_size = 3, kernel_initializer=my_Vfilter, padding='same')(x3)
Gx=K.sqrt(Gh*Gh + Gv*Gv)

x4 = Concatenate()([x,Gx])

x=Activation('relu')(x4)
x=Conv2D(65,(3,3),dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x11=Conv2D(65,(3,3),dilation_rate=3,padding="same")(x)

x12=Conv2DTranspose(65,(3,3),strides=2)(x11)
x=Conv2D(65,(3,3),strides=2)(x12)
x = Subtract()([x11, x])
x=Conv2DTranspose(65,(3,3),strides=2)(x)
x13 = Add()([x12, x])

x=Activation('relu')(x13)
x13=Conv2D(65,(3,3),padding="same")(x)

x14=Conv2D(65,(3,3),strides=2)(x13)
x=Conv2DTranspose(65,(3,3),strides=2)(x14)
x = Subtract()([x13, x])
x=Conv2D(65,(3,3),strides=2)(x)
x_upper = Add()([x14, x])

x_joint1= Concatenate()([x11,x_upper])
x_joint1= Conv2D(65,(3,3),padding="same")(x_joint1)

x12=Conv2DTranspose(65,(3,3),strides=2)(x_joint1)
x=Conv2D(65,(3,3),strides=2)(x12)
x = Subtract()([x_joint1, x])
x=Conv2DTranspose(65,(3,3),strides=2)(x)
x13 = Add()([x12, x])

x=Activation('relu')(x13)
x13=Conv2D(65,(3,3),padding="same")(x)

x14=Conv2D(65,(3,3),strides=2)(x13)
x=Conv2DTranspose(65,(3,3),strides=2)(x14)
x = Subtract()([x13, x])
x=Conv2D(65,(3,3),strides=2)(x)
x_upper = Add()([x14, x])

x_joint2= Concatenate()([x_joint1,x_upper])

x=Activation('relu')(x_joint2)
x=Conv2D(65,(3,3),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(65,(3,3),dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x5=Conv2D(65,(3,3),dilation_rate=1,padding="same")(x)

x = Subtract()([x5, x4])

x=Conv2D(1,(3,3),padding="same")(x)
x6 = Add()([x, input_img])
model = Model(inputs=input_img, outputs=x6)

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")

def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        noise=random.randint(0,55)
        trueNoiseBatch=np.random.normal(0,noise/255.0,batch.shape)
        noisyImagesBatch=batch+trueNoiseBatch
        yield (noisyImagesBatch,trueNoiseBatch)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(learning_rate=0.001)
def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    lp=K.sum(diff*diff)/(2*config.batch_size)
    return lp
model.compile(loss=custom_loss,optimizer=opt)

# train
model.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
model.save('./Pretrained_models/GBGNRNet_Gray.h5')
