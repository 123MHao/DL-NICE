import os
import torch
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import   Input
from tensorflow.keras.models import Model
from utils import whitening
import pandas as pd
from utils import freq_Spectrum
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from utils.Network_Operate import Shuffle,SplitVector,ConcatVector,AddCouple,Scale,build_basic_model_v2
from utils.Network_Operate import build_NICE1 ,build_NICE2, build_NICE_reverse

def TFData_preprocessing(x1,x2,batch_size,conditional=True):
  if conditional:
      x=tf.data.Dataset.from_tensor_slices((x1,x2))
      x=x.shuffle(231).batch(batch_size)

  return x

def do_fft_norm(x):
    """
    conduct fft, then normalize the amplitude
    Args:
        x: a sliced vector with zero mean

    Returns: normalized amplitude

    """
    xx0_fft=np.abs(np.fft.fft(x))*2/len(x) # normalization constant: 2/N
    xx0_fft=xx0_fft[:len(x)] # take half of the symmetric result
    return xx0_fft

def freq_Analysis(x,samplingfrequency=12800):
    block_size=len(x)
    x -= np.mean(x)
    x = do_fft_norm(x)
    freqAxis = samplingfrequency * np.array(range(block_size//2)) / block_size
    return freqAxis,x



data = LoadData_pickle(path='../Data/', name='RE21_0', type='rb')

_,x_freq=freq_Analysis(data)
original_dim = 1024
rnd_idx=np.random.choice(50)

# zca processing
zca1=whitening.ZCA(x=data)  #time
X_timezca=zca1.apply(data)

zca2=whitening.ZCA(x=x_freq)  #freq
X_freqzca=zca2.apply(x_freq)



X_train1 = X_timezca
X_train2 = X_freqzca

train_db = TFData_preprocessing(X_train1,X_train2,32)

# Positive - Standardization
# Time series
x_in,x=build_NICE1(original_dim)
encoder1 = Model(x_in, x)
encoder1.summary()

# Frequency
x_in_f,x_f=build_NICE2(original_dim)
encoder2 = Model(x_in_f, x_f)
encoder2.summary()

opt=tf.keras.optimizers.Adam(lr=0.002)
loss_bool =lambda y_true,y_pred:0.5 * K.sum(y_pred**2, 1)
def train_loss(X_train1,X_train2):
    with tf.GradientTape() as tape:
        pred_y1 = encoder1(X_train1)
        pred_y2 = encoder2(X_train2)
        loss = tf.reduce_mean(loss_bool(X_train1, pred_y1))
        var = encoder1.trainable_variables
        opt.apply_gradients(zip(dis_grads, var))
        return loss.numpy()

@tf.function
def train_on_step(image_batch1,image_batch2,epochs):
    for epoch in range(epochs):
        loss = train_loss(image_batch1, image_batch2)
        print(epoch,loss)
    return loss

for images_batch in train_db:
    images1, images2 = images_batch
    loss = train_on_step(images1,images2,10)
    print(loss)

encoder1.save_weights('./weights_T/BALL21.h5')

weights1 = np.array(encoder1.get_weights())

# Inverse-Sample Generated Data
# Time series
x_in,x=build_NICE_reverse(original_dim)
decoder1 = Model(x_in, x)
decoder1.summary()

decoder1.load_weights('./weights_T/BALL21.h5')

# Frequency
x_in_f,x_f=build_NICE_reverse(original_dim)
decoder2 = Model(x_in_f, x_f)
decoder2.summary()

# Generate samples
data_z =[]
for i in range (650):
    z_sample=np.array(np.random.randn(1,original_dim))*0.75
    x_decoded = decoder1.predict(z_sample)
    data_z.extend(x_decoded)
data_z_trans = np.array(data_z)
# print(data_z_trans.shape)

# save
def save_pickle_v1(path,name,x):#
    with open(path+name, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
        print('save to path:', path)
        print('Save successfully!')

path_out= './dataset/'
os.makedirs(path_out,exist_ok=True)
print('Next')
save_pickle_v1(path_out,name='G_SL_2_T_epoch1_zca.pkl',x=data_z_trans)


