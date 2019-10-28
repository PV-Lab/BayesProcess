"""
JV surrogate model
filneame: JV_surrogate.py version: 1.0
    
Surrogate model for denoising experimental JV curves and predicting JV curves from materail descriptors 
@authors: Danny Zekun Ren and Felipe Oviedo
MIT Photovoltaics Laboratory / Singapore and MIT Alliance for Research and Tehcnology
All code is under Apache 2.0 license, please cite any use of the code as explained 
in the README.rst file, in the GitHub repository.
"""

################################################################# 
#Libraries and dependencies
################################################################

from keras import backend as K
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input, Dense, Lambda,Conv1D,Conv2DTranspose, LeakyReLU,Activation,Flatten,Reshape
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Clear Keras F session, if run previously
K.clear_session()

################################################################
# Load data and preprocess
################################################################

# Load simulated and unormalized JV dataset
JV_raw = np.loadtxt('./Dataset/GaAs_sim_nJV.txt')

# Load material parameters that generated the JV dataset
par = np.loadtxt('./Dataset/GaAs_sim_label.txt')



def Conv1DTranspose(input_tensor, filters, kernel_size, strides ):
    
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),padding='SAME')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

#Covert labels from log10 form to log
        
def log10_ln(x):
    return np.log(np.power(10,x))

par = log10_ln(par)


#Data normalization for the whole JV dataset

def min_max(x):
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min),max,min

#Normalize raw JV data

JV_norm,JV_max,JV_min = min_max(JV_raw)

#Normalize JV descriptors column-wise
scaler = MinMaxScaler()

par_n = scaler.fit_transform(par)   

#create training and testing datset

X_train, X_test, y_train, y_test = train_test_split(JV_norm,par_n, test_size=0.2)

#add in Gaussian noise to train the denoising Autoencoder

X_train_nos = X_train+0.002 * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 

X_test_nos = X_test+0.002 * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

################################################################
# build the denosiing AE 
################################################################

input_dim = X_train.shape[1]
label_dim = y_train.shape[1]
#JVi dim
x = Input(shape=(input_dim,))
#materail descriptor dim
y = Input(shape =(label_dim,))

# Network Parameters
max_filter = 256
strides = [5,2,2]
kernel = [7,5,3]
Batch_size = 128

#build the encoder
def encoder(x):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)    
    en0 = Conv1D(max_filter//4,kernel[0],strides= strides[0], padding='SAME')(x)
    en0 = LeakyReLU(0.2)(en0)
    en1 = Conv1D(max_filter//2,kernel[1],strides=strides[1], padding='SAME')(en0)
    en1 = LeakyReLU(0.2)(en1)
    en2 = Conv1D(max_filter,kernel[2], strides=strides[2],padding='SAME')(en1)
    en2 = LeakyReLU(0.2)(en2)
    en3 = Flatten()(en2)
    en3 = Dense(100,activation = 'relu')(en3)
    z = Dense(label_dim,activation = 'linear')(en3) 
    
    return z

z = encoder(x)
encoder_ = Model(x,z)
map_size = K.int_shape(encoder_.layers[-4].output)[1]

#build the decoder
z1 = Dense(100,activation = 'relu')(z)
z1 = Dense(max_filter*map_size,activation='relu')(z1)
z1 = Reshape((map_size,1,max_filter))(z1)
z2 =  Conv2DTranspose( max_filter//2, (kernel[2],1), strides=(strides[2],1),padding='SAME')(z1)
z2 = Activation('relu')(z2)
z3 = Conv2DTranspose(max_filter//4, (kernel[1],1), strides=(strides[1],1),padding='SAME')(z2)
z3 = Activation('relu')(z3)
z4 = Conv2DTranspose(1, (kernel[0],1), strides=(strides[0],1),padding='SAME')(z3)
decoded_x = Activation('sigmoid')(z4)
decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)
decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)

#Denoising autoencoder
ae = Model(inputs= x,outputs= decoded_x)

#ae loss
def ae_loss(x, decoded_x):      
    ae_loss = K.mean(K.sum(K.square(x- decoded_x),axis=-1))          
    return ae_loss

ae.compile(optimizer = 'adam', loss= ae_loss)
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                                      patience=5, min_lr=0.00001)
ae.fit(X_train_nos,X_train,shuffle=True, 
        batch_size=128,epochs = 200,
        validation_split=0.0, validation_data=None, callbacks=[reduce_lr])


x_test_decoded= ae.predict(X_test_nos)

#plot the nosiy JVi and reconstructed JVi
plt.figure(figsize=(6, 6))
rand_ind = np.random.randint(0,100)
plt.plot(x_test_decoded[rand_ind,:],'--',label='AE')
plt.plot(X_test_nos[rand_ind,:],label='raw')
plt.legend()
plt.show()

################################################################
# build the regression model using the same structure of decoder
################################################################

z_in = Input(shape=(label_dim,))
z1 = Dense(100,activation = 'relu')(z_in)
z1 = Dense(max_filter*map_size,activation='relu')(z1)
z1 = Reshape((map_size,1,max_filter))(z1)
z2 =  Conv2DTranspose( max_filter//2, (kernel[2],1), strides=(strides[2],1),padding='SAME')(z1)
z2 = Activation('relu')(z2)
z3 = Conv2DTranspose(max_filter//4, (kernel[1],1), strides=(strides[1],1),padding='SAME')(z2)
z3 = Activation('relu')(z3)
z4 = Conv2DTranspose(1, (kernel[0],1), strides=(strides[0],1),padding='SAME')(z3)
decoded_x = Activation('sigmoid')(z4)
decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)
decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)
reg = Model(z_in,decoded_x)
reg.compile(loss='mse',optimizer='adam')
reg.fit(y_train,X_train,shuffle=True,batch_size=128,epochs = 100,
            validation_split=0.0, validation_data=None)


y_hat_train = reg.predict(y_train)
y_hat_test = reg.predict(y_test)


#voltage sweep
v_sweep = np.linspace (0,1.1,100)

v_total =np.tile(v_sweep,5).reshape(1,-1)


mse = mean_squared_error
mse_train = mse(y_hat_train,X_train)
mse_test = mse(y_hat_test,X_test)

print ('train mse: %.6f' % (mse_train))
print ('test mse: %.6f' % (mse_test))

ae.save('./TrainedModel/GaAs_AE.h5')
reg.save('./TrainedModel/GaAs_reg.h5')

