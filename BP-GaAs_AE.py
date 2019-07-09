# -*- coding: utf-8 -*-

from keras import backend as K
from keras.models import Sequential


from keras.models import Model
from keras import metrics
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from keras.layers import Input, Dense, Lambda,Conv1D,Conv2DTranspose, LeakyReLU,GlobalAveragePooling1D,Activation,Flatten,Reshape
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

import seaborn as sns

import pandas as pd

from scipy import interpolate as interp
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

K.clear_session()
JV_raw = np.loadtxt('GaAs_sim_nJV.txt')
JV_raw1 = JV_raw[~np.all(JV_raw==0,axis=1)]

par = np.loadtxt('GaAs_sim_label.txt')
par = par.T

        
def Conv1DTranspose(input_tensor, filters, kernel_size, strides ):
    
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),padding='SAME')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


        
def log10_ln(x):
    return np.log(np.power(10,x))

par_ln = log10_ln(par)
from sklearn.preprocessing import MinMaxScaler

def min_max(x):
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min),max,min

JV_norm,JV_max,JV_min = min_max(JV_raw)

plt.figure()
plt.plot(JV_norm[0,:])
scaler = MinMaxScaler()
par_n = scaler.fit_transform(par_ln) 
par_exp_unnorm = np.loadtxt('GaAs_exp_label.txt')
par_exp_unnorm = par_exp_unnorm.T

#epsilon_std = 1
par_exp_ln = log10_ln(par_exp_unnorm)

par_exp = scaler.transform(log10_ln(par_exp_unnorm ))   
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(JV_norm,par_n, test_size=0.2)


input_dim = X_train.shape[1]
label_dim = y_train.shape[1]

x = Input(shape=(input_dim,))


y = Input(shape =(label_dim,))

max_filter = 256

strides = [5,3,2]
kernel = [7,5,3]


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
    z_mean = Dense(label_dim,activation = 'linear')(en3)
   
    
    return z_mean


    


#def sampling(args):
#    z_mean, z_log_var =args
#    epsilon = K.random_normal(shape = (K.shape(z_mean)[0],label_dim),mean=0., stddev = epsilon_std)
#    return z_mean+K.exp(0.5*z_log_var/2)*epsilon

z_mean = encoder(x)
encoder_ = Model(x,z_mean)
encoder_.summary()

map_size = K.int_shape(encoder_.layers[-4].output)[1]
#z = Lambda(sampling, output_shape=(label_dim,))([z_mean,z_log_var])



# do this for recalling the decoder later

z_in = Input(shape=(label_dim,))
z1 = Dense(100,activation = 'relu')(z_in)


z1 = Dense(max_filter*map_size,activation='relu')(z1)
z1 = Reshape((map_size,1,max_filter))(z1)

#x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
#    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),padding='SAME')(x)

z2 =  Conv2DTranspose( max_filter//2, (kernel[2],1), strides=(strides[2],1),padding='SAME')(z1)
z2 = Activation('relu')(z2)

z3 = Conv2DTranspose(max_filter//4, (kernel[1],1), strides=(strides[1],1),padding='SAME')(z2)
z3 = Activation('relu')(z3)

z4 = Conv2DTranspose(1, (kernel[0],1), strides=(strides[0],1),padding='SAME')(z3)
decoded_x = Activation('linear')(z4)

decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)
decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)



    
decoder_ = Model(z_in,decoded_x)
decoder_.summary()

decoded_x = decoder_(y)

ae = Model(inputs= [x,y],outputs= [decoded_x,z_mean])





def ae_loss(x, decoded_x): 
#encoder loss

    encoder_loss = K.sum(K.square(z_mean-y),axis=-1)/label_dim
    
    #decoder loss
    decoder_loss = 50*K.sum(K.square(x- decoded_x),axis=-1)/input_dim
    
    
    #
    ##KL loss
    #kl_loss = K.mean(-0.5* K.sum(1+z_log_var-K.square(z_mean-y)-K.exp(z_log_var),axis=-1))
    
    
    ae_loss = K.mean(encoder_loss+decoder_loss)
    
    return ae_loss
#ae.add_loss(ae_loss)

ae.compile(optimizer = 'adam', loss= ae_loss)
ae.summary()
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                                      patience=5, min_lr=0.00001)
ae.fit(([X_train,y_train]),([X_train,y_train]),shuffle=True, 
        batch_size=128,epochs = 500,
        validation_split=0.0, validation_data=None, callbacks=[reduce_lr])



#build encoder


encoder_ = Model(x,z_mean)

x_test_encoded_1 = encoder_.predict(X_test)


plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded_1[:, 0], y_test[:, 0])

plt.show()







x_test_decoded_1 = decoder_.predict(y_test)


x_test_decoded, x_test_encoded = ae.predict([X_test,y_test])

plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 1], x_test_encoded_1[:, 1])

plt.show()

plt.plot(x_test_decoded[100,:])
#plt.plot(x_test_decoded_1[10,:])
plt.plot(X_test[100,:])
plt.show()




y_hat_train = decoder_.predict(y_train)
y_hat_test = decoder_.predict(y_test)
# voltage sweep unified



v_sweep = np.linspace (0,1.1,50)

v_total =np.tile(v_sweep,6).reshape(1,-1)

p_total = np.multiply(JV_norm,v_total)

sim_eff = np.max(p_total,axis=1)
np.save('sim_eff',sim_eff)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error
mse_train = mse(y_hat_train,X_train)
mse_test = mse(y_hat_test,X_test)

print ('train mse: %.6f' % (mse_train))
print ('test mse: %.6f' % (mse_test))

ae.save('GaAs_AE.h5')
encoder_.save('GaAs_en.h5')

decoder_.save('GaAs_De.h5')
#
from keras.models import load_model
decoder_ = load_model('GaAs_De.h5')

i_par= np.random.randint(1000)
plt.figure()

plt.plot(X_test[i_par,:])
plt.plot(y_hat_test[i_par,:],'--')

JV_exp = np.loadtxt('GaAs_exp_nJV.txt')

JV_exp = (JV_exp-JV_min)/(JV_max -JV_min)

p_exp = np.multiply(JV_exp,v_total)

exp_eff = np.max(p_exp,axis=1)
np.save('exp_eff',exp_eff)

#plt.plot(JV_exp[1,:])

import emcee

from emcee import PTSampler

## we use Parallel Tempering emcee for computing potential muti-modal distribution
Temp = [530,580,630,650,680]

x = -(1000)/(np.array(Temp)+273)

def log_norm_pdf(y,mu,sigma):
    return -0.5*np.sum((y-mu)**2/sigma)+np.log(sigma)
#
#def log_likelihood(theta,x,y,sigma):
#    a1,b1, c1, a2,b2,c2, a3,b3,c3, a4,b4,c4, a5,b5,c5 = theta
#    emitter_doping = np.exp(np.multiply(a1*x,a1*x)+b1*x+c1)
#    back_doping = np.exp(np.multiply(a2*x,a2*x)+b2*x+c2)
#    tau = np.exp(np.multiply(a1*x,a1*x)+b1*x+c1)
#    fsrv = np.exp(np.multiply(a1*x,a1*x)+b1*x+c1)
#    rsrv = np.exp(np.multiply(a1*x,a1*x)+b1*x+c1)
#    
#    par_input = np.stack((emitter_doping,back_doping,tau,fsrv,rsrv),axis=-1)
#    
#    sim_curves= model.predict(par_input)
#    
#    return log_norm_pdf(sim_curves, y,sigma)
    
#import time
#
#start_time = time.time()
#aa=decoder_.predict(par_n[:30])
#
#print("--- %s seconds ---" % (time.time() - start_time))



def log_probability(theta,x,y,sigma):
    a1,b1, c1, a2,b2,c2, a3,b3,c3, a4,b4,c4, a5,b5,c5 = theta
    emitter_doping = (a1*np.multiply(x,x)+b1*x+c1)
    back_doping = (a2*np.multiply(x,x)+b2*x+c2)
    tau = (a3*np.multiply(x,x)+b3*x+c3)
    fsrv = (a4*np.multiply(x,x)+b4*x+c4)
    rsrv = (a5*np.multiply(x,x)+b5*x+c5)
    
    
    
    par_input = np.stack((emitter_doping,back_doping,tau,fsrv,rsrv),axis=-1)
    coeff = [a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5]
    
    #setting prior and constraints
    if all(-20<x<20 for x in coeff):
    
        if np.max(par_input)<0.8 and np.min(par_input)>0.2:
            sim_curves= decoder_.predict(par_input)       
        
            return log_norm_pdf(sim_curves, y,sigma)
        return -np.inf
    return -np.inf
    
def logp(x):
    return 0.0      


sigma = 1e-3
ntemp = 5
pos = np.tile(np.array([-1,-2.2,-0.71]),5)+1e-1*np.random.randn(ntemp,64, 15)

ntemps, nwalkers, ndim = pos.shape

nruns = 20000
Temp_i = 0

sampler = PTSampler(ntemps,nwalkers, ndim, log_probability,logp, loglargs=(x, JV_exp, sigma))
sampler.run_mcmc(pos, nruns )



''''

break


'''

samples = sampler.chain
pos_update = samples[:,:,-1,:]+1e-3*np.random.randn(ntemp,64, 15)

sampler.reset()

sampler = PTSampler(ntemps,nwalkers, ndim, log_probability,logp, loglargs=(x, JV_exp, sigma))

sampler.run_mcmc(pos_update, nruns);
flat_samples = sampler.flatchain
zero_flat_samples = flat_samples[Temp_i,:,:]

zero_samples = samples[Temp_i,:,:,:]

plt.figure()
plt.plot(zero_samples[0,:,0])

zero_flat_loss = sampler.lnprobability[Temp_i,:,:]

plt.figure()
plt.plot(zero_flat_loss[1,:])

#import corner
#fig = corner.corner(flat_samples);    

def check_plot(theta,x,sim):
    
    a1,b1, c1, a2,b2,c2, a3,b3,c3, a4,b4,c4, a5,b5,c5 = theta
    emitter_doping = (a1*np.multiply(x,x)+b1*x+c1)
    back_doping = (a2*np.multiply(x,x)+b2*x+c2)
    tau = (a3*np.multiply(x,x)+b3*x+c3)
    fsrv = (a4*np.multiply(x,x)+b4*x+c4)
    rsrv = (a5*np.multiply(x,x)+b5*x+c5)

    
    
    par_input = np.stack((emitter_doping,back_doping,tau,fsrv,rsrv),axis=-1)
    if sim == 0 :
        unnorm_par = scaler.inverse_transform(par_input)
        return par_input,unnorm_par
        
    sim_curves= decoder_.predict(par_input)
           
  
    return sim_curves, par_input

    

sim_JVs,_ = check_plot(flat_samples[Temp_i,-1,:],x,1)  

par_in = []

x_step = np.linspace(min(x),max(x),50)

for i in range(zero_flat_samples.shape[0]):
    _,par_input = check_plot(zero_flat_samples[i,:],x_step,0)
    
    par_in.append(par_input)
    
    
par_in= np.array(par_in)
par_in = par_in[2000:,:,:]


fig,ax = plt.subplots(5,1)
for i in range(5):
    ax[i,].plot(sim_JVs[i,:],'--')
    ax[i,].plot(JV_exp[i,:])
    
 
    
#import corner 
##
#fig = corner.corner(zero_flat_samples)
#
#
#inds = np.random.randint(len(flat_samples), size=100)


def plot_uncertain(x,y):
    
    mu = np.mean(y,axis = 0)
    std = np.std(y, axis = 0)
    plt.fill_between(x, mu+std,mu-std,alpha=0.1)
    plt.plot(x,mu,label = 'Mean')
    plt.legend()
fig = plt.figure()
for i in range(5):
    plt.subplot(5,1,i+1)
    
    plot_uncertain(x_step,par_in[:,:,i])
    plt.scatter(x, par_exp_ln[:,i])
    
    
par_test = par_in[100,:,:]
f_par = par_test[:,-2]
r_par = par_test[:,-1]
b_par = np.repeat(par_test[-1,:3].reshape(-1,1), len(x_step), axis=1)

#plot_par = np.vstack((b_par,f_par,r_par)).T
t_eff= []
for i in range(len(x_step)):
    f_par = np.roll(f_par,i)
    plot_par = np.vstack((b_par,f_par,r_par)).T
    plot_JV = decoder_.predict(scaler.transform(plot_par))
    plot_p = np.multiply(plot_JV,v_total)

    plot_eff = np.max(plot_p ,axis=1)
    t_eff.append(plot_eff)
    
t_eff = np.array(t_eff)
t_eff1 = t_eff*30

plot_JV = decoder_.predict(plot_par)

plot_p = np.multiply(plot_JV,v_total)

plot_eff = np.max(plot_p ,axis=1)


    
#plt.plot(flat_samples[0,:,2])    


    