"""
Baesian network
filneame: Bayes.py version: 1.0
    
Two step Bayesian inference(Bayesian network) to map process conditions to materaal properties
@authors: Danny Zekun Ren and Felipe Oviedo
MIT Photovoltaics Laboratory / Singapore and MIT Alliance for Research and Tehcnology
All code is under Apache 2.0 license, please cite any use of the code as explained 
in the README.rst file, in the GitHub repository.
"""

################################################################# 
#Libraries and dependencies
################################################################

from keras.models import load_model
import numpy as np
from emcee import PTSampler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
def ae_loss(x, decoded_x):      
    ae_loss = K.mean(K.sum(K.square(x- decoded_x),axis=-1))          
    return ae_loss

#load JV surrogate model
reg = load_model('./TrainedModel/GaAs_reg.h5')
ae = load_model('./TrainedModel/GaAs_AE.h5',custom_objects={'ae_loss':ae_loss})

#MOCVD growth tempearture
Temp = np.array([530,580,630,650,680])

#convert Tempearture to -1/T*1000 for Arrhenius equation input

x = -1000/(np.array(Temp))

JV_exp =np.loadtxt('./Dataset/GaAs_exp_nJV.txt')

#denoise experimetnal JV using AE
JV_exp = ae.predict(JV_exp)
# Load material parameters that generated the JV dataset
par = np.loadtxt('./Dataset/GaAs_sim_label.txt')

#Normalize JV descriptors column-wise
scaler = MinMaxScaler()

par_n = scaler.fit_transform(par)   


################################################################# 
#Parallel Tempering MCMC to compute the latent parameters
################################################################

#define the lognormal pdf

def log_norm_pdf(y,mu,sigma):
    return -0.5*np.sum((y-mu)**2/sigma)+np.log(sigma)

#define the logprobability based on Arrhenius equation 

def log_probability(theta,x,y,sigma):
    a1,b1, c1, a2,b2,c2, a3,b3,c3, a4,b4,c4, a5,b5,c5 = theta
    emitter_doping = a1*np.log(-1/x)+b1*x+c1
    back_doping = a2*np.log(-1/x)+b2*x+c2
    tau = (a3*np.log(-1/x)+b3*x+c3)
    fsrv = (a4*np.log(-1/x)+b4*x+c4)
    rsrv = (a5*np.log(-1/x)+b5*x+c5)
       
    #stack all 5 materail descriptors
    par_input = 10*np.stack((emitter_doping,back_doping,tau,fsrv,rsrv),axis=-1)
    coeff = [a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5]
    
    #setting prior and constraints
    if all(-10<x<10 for x in coeff) and max(np.abs(coeff[0::3]))<5:
    
        if np.max(par_input)<1 and np.min(par_input)>0:
            sim_curves= reg.predict(par_input)       
        
            return log_norm_pdf(sim_curves, y,sigma)
        return -np.inf
    return -np.inf
    
def logp(x):
    return 0.0      

# Training Parameters
    
sigma = 1e-6
ntemp = 10
nruns = 10000
Temp_i = 0
#initialize the chian with a=0, b=0, c=0.5
pos = np.tile((0,0,0.5),5)/10+1e-4*np.random.randn(ntemp,64, 15)

ntemps, nwalkers, ndim = pos.shape

#first MCMC chain
sampler = PTSampler(ntemps,nwalkers, ndim, log_probability,logp, loglargs=(x, JV_exp, sigma))
sampler.run_mcmc(pos, nruns )
samples = sampler.chain

#use the values obtained in the first MCMC chain to update the inistal estimate
pos_update = samples[:,:,-1,:]+1e-5*np.random.randn(ntemp,64, 15)
sampler.reset()
#second MCM chain
sampler = PTSampler(ntemps,nwalkers, ndim, log_probability,logp, loglargs=(x, JV_exp, sigma))
sampler.run_mcmc(pos_update, nruns);
flat_samples = sampler.flatchain
zero_flat_samples = flat_samples[Temp_i,:,:]
zero_samples = samples[Temp_i,:,:,:]

#visulize a1
plt.figure()
plt.plot(zero_samples[0,:,0])
zero_flat_loss = sampler.lnprobability[Temp_i,:,:]
#visulize loss
plt.figure()
plt.plot(zero_flat_loss[1,:])


#function to show the predicted JV 
def check_plot(theta,x,sim):
    
    a1,b1, c1, a2,b2,c2, a3,b3,c3, a4,b4,c4, a5,b5,c5 = theta
    emitter_doping = a1*np.log(-1/x)+b1*x+c1
    back_doping = a2*np.log(-1/x)+b2*x+c2
    tau = (a3*np.log(-1/x)+b3*x+c3)
    fsrv = (a4*np.log(-1/x)+b4*x+c4)
    rsrv = (a5*np.log(-1/x)+b5*x+c5)

    
    
    par_input = 10*np.stack((emitter_doping,back_doping,tau,fsrv,rsrv),axis=-1)
    if sim == 0 :
        unnorm_par = scaler.inverse_transform(par_input)
        return par_input,unnorm_par
        
    sim_curves= reg.predict(par_input)
           
  
    return sim_curves, par_input



sim_JVs,_ = check_plot(flat_samples[Temp_i,-1,:],x,1)  

#check the fitted results
fig,ax = plt.subplots(5,1)
for i in range(5):
    ax[i,].plot(sim_JVs[i,:],'--')
    ax[i,].plot(JV_exp[i,:])

#Extract materail properties in a finer (-1/T) grid 
x_step = np.linspace(min(x),max(x),50)

par_in = []

for i in range(zero_flat_samples.shape[0]):
    _,par_input = check_plot(zero_flat_samples[i,:],x_step,0)
    
    par_in.append(par_input)
    
   
par_in= np.array(par_in)
#discard the values obtained at the begeinning of the chain
par_in = par_in[2000:,:,:]

par_in = (np.exp(par_in))



################################################################# 
#plotting the materail properties vs temperature 
################################################################

def plot_uncertain(x,y):
    
    mu = np.mean(y,axis = 0)
    std = np.std(y, axis = 0)
    plt.fill_between(x, mu+std,mu-std,alpha=0.1,color='grey')
    plt.plot(x,mu,color='black')

plt.rcParams["figure.figsize"] = [8, 10]
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "calibri"    
fig = plt.figure()
y_label = ['Conc.[cm-3]','Conc.[cm-3]', r'$\tau$ [s]', 'SRV [cm/S]','SRV [cm/S]']
x_labels = ['-1/530' ,'-1/580','-1/630','-1/680']
title = ['Zn emitter doping' , 'Si base doping' ,'bulk lifetime','Front SRV', 'Rear SRV']


for i in range(5):
    plt.subplot(5,1,i+1)
    
    l1=plot_uncertain(x_step,par_in[:,:,i]) 
    plt.yscale('log') 
    plt.ylabel(y_label[i])
    plt.xticks([-1000/530,-1000/580,-1000/630,-1000/680],[])
    plt.title(title[i],fontsize=15,fontweight='bold')
    plt.xlim(-1000/530,-1000/680)
    
  
plt.xticks([-1000/530,-1000/580,-1000/630,-1000/680], x_labels)

plt.xlabel(r'-1/T [1/C]') 

fig.align_labels()
#fig.savefig('figure3.png', format='png',dpi=600)  
    
