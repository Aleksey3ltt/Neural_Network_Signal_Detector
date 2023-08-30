##### Neural Network Search for the Spectrum of a Rectangular Pulse in a Noisy Signal

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal
import numpy as np
from keras import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout,  BatchNormalization, Conv1D, Flatten, MaxPool1D

def pulse_train(Np,fs,tau,t0):
    """
    ------
     Np = number of periods to generate
     fs = samples per period
     tau = duty cycle
     t0 = pulse delay time relative to first rising edge at t = 0

    Return
    ------
    t = time axis array
    x = waveform
    """
    t = np.arange(0,Np*fs+1,1)/fs #time is normalized to make period T0 = 1.0
    x = np.zeros_like(t)
    # Using a brute force approach, just fill x with the sample values
    for k,tk in enumerate(t):
        if np.mod(tk-t0,1) <= tau and np.mod(tk-t0,1) >= 0:
            x[k] = 1
    return t,x

def FT_approx(x,t,Nfft):
    '''
    Approximate the Fourier transform of a finite duration
    signal using scipy.signal.freqz()

    Inputs
    ------
       x = input signal array
       t = time array used to create x(t)
    Nfft = the number of frdquency domain points used to
           approximate X(f) on the interval [fs/2,fs/2], where
           fs = 1/Dt. Dt being the time spacing in array t

    Return
    ------
    f = frequency axis array in Hz
    X = the Fourier transform approximation (complex)
    '''
    fs = 1/(t[1] - t[0])
    t0 = (t[-1]+t[0])/2     # time delay at center
    N0 = len(t)/2           # FFT center in samples
    f = np.arange(-1/2,1/2,1/Nfft)
    w, X = signal.freqz(x,1,2*np.pi*f)
    X /= fs                 # account for dt = 1/fs in integral
    X *= np.exp(-1j*2*np.pi*f*fs*t0)    # time interval correction
    X *= np.exp(1j*2*np.pi*f*N0)        # FFT time interval is [0,Nfft-1]
    F = f*fs
    return F, X

def roll(array, window_size,freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],freq)]

model = Sequential()
model.add(keras.Input(shape=(224, 1)))

# 1st Convolution
model.add(Conv1D(filters = 32, kernel_size = 32, strides=1, padding="same", trainable=True))
model.add(Activation('relu', trainable=True))
model.add(BatchNormalization())
#model.add(MaxPool1D(pool_size=28, strides=1))
model.add(tf.keras.layers.AveragePooling1D(pool_size=28, strides=1))
model.add(Conv1D(filters = 32, kernel_size = 32, strides=1, padding="same", trainable=True))
model.add(Activation('relu', trainable=True))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=28, strides=1))

# 2nd Convolution layer
model.add(Conv1D(filters = 24, kernel_size = 24, strides=1, padding="same", trainable=True ))
model.add(Activation('relu', trainable=True))
model.add(BatchNormalization())
#model.add(MaxPool1D(pool_size=16, strides=1))
model.add(tf.keras.layers.AveragePooling1D(pool_size=16, strides=1))
model.add(Conv1D(filters = 24, kernel_size = 24, strides=1, padding="same", trainable=True ))
model.add(Activation('relu', trainable=True))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=16, strides=1))

# Flattening
model.add(Flatten())

# Fully connected layer
model.add(keras.layers.Dense(32,activation=keras.layers.LeakyReLU(alpha=0.1))) 
#model.add(keras.layers.Dense(32,activation='relu')) 
#model.add(Dropout(0.3))
model.add(BatchNormalization())
#model.add(keras.layers.Dense(1,  activation = 'relu'))
model.add(keras.layers.Dense(1,  activation = 'sigmoid'))


Dmax=20000     # distance to target, m
c=299792458     # speed of light, m/s
tmax=2*Dmax/c   # s
sigmax=1/tmax   # Hz
print('t_max=',tmax, 'v_scan=',sigmax)

sigm=10e6   # Carrier wave, Hz
Tp=1/sigm   # oscillation period, s
tpi=3e-6    # pulse duration
Npi=tpi/Tp
Npmax=tmax/Tp
print('Npi=', Npi,'Npmax=',Npmax)

Np=tmax
tau = tpi    #
fs = 96e6    # sampling frequency
t0 = random.uniform(tmax/4, tmax-tau) # note t0 = tau/2

t,xEnvelope = pulse_train(Np,fs,tau,t0)  #rectangular pulse, envelope
print('num_samples=',len(xEnvelope), max(t), 't0=',t0)
sig0=np.sin(2*np.pi*sigm*(t-t0))*(1/1)  # sinusoidal wave
rectangularPulse = sig0*xEnvelope*1     # rectangular pulse

index_t=round(t0/t[1])
print('index_t=',index_t)

Nsample_width=224   # spectrum window, width 448
Nsample_shift=224   # spectrum window, shift 336
Nsample1_f=2048     # number of reports in frequency spectrum

mean = 0
std = 1.0/0.9
num_samples = len(t)
samples = np.random.normal(mean, std, size=num_samples) + rectangularPulse

#Rolling window
a=roll(t,Nsample_width, Nsample_shift)
b=roll(samples,Nsample_width, Nsample_shift)
sig=roll(sig0,Nsample_width, Nsample_shift)
print('roll - ok, ', 'chanels=',len(a))     

yy1=np.zeros((len(a),Nsample1_f)) 
xx1=np.zeros((len(a),Nsample1_f)) 
zz1=np.zeros((len(a),Nsample1_f)) 
yys_t=np.zeros((len(a),224)) 
m_y=np.zeros(len(a))
cnn=np.zeros(len(a))
count=np.zeros(len(a))   

index_a=round(index_t/Nsample_shift)
print('index_a=',index_a)
deltaF=1/t[1]/Nsample1_f
print('deltaF=',deltaF)
index_sigm=int(sigm/deltaF)
print('index_sigm=',index_sigm)
index_w1=int(0.95e7/deltaF); index_w2=int(1.05e7/deltaF)
index_w12=int(0.47e7/deltaF); index_w22=int(1.5234e7/deltaF)

model.load_weights('c:/rls_train.h5')
for i in range(len(a)):
    print(i)
    xx0,yy0 = FT_approx(b[i],a[i],Nsample1_f)
    xx_0,zz0 = FT_approx(sig[i],a[i],Nsample1_f)
    yy1[i]=yy0
    xx1[i]=xx0
    zz1[i]=zz0
    m_y[i]=max(abs(yy0))
    yys=abs(yy1[i, index_w12+int(Nsample1_f/2):index_w22+int(Nsample1_f/2)])
    cnn0 = model.predict(tf.expand_dims(yys, axis=0),batch_size=1)
    yys_t[i]=yys
    print(cnn0)
    cnn[i]=cnn0
    count[i]=i
    
#C:\python\rls_train

print('A_noise=', np.mean(m_y))
print('index_w1=',index_w1,' index_w2=',index_w2)
print('index_w12=',index_w12,' index_w22=',index_w22)

yy1s=abs(yy1[index_a, index_w1+int(Nsample1_f/2):index_w2+int(Nsample1_f/2)])
yy2s=abs(yy1[index_a, index_w12+int(Nsample1_f/2):index_w22+int(Nsample1_f/2)])

print(model.predict(tf.expand_dims(yy2s, axis=0)))

cumNoise=np.cumsum(m_y[0:index_w2-index_w1])
cumSig=np.cumsum(yy1s)
Pnoise=(index_w2-index_w1)*np.mean(m_y)
SNR=cumSig[index_w2-index_w1-1]/Pnoise
print('Signal-to-noise ratio SNR=',SNR)

xn=np.arange(0,index_w2-index_w1)
Pnoisen=np.full(index_w2-index_w1,Pnoise)

###########
cm = 1/2.54  # centimeters in inches
plt.subplots(figsize=(20*cm, 20*cm))
plt.subplot(3,1,1)
plt.plot(t,xEnvelope, t,rectangularPulse)
plt.ylim([-1.1,1.1])
plt.xlim([0,tmax])
plt.xticks(fontsize=10, rotation=0)
plt.yticks(fontsize=10, rotation=0)
plt.grid()
plt.xlabel(r'Time, s')
plt.ylabel(r'$x_a(t)$')
plt.title(r'Pulse Train Signal:  $x_a(t)$', fontsize=10)

plt.subplot(3,1,2)
plt.plot(t,samples, t,rectangularPulse, t,xEnvelope)
plt.ylim([-4.1,4.1])
plt.xlim([t0-tau/1,t0+2*tau/1])
plt.xticks(fontsize=10, rotation=0)
plt.yticks(fontsize=10, rotation=0)
plt.grid()
plt.xlabel(r'Time, s')
plt.title(r'Signal with noise:  $x_a(t)$', fontsize=10)

plt.subplot(3,1,3)
plt.step(count,cnn)
plt.xlim([0,i])
plt.grid()
plt.xticks(fontsize=10, rotation=0)
plt.yticks(fontsize=10, rotation=0)
plt.title(r'signal recognition by a neural network (probability)', fontsize=10)
plt.xlabel(r'Signal detector channel number (Range detection channel)')
plt.ylabel(r'Probability of finding the signal')
plt.tight_layout()

#plt.subplot(5,1,5)
#plt.plot(xx,abs(yy))
#plt.xlim([0,max(xx)])
#plt.xticks(np.arange(0, max(xx), step=0.15e9), fontsize=9, rotation=0)
#plt.yticks(fontsize=9, rotation=0)
#plt.grid()
#plt.title(r'Spectra Signal with noise', fontsize=7);

#fig2=plt.figure(figsize=(11,11))
#plt.subplot(2,1,1)
#plt.imshow(abs(yy1), cmap='jet')
#plt.subplot(2,1,2)
#plt.plot(xx1[index_a],abs(yy1[index_a]))
#plt.xlim([0,max(xx)])
#plt.grid()
#plt.yticks(fontsize=9, rotation=0)
'''
fig3=plt.figure(figsize=(11,11))
plt.subplot(2,1,1)
for i in range(len(yy1)):
    plt.plot(xx1[i],abs(yy1[i]))
    #plt.xlim([0,max(xx1[i])])
plt.xlim([0,2*sigm])    
plt.grid()
plt.yticks(fontsize=9, rotation=0)

plt.subplot(2,1,2)
plt.plot(xx1[index_a],abs(zz1[index_a]),xx1[index_a],abs(yy1[index_a])) 
#plt.plot(xx1[index_a],abs(yy1[index_a])) 
#plt.xlim([0,2*sigm])
plt.xlim([0.75e7,1.25e7])
plt.grid()
plt.yticks(fontsize=9, rotation=0)

fig4=plt.figure(figsize=(11,11))
plt.subplot(2,1,1)
plt.plot(xn,cumSig, xn, cumNoise)
plt.grid()
plt.yticks(fontsize=9, rotation=0)
'''