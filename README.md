# Neural_Network_Signal_Detector

A neural network method for detecting signals is being investigated. It is of interest to detect signals at a low signal-to-noise ratio (SNR level is approximately or less than 1).
A probing signal, a radio signal emitted by a radar antenna, often has the form of a rectangular pulse. The pulse duration is 3 microseconds.
The main task is to find a rectangular pulse in a noisy signal using a neural network. 
The neural network processes the part of the signal that falls on one detection channel, where the number of these channels determines the radar's range resolution.
The neural network was trained on the basis of a set of placed implementations of a mixture of signals with noise, which were represented by discrete samples.
The result of recognition of a noisy signal (SNR=0.8-0.9) with high reliability was obtained.

https://youtu.be/-877e5pw_aU
