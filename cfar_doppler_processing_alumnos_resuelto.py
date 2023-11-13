# -*- coding: utf-8 -*-
"""

@author: fdadam
"""

#%% Libs and functions
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft, fftshift, fftfreq

def fastconv(A,B):
    out_len = len(A)+len(B)-1
    
    # Next nearest power of 2
    sizefft = int(2**(np.ceil(np.log2(out_len))))
    
    Afilled = np.concatenate((A,np.zeros(sizefft-len(A))))
    Bfilled = np.concatenate((B,np.zeros(sizefft-len(B))))
    
    fftA = fft(Afilled)
    fftB = fft(Bfilled)
    
    fft_out = fftA * fftB
    out = ifft(fft_out)
    
    out = out[0:out_len]
    
    return out

#%% Parameters

c = 3e8 # speed of light [m/s]
k = 1.380649e-23 # Boltzmann

fc = 1.3e9 # Carrier freq
fs = 10e6 # Sampling freq
Np = 100 # Intervalos de sampling
Nint = 10
NPRIs = Nint*Np
ts = 1/fs

Te = 5e-6 # Tx recovery Time[s]
Tp = 10e-6 # Tx Pulse Width [s]
BW = 2e6 # Tx Chirp bandwidth [Hz]
PRF = 1500 # Pulse repetition Frequency [Hz]

wlen = c/fc # Wavelength [m]
kwave = 2*np.pi/wlen # Wavenumber [rad/m]
PRI = PRF**(-1) # Pulse repetition interval [s]
ru = (c*(PRI-Tp-Te))/2 # Unambigous Range [m]
vu_ms = wlen*PRF/2 # Unambigous Velocity [m/s]
vu_kmh = vu_ms*3.6 # Unambigous Velocity [km/h]

rank_min = (Tp/2+Te)*c/2 # Minimum Range [m]
rank_max = 30e3 # Maximum Range [m] (podría ser el Ru)
#rank_max = ru
rank_res = ts*c/2 # Range Step [m]
tmax = 2*rank_max/c # Maximum Simulation Time

radar_signal = pd.read_csv('signal_example.csv',index_col=None)
radar_signal = np.array(radar_signal['real']+1j*radar_signal['imag'])
radar_signal = radar_signal.reshape(Np,-1)

print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu_ms:.2f} m/s')
print(f'Unambiguous Velocity. Vu = {vu_kmh:.2f} km/h')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')

#%% Signals

# Independant Variables

Npts = int(tmax/ts) # Simulation Points
t = np.linspace(-tmax/2,tmax/2,Npts)
ranks = np.linspace(rank_res,rank_max,Npts) # Range Vector
f = fftfreq(Npts,ts) # Freq Vector

# Tx Signal

tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2) # Tx Linear Chiprs (t)
tx_rect = np.where(np.abs(t)<=Tp/2,1,0) # Rect Function
tx_chirp = tx_rect*tx_chirp # Tx Chirp Rectangular
tx_chirp_f = fft(tx_chirp,norm='ortho') # Tx Chirp (f)

# Matched Filter

matched_filter = np.conj(np.flip(tx_chirp))
matched_filter_f = fft(matched_filter,norm='ortho')



#%% Plot Signals

fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True)

fig.suptitle('Received Signal')

ax = axes[0]
ax.plot(ranks/1e3,np.real(radar_signal[0]))
ax.plot(ranks/1e3,np.imag(radar_signal[0]))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(radar_signal[0]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)


## ================= RESOLUCION =================

#%% Compresion de señal

signal_comp = []
for t in range(len(radar_signal)):
    # convolucionar y quitar colas de convolucion
    comp = fastconv(radar_signal[t],matched_filter)[len(matched_filter)//2:len(radar_signal[t])+len(matched_filter)//2]
    signal_comp.append(comp)
signal_comp = np.stack(signal_comp,axis=0)

fig, axes = plt.subplots(2,1,figsize=(10,10))

fig.suptitle('Uncompressed & Compressed Signal')

ax = axes[0]
ax.plot(ranks/1e3,np.abs(radar_signal[0]))
ax.set_ylabel('Abs value')
ax.set_xlabel('Rx Raw signal')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(signal_comp[0]))
ax.set_ylabel('Abs value')
ax.set_xlabel('Rx compressed signal')
ax.grid(True)

#%% CFAR window

n_gap = 15
n_ref = 100
v_ref = 1
h_cfar_gain = (1/(n_ref*2*v_ref))
h_cfar = np.concatenate((np.repeat(v_ref,n_ref),
                         np.repeat(0,n_gap*2+1),
                         np.repeat(v_ref,n_ref)))*h_cfar_gain

fig, ax = plt.subplots(1,1,figsize=(8,8))

fig.suptitle('CFAR Window')

ax.step(range(len(h_cfar)),h_cfar,marker='.')
ax.set_xlabel('Sample')
ax.set_ylabel('CFAR window Value')
ax.annotate(text=f'Reference Cells: {n_ref*2}\nGap Cells: {n_gap}',
                    xy=(0.2,0.2),
                    xycoords='figure fraction',
                    bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
ax.grid(True)

#%% MTI SC

# Se resta la segunda fila a la primera

MTI_matrix_sc = np.array([1,-1])
cfar_MTI_sc = np.inner(signal_comp[0:2].T,MTI_matrix_sc)

gain_MTI = 4

# convolucionar y quitar colas de convolucion
th_cfar_MTI_sc = gain_MTI*fastconv(np.abs(cfar_MTI_sc),h_cfar)
th_cfar_MTI_sc = th_cfar_MTI_sc[len(h_cfar)//2:len(cfar_MTI_sc)+len(h_cfar)//2]

dif_th_MTI_sc = np.sign(np.abs(cfar_MTI_sc) - np.abs(th_cfar_MTI_sc))
dif_th_MTI_sc = np.diff(dif_th_MTI_sc,n=1)
dif_th_MTI_sc = np.concatenate((dif_th_MTI_sc,np.zeros(len(ranks)-len(dif_th_MTI_sc))))

targets_rank = ranks[dif_th_MTI_sc==2]/1e3

# Plot

fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
fig.suptitle('MTI Simple Cancelador')
fig.subplots_adjust(left=0.3)

ax = axes[0]
ax.plot(ranks/1e3,np.abs(radar_signal[0]),label='Rx $t_0$')
ax.plot(ranks/1e3,np.abs(radar_signal[1]),label='Rx $t_1$')
ax.set_title('Rx Raw signals')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(signal_comp[0]),label='Comp $t_0$')
ax.plot(ranks/1e3,np.abs(signal_comp[1]),label='Comp $t_1$')
ax.set_title('Rx compressed signals')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[2]
ax.plot(ranks/1e3,
        np.abs(cfar_MTI_sc),
        label='MTI Signal')

ax.plot(ranks/1e3,
        np.abs(th_cfar_MTI_sc),
        label='MTI Threshold')
ax.set_title('MTI SC')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[3]
ax.plot(ranks/1e3,
        np.abs(dif_th_MTI_sc)
        )

ax.annotate(text='\n'.join([f'Target {ptr+1} : {targets_rank[ptr]:.2f} km' for ptr in range(len(targets_rank))]),
                xy=(0.05,0.8),
                xycoords='figure fraction',
                bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})

ax.set_title('MTI_SC signal')
ax.set_ylabel('Value')
ax.set_xlabel('Range [km]')
ax.grid(True)

#%% STI SC

# Se resta la segunda fila a la primera

STI_matrix_sc = np.array([1,1])
cfar_STI_sc = np.inner(signal_comp[0:2].T,STI_matrix_sc)

gain_sti = 4

# convolucionar y quitar colas de convolucion
th_cfar_STI_sc = gain_sti*fastconv(np.abs(cfar_STI_sc),h_cfar)
th_cfar_STI_sc = th_cfar_STI_sc[len(h_cfar)//2:len(cfar_STI_sc)+len(h_cfar)//2]

dif_th_STI_sc = np.sign(np.abs(cfar_STI_sc) - np.abs(th_cfar_STI_sc))
dif_th_STI_sc = np.diff(dif_th_STI_sc,n=1)
dif_th_STI_sc = np.concatenate((dif_th_STI_sc,np.zeros(len(ranks)-len(dif_th_STI_sc))))

targets_rank = ranks[dif_th_STI_sc==2]/1e3

# Plot

fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
fig.suptitle('STI Simple Cancelador')
fig.subplots_adjust(left=0.3)

ax = axes[0]
ax.plot(ranks/1e3,np.abs(radar_signal[0]),label='Rx $t_0$')
ax.plot(ranks/1e3,np.abs(radar_signal[1]),label='Rx $t_1$')
ax.set_title('Rx Raw signals')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(signal_comp[0]),label='Comp $t_0$')
ax.plot(ranks/1e3,np.abs(signal_comp[1]),label='Comp $t_1$')
ax.set_title('Rx compressed signals')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[2]
ax.plot(ranks/1e3,
        np.abs(cfar_STI_sc),
        label='STI Signal')

ax.plot(ranks/1e3,
        np.abs(th_cfar_STI_sc),
        label='STI Threshold')
ax.set_title('STI SC')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[3]
ax.plot(ranks/1e3,
        np.abs(dif_th_STI_sc)
        )

ax.annotate(text='\n'.join([f'Target {ptr+1} : {targets_rank[ptr]:.2f} km' for ptr in range(len(targets_rank))]),
                xy=(0.05,0.8),
                xycoords='figure fraction',
                bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})

ax.set_title('STI_SC signal')
ax.set_ylabel('Value')
ax.set_xlabel('Range [km]')
ax.grid(True)

#%% MTI SC

# Se resta la segunda fila a la primera

MTI_matrix_dc = np.array([1,-2,1])
cfar_MTI_dc = np.inner(signal_comp[0:3].T,MTI_matrix_dc)

gain_sti = 4

# convolucionar y quitar colas de convolucion
th_cfar_MTI_dc = gain_sti*fastconv(np.abs(cfar_MTI_dc),h_cfar)
th_cfar_MTI_dc = th_cfar_MTI_dc[len(h_cfar)//2:len(cfar_MTI_dc)+len(h_cfar)//2]

dif_th_MTI_dc = np.sign(np.abs(cfar_MTI_dc) - np.abs(th_cfar_MTI_dc))
dif_th_MTI_dc = np.diff(dif_th_MTI_dc,n=1)
dif_th_MTI_dc = np.concatenate((dif_th_MTI_dc,np.zeros(len(ranks)-len(dif_th_MTI_dc))))

targets_rank = ranks[dif_th_MTI_dc==2]/1e3

# Plot

fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
fig.suptitle('MTI Doble Cancelador')
fig.subplots_adjust(left=0.3)

ax = axes[0]
ax.plot(ranks/1e3,np.abs(radar_signal[0]),label='Rx $t_0$')
ax.plot(ranks/1e3,np.abs(radar_signal[1]),label='Rx $t_1$')
ax.plot(ranks/1e3,np.abs(radar_signal[2]),label='Rx $t_2$')
ax.set_title('Rx Raw signals')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(signal_comp[0]),label='Comp $t_0$')
ax.plot(ranks/1e3,np.abs(signal_comp[1]),label='Comp $t_1$')
ax.plot(ranks/1e3,np.abs(signal_comp[2]),label='Comp $t_2$')
ax.set_title('Rx compressed signals')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[2]
ax.plot(ranks/1e3,
        np.abs(cfar_MTI_dc),
        label='MTI Signal')

ax.plot(ranks/1e3,
        np.abs(th_cfar_MTI_dc),
        label='MTI Threshold')
ax.set_title('MTI DC')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[3]
ax.plot(ranks/1e3,
        np.abs(dif_th_MTI_dc)
        )

ax.annotate(text='\n'.join([f'Target {ptr+1} : {targets_rank[ptr]:.2f} km' for ptr in range(len(targets_rank))]),
                xy=(0.05,0.8),
                xycoords='figure fraction',
                bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})

ax.set_title('MTI_dc signal')
ax.set_ylabel('Value')
ax.set_xlabel('Range [km]')
ax.grid(True)

#%%
from matplotlib import cm

cfar_mti_doppler = np.zeros(signal_comp.shape,dtype=np.complex64)

for t in range(1,Np):
    cfar_mti_doppler[t] = signal_comp[t]-signal_comp[t-1]

FD = np.exp(-1j*2*np.pi*np.outer(np.arange(1,Np+1),np.arange(1,Np+1).T)/(Np+1))
#FD = np.exp(-1j*2*np.pi*np.outer(np.arange(1,Np+1),np.arange(1,Np+1).T)/Np)
filtered = (cfar_mti_doppler.T@FD).T

#FD = np.exp(-1j*2*np.pi*PRI*np.arange(Np))
#filtered = (cfar_mti_doppler.T * FD).T

#filtered = (signal_comp.T@FD).T

vel_vect = -np.linspace(-vu_ms/2,vu_ms/2,Np,endpoint=True) # El signo menos es solo para plotear

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(10,8))
fig.suptitle('Producto Doppler')
x = vel_vect
y = ranks/1e3
X, Y = np.meshgrid(x, y)
Z = np.abs(fftshift(filtered,axes=0)).T
surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
ax.set_xlabel('Velocity [m/s]')
ax.set_ylabel('Range [km]')
ax.set_zlabel('Doppler Product')
fig.colorbar(surf, shrink=0.5)