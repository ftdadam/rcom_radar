# -*- coding: utf-8 -*-
"""

@author: fdadam
"""
#%% Libs and functions
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import spectrogram
from matplotlib.gridspec import GridSpec
import pandas as pd

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

np.random.seed(0)

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

print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu_ms:.2f} m/s')
print(f'Unambiguous Velocity. Vu = {vu_kmh:.2f} km/h')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')

#%% Signals

# Independant Variables

Npts = int(tmax/ts) # Simulation Points
#t = np.linspace(0,tmax,Npts) # Time Vector
t = np.linspace(-tmax/2,tmax/2,Npts) # Time Vector
ranks = np.linspace(rank_res,rank_max,Npts) # Range Vector
f = fftfreq(Npts,ts) # Freq Vector

# Tx Signal

tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2) # Tx Linear Chiprs (t)
#tx_rect = np.where(np.abs(t)<=Tp,1,0) # Rect Function
tx_rect = np.where(np.abs(t)<=Tp/2,1,0) # Rect Function
tx_chirp = tx_rect*tx_chirp # Tx Chirp Rectangular
tx_chirp_f = fft(tx_chirp,norm='ortho') # Tx Chirp (f)

# Matched Filter

matched_filter = np.conj(np.flip(tx_chirp))
#matched_filter = np.exp(-1j*np.pi*BW/Tp * t**2)
matched_filter_f = fft(matched_filter,norm='ortho')
#matched_filter_f = matched_filter_f * np.exp(1j*2*np.pi*f*Tp/2)

# Radar Characteristics

Gt_db = 60
Gt = 10**(Gt_db/10)
Pt = 5e3 #W
A_ant = 2*np.pi*(5**2)
Ae = A_ant*0.6


#%% Rx Noise


# =============================================================================
# Rx Temperature:
# https://www.antenna-theory.com/basics/temperature.php
# https://www.sciencedirect.com/topics/earth-and-planetary-sciences/noise-temperature
# https://www.ece.mcmaster.ca/faculty/nikolova/antenna_dload/current_lectures/L07_Noise.pdf
# https://www.satsig.net/noise.htm
# http://www.marinesatellitesystems.com/index.php?page_id=888
# =============================================================================

Ts = 300 # K
rx_noise_power = k*BW*Ts

# Rx noise is calculated each 

    
#%% Clutter Noise 
noise_power_db = -80 #[dBm]
noise_power = 10**(noise_power_db/10)
clutter_noise = np.random.normal(loc = 0,scale = np.sqrt(noise_power),size = len(t)*2)
clutter_noise = clutter_noise[:len(clutter_noise)//2] + 1j*clutter_noise[len(clutter_noise)//2::]

# Clutter noise could change between adqusitions (grass or tree leaves moving, dust, insects, clouds, etc)
# In this exercise, to avoid filtering, low power fixed clutter is assumed

#%% # Target Characteristics

class Target:
  def __init__(self, init_rank_km, vel_ms,relative_gain):
    self.init_rank_km = init_rank_km
    self.init_rank_m = init_rank_km*1e3
    self.vel_ms = vel_ms
    self.vel_kmh = vel_ms*3.6
    #self.snr_db = snr_db
    self.relative_gain = relative_gain

# Range [km], Vel [km/h], Relative Power
t1 = Target(10,50,1)
t2 = Target(15,-40,4)
t3 = Target(18,0,27)
t4 = Target(20,30,32)

targets = [t1,t2,t3,t4]

#%% Radar Signal Generation

radar_signal = []


for pri_ptr in range(Np):
    
    rx_signal_noisy = np.zeros(Npts,dtype=np.complex64)
    
    for int_ptr in range(Nint):
        
        echo_chirp_f  = np.zeros(Npts,dtype=np.complex64) # Reset Chirp each integration
        
        for target in targets:
            target_rank = pri_ptr*PRI*target.vel_ms + target.init_rank_m
            target_t_shift = 2*target_rank/c
            echo_chirp_f += target.relative_gain * tx_chirp_f * np.exp(-1j*2*np.pi*f*target_t_shift) * np.exp(-1j*kwave*(2*target_rank))
            
        echo_chirp = ifft(echo_chirp_f)
        echo_chirp = np.concatenate((echo_chirp[Npts//2::],echo_chirp[0:Npts//2]))

        echo_comp_f = (echo_chirp_f*matched_filter_f)
        echo_comp = ifft(echo_comp_f)

        echo_chirp_noisy = echo_chirp + clutter_noise
        
        rx_rect = np.where(ranks>rank_min,1,0)
        rx_signal = echo_chirp_noisy * (Pt*Gt*Ae)/((4*np.pi**2)*(ranks**4)) 
        
        rx_noise = np.random.normal(loc = 0,scale = np.sqrt(rx_noise_power),size = len(t)*2)
        rx_noise = rx_noise[:len(rx_noise)//2] + 1j*rx_noise[len(rx_noise)//2::]
        
        rx_signal_noisy += rx_rect * rx_signal + rx_rect * rx_noise
        
        #rx_signal_noisy_db = 10*np.log10(np.abs(rx_signal_noisy))
    radar_signal.append(rx_signal_noisy)
    
    if(pri_ptr == 0):
        fig = plt.figure(layout="tight",figsize=(16,9))
        gs = GridSpec(4, 2, figure=fig)
        
        ax = fig.add_subplot(gs[0,0])
        ax.plot(t*1e6,np.real(tx_chirp))
        ax.plot(t*1e6,np.imag(tx_chirp))
        ax.plot(t*1e6,np.abs(tx_chirp))
        ax.plot(t*1e6,tx_rect,'k--',lw=2)
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_title('Tx Chirp')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[0,1])
        ax.plot(f[0:Npts//2]/1e6,np.real(tx_chirp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.imag(tx_chirp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.abs(tx_chirp_f[0:Npts//2]))
        ax.set_xlabel('Freq [MHz]')
        ax.set_xlim(0,3*BW/1e6)
        ax.set_title('Tx Chirp')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(t*1e6,np.real(echo_chirp))
        ax.plot(t*1e6,np.imag(echo_chirp))
        ax.plot(t*1e6,np.abs(echo_chirp))
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_title('Echo Chirp')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[1,1])
        ax.plot(f[0:Npts//2]/1e6,np.real(echo_chirp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.imag(echo_chirp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.abs(echo_chirp_f[0:Npts//2]))
        ax.set_xlabel('Freq [MHz]')
        ax.set_xlim(0,3*BW/1e6)
        ax.set_title('Echo Chirp')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[2,0])
        ax.plot(t*1e6,np.real(echo_comp))
        ax.plot(t*1e6,np.imag(echo_comp))
        ax.plot(t*1e6,np.abs(echo_comp))
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_title('Echo Comp Noiseless')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[2,1])
        ax.plot(f[0:Npts//2]/1e6,np.real(echo_comp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.imag(echo_comp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.abs(echo_comp_f[0:Npts//2]))
        ax.set_xlabel('Freq [MHz]')
        ax.set_xlim(0,3*BW/1e6)
        ax.set_title('Echo Comp Noiseless')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[3,:])
        ax.plot(ranks/1e3,np.abs(echo_comp))
        ax.set_xlabel('Range [km]')
        ax.grid(True)
        
        fig = plt.figure(layout="tight",figsize=(16,9))
        gs = GridSpec(4, 1, figure=fig)
        
        ax = fig.add_subplot(gs[0,:])
        ax.plot(ranks/1e3,np.abs(echo_chirp))
        ax.set_xlabel('Range [km]')
        ax.set_title('Echo Chirp')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[1,:],sharex=ax)
        ax.plot(ranks/1e3,np.abs(clutter_noise))
        ax.set_xlabel('Range [km]')
        ax.set_title('Clutter Noise')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[2,:],sharex=ax)
        ax.plot(ranks/1e3,np.abs(echo_chirp_noisy))
        ax.set_xlabel('Range [km]')
        ax.set_title('Echo Chirp & Clutter')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[3,:],sharex=ax)
        ax.plot(ranks/1e3,np.abs(rx_signal_noisy))
        ax.set_xlabel('Range [km]')
        ax.set_title('Rx Signal')
        ax.grid(True)
        
    
radar_signal = np.stack(radar_signal,axis=0)
#%%
data = {'real':radar_signal.flatten().real,
        'imag':radar_signal.flatten().imag}

signal = pd.DataFrame(data=data)

signal.to_csv('./signal.csv')

#plt.figure()
#plt.plot(fastconv(matched_filter,rx_signal_noisy))

#%%


