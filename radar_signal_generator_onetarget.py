# -*- coding: utf-8 -*-
"""

@author: fdadam
"""


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import spectrogram
from matplotlib.gridspec import GridSpec
import pandas as pd

#%%

def calc_noise_power(signal,snr):
    signal_power = np.mean(np.abs(signal)**2/rx_z)
    signal_power_db = 10* np.log10(signal_power)
    noise_power_db = signal_power_db - snr
    noise_power = 10**(noise_power_db/10)
    
    return noise_power

#%%

c = 3e8 # speed of light [m/s]
fc = 1.3e9 # Carrier freq
fs = 10e6 # Sampling freq
Np = 10 # Intervalos de sampling
Nint = 10
NPRIs = Nint*Np
ts = 1/fs


Te = 5e-6 # recovery Time 
Tp = 10e-6 # Pulse Width
BW = 2e6 # Chirp bandwidth
PRF = 1500 # Pulse repetition Frequency [Hz]

wlen = c/fc # Wavelength
kwave = 2*np.pi/wlen
PRI = PRF**(-1) # Pulse repetition interval [s]
ru = (c*(PRI-Tp-Te))/2 # Unambigous Range
vu = wlen*PRF/2 # Unambigous Velocity

rank_min = (Tp/2+Te)*c/2 # Minimum Range
rank_max = 30e3 # Rango ficticio, deería ser el ambiguo pero se relentiza la simulación
#rank_max = ru
rank_res = ts*c/2
tmax = 2*rank_max/c 

Npts = int(tmax/ts)
t = np.linspace(-tmax/2,tmax/2,Npts)
ranks = np.linspace(rank_res,rank_max,Npts)
f = fftfreq(Npts,ts)

print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu:.2f} m/s')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')


#%%


# Signals

tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2)
tx_rect = np.where(np.abs(t)<=Tp/2,1,0)
tx_chirp = tx_rect*tx_chirp
tx_chirp_f = fft(tx_chirp,norm='ortho')

# Matched Filter

matched_filter = np.conj(np.flip(tx_chirp))
#matched_filter = np.exp(-1j*np.pi*BW/Tp * t**2)
matched_filter_f = fft(matched_filter,norm='ortho')

# Radar Characteristics

Gt_db = 60
Gt = 10**(Gt_db/10)
Pt = 5e3 #W
A_ant = 2*np.pi*(5**2)
Ae = A_ant*0.6
rx_z = 75

# Target Characteristics

r0 = 10e3
vel = 100 # m/s
#target_rcs = 100
snr = 10 #db


np.random.seed(0)

radar_signal = []

k = 1.380649e-23 # Boltzmann
#Ts = 300 # K
Ts = 150 + 150 + 300
rx_noise_power = k*BW*Ts
echo_time = np.linspace(0,tmax,Npts) # Only to plot

"""
Rx Temperature

https://www.antenna-theory.com/basics/temperature.php
https://www.sciencedirect.com/topics/earth-and-planetary-sciences/noise-temperature
https://www.ece.mcmaster.ca/faculty/nikolova/antenna_dload/current_lectures/L07_Noise.pdf
https://www.satsig.net/noise.htm
http://www.marinesatellitesystems.com/index.php?page_id=888
"""

for pri_ptr in range(Np):
    rx_signal_noisy = np.zeros(Npts,dtype=np.complex64)
    for int_ptr in range(Nint):
        vel_kmh = 200
        vel_ms = vel_kmh/3.6
        target_rank = pri_ptr*PRI*vel_ms + r0
        
        target_t_shift = 2*target_rank/c
        # Compression
        
        echo_chirp_f = tx_chirp_f * np.exp(-1j*2*np.pi*f*target_t_shift) * np.exp(-1j*kwave*(2*target_rank))
        echo_chirp = ifft(echo_chirp_f)
        echo_chirp = np.concatenate((echo_chirp[Npts//2::],np.zeros(Npts//2)))
        
        
        echo_comp_f = (echo_chirp_f*matched_filter_f)
        echo_comp = ifft(echo_comp_f)
        
        noise_power = calc_noise_power(echo_chirp,snr)
        
        if(pri_ptr==0):
            clutter_noise = np.random.normal(loc = 0,scale = np.sqrt(noise_power),size = len(echo_time)*2)
            clutter_noise = clutter_noise[:len(clutter_noise)//2] + 1j*clutter_noise[len(clutter_noise)//2::]
        
        echo_chirp_noisy = echo_chirp + clutter_noise
        
        rx_rect = np.where(ranks>rank_min,1,0)
        rx_signal = echo_chirp_noisy * (Pt*Gt*Ae)/((4*np.pi**2)*(ranks**4)) 
        
        rx_noise = np.random.normal(loc = 0,scale = np.sqrt(rx_noise_power),size = len(echo_time)*2)
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
        #ax.set_xlim(0,150)
        ax.grid(True)
        
        ax = fig.add_subplot(gs[0,1])
        ax.plot(f[0:Npts//2]/1e6,np.real(tx_chirp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.imag(tx_chirp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.abs(tx_chirp_f[0:Npts//2]))
        ax.set_xlabel('Freq [MHz]')
        ax.set_xlim(0,3*BW/1e6)
        ax.grid(True)
        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(echo_time*1e6,np.real(echo_chirp))
        ax.plot(echo_time*1e6,np.imag(echo_chirp))
        ax.plot(echo_time*1e6,np.abs(echo_chirp))
        ax.set_xlabel('Time [$\mu$s]')
        #ax.set_xlim(0,150)
        ax.grid(True)
        
        ax = fig.add_subplot(gs[1,1])
        ax.plot(f[0:Npts//2]/1e6,np.real(echo_chirp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.imag(echo_chirp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.abs(echo_chirp_f[0:Npts//2]))
        ax.set_xlabel('Freq [MHz]')
        ax.set_xlim(0,3*BW/1e6)
        ax.grid(True)
        
        ax = fig.add_subplot(gs[2,0])
        ax.plot(echo_time*1e6,np.real(echo_comp))
        ax.plot(echo_time*1e6,np.imag(echo_comp))
        ax.plot(echo_time*1e6,np.abs(echo_comp))
        ax.set_xlabel('Time [$\mu$s]')
        #ax.set_xlim(0,150)
        ax.grid(True)
        
        ax = fig.add_subplot(gs[2,1])
        ax.plot(f[0:Npts//2]/1e6,np.real(echo_comp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.imag(echo_comp_f[0:Npts//2]))
        ax.plot(f[0:Npts//2]/1e6,np.abs(echo_comp_f[0:Npts//2]))
        ax.set_xlabel('Freq [MHz]')
        ax.set_xlim(0,3*BW/1e6)
        ax.grid(True)
        
        ax = fig.add_subplot(gs[3,:])
        ax.plot(ranks/1e3,np.abs(echo_comp))
        ax.set_xlabel('Range [km]')
        #ax.set_xlim(0,150)
        ax.grid(True)
        
        fig = plt.figure(layout="tight",figsize=(16,9))
        gs = GridSpec(4, 1, figure=fig)
        
        ax = fig.add_subplot(gs[0,:])
        ax.plot(ranks/1e3,np.abs(echo_chirp))
        ax.set_xlabel('Range [km]')
        #ax.set_xlim(0,30)
        ax.set_title('Echo Chirp')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[1,:],sharex=ax)
        ax.plot(ranks/1e3,np.abs(clutter_noise))
        ax.set_xlabel('Range [km]')
        #ax.set_xlim(0,30)
        ax.set_title('Clutter Noise')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[2,:],sharex=ax)
        ax.plot(ranks/1e3,np.abs(echo_chirp_noisy))
        ax.set_xlabel('Range [km]')
        #ax.set_xlim(0,30)
        ax.set_title('Echo Chirp & Clutter')
        ax.grid(True)
        
        ax = fig.add_subplot(gs[3,:],sharex=ax)
        ax.plot(ranks/1e3,np.abs(rx_signal_noisy))
        ax.set_xlabel('Range [km]')
        #ax.set_xlim(0,30)
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


