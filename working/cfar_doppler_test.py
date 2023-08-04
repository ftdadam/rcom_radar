# -*- coding: utf-8 -*-
"""

@author: fdadam
"""

#%% Libs and functions

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from scipy.fft import fft, ifft, fftshift, fftfreq

def chirp_signal(direction,BW,tp,fs):
    if(direction == 'upchirp'):
        K = BW/tp
    elif(direction == 'downchirp'):
        K = -BW/tp
    
    time = np.arange(-tp/2,tp/2,1/fs)
    
    chirp_out = np.exp(1j*(np.pi*K*time**2))
    return chirp_out

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

def getTime(s):
    return 1/(fs*np.arange(len(s)))

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


#%%

# Vectors

rank_min = (Tp/2+Te)*c/2 # Minimum Range
rank_max = 30e3 # Rango ficticio, deería ser el ambiguo pero se relentiza la simulación
#rank_max = ru
rank_res = ts*c/2
tmax = 2*rank_max/c 

Npts = int(tmax/ts)
t = np.linspace(-tmax/2,tmax/2,Npts)
ranks = np.linspace(rank_res,rank_max,Npts)
f = fftfreq(Npts,ts)

# Signals

tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2)
tx_rect = np.where(np.abs(t)<=Tp/2,1,0)
tx_chirp = tx_rect*tx_chirp
tx_chirp_f = fft(tx_chirp,norm='ortho')

# Matched Filter

matched_filter = np.conj(np.flip(tx_chirp))
#matched_filter = np.exp(-1j*np.pi*BW/Tp * t**2)
matched_filter_f = fft(matched_filter,norm='ortho')


#%% Signals

rxSignal = pd.read_csv('signal.csv',index_col=None)
rxSignal = np.array(rxSignal['real']+1j*rxSignal['imag'])
rxSignal = rxSignal.reshape(Np,-1)

print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu:.2f} m/s')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')

#%% Plot Signals

fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True)

fig.suptitle('Received Signal')

ax = axes[0]
ax.plot(ranks/1e3,rxSignal[0])
ax.set_ylabel('Amplitude [V]')
ax.set_xlabel('Range [km]')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(rxSignal[0]))
ax.set_ylabel('Abs Amplitude [V]')
ax.set_xlabel('Range [km]')
ax.grid(True)

#%% Plot Compressed Signals
signal_comp = []
for t in range(len(rxSignal)):
    # convolucionar y quitar colas de convolucion
    comp = fastconv(rxSignal[t],matched_filter)[len(matched_filter)//2:len(rxSignal[t])+len(matched_filter)//2]
    signal_comp.append(comp)
signal_comp = np.stack(signal_comp,axis=0)

fig, axes = plt.subplots(2,1,figsize=(10,10))

ax = axes[0]
ax.plot(ranks/1e3,np.abs(rxSignal[0]))
ax.set_ylabel('Abs value')
ax.set_xlabel('Rx Raw signal')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(signal_comp[0]))
ax.set_ylabel('Abs value')
ax.set_xlabel('Rx compressed signal')
ax.grid(True)

#%% CFAR Window

n_gap = 15
n_ref = 100
v_ref = 1
h_cfar_gain = (1/(n_ref*2*v_ref))
h_cfar = np.concatenate((np.repeat(v_ref,n_ref),
                         np.repeat(0,n_gap*2+1),
                         np.repeat(v_ref,n_ref)))*h_cfar_gain

# Plot

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.step(range(len(h_cfar)),h_cfar,marker='.')
ax.set_xlabel('Sample')
ax.set_ylabel('CFAR window Value')
ax.grid(True)

#%% mti simple cancelador (mti_sc)

def calc_th_mti_sc(gain_mti,cfar_mti_sc,ranks):
    # convolucionar y quitar colas de convolucion
    th_cfar_mti_sc = gain_mti*fastconv(np.abs(cfar_mti_sc),h_cfar)
    th_cfar_mti_sc = th_cfar_mti_sc[len(h_cfar)//2:len(cfar_mti_sc)+len(h_cfar)//2]
        
    dif_th_mti_sc = np.sign(np.abs(cfar_mti_sc) - np.abs(th_cfar_mti_sc))
    dif_th_mti_sc = np.diff(dif_th_mti_sc,n=1)
    dif_th_mti_sc = np.concatenate((dif_th_mti_sc,np.zeros(len(ranks)-len(dif_th_mti_sc))))
    
    #targets_rank = ranks/1e3
    targets_rank = ranks[dif_th_mti_sc==2]/1e3
    
    return (th_cfar_mti_sc,dif_th_mti_sc,targets_rank)

# (4,30) 3.203125
init_gain_mti = 2

# Se resta la segunda fila a la primera

mti_matrix_sc = np.array([1,-1])
cfar_mti_sc = np.inner(signal_comp[0:2].T,mti_matrix_sc)

# Plot

fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
fig.suptitle('MTI Simple Cancelador')
fig.subplots_adjust(left=0.3)

# vertical slider para la gain.

axfreq = fig.add_axes([0.1, 0.1, 0.0225, 0.6])
gain_slider = Slider(
    ax=axfreq,
    label='MTI Gain',
    valmin=1,
    valmax=5,
    valinit=init_gain_mti,
    orientation="vertical"
)

# Update plot with slider

def update(val):
    global target_anotation
    th_cfar_mti_sc = calc_th_mti_sc(gain_slider.val,cfar_mti_sc,ranks)[0]
    dif_th_mti_sc = calc_th_mti_sc(gain_slider.val,cfar_mti_sc,ranks)[1]
    targets_rank = calc_th_mti_sc(gain_slider.val,cfar_mti_sc,ranks)[2]
    
    line_th_cfar_mti_sc.set_ydata(np.abs(th_cfar_mti_sc))
    line_dif_th_mti_sc.set_ydata(np.abs(dif_th_mti_sc))
    target_anotation.remove()
    if(len(targets_rank) > 5 ):
    
        target_anotation = ax.annotate(text=f'{len(targets_rank)} Targets',
                                       xy=(0.05,0.8),
                                       xycoords='figure fraction',
                                       bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
    else:
        target_anotation = ax.annotate(text='\n'.join([f'Target {ptr+1} : {targets_rank[ptr]:.2f} km' for ptr in range(len(targets_rank))]),
                                       xy=(0.05,0.8),
                                       xycoords='figure fraction',
                                       bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
    
    #print(targets_rank)
    fig.canvas.draw_idle()

gain_slider.on_changed(update)

# Plot

ax = axes[0]
ax.plot(ranks/1e3,np.abs(rxSignal[0]),label='Rx $t_0$')
ax.plot(ranks/1e3,np.abs(rxSignal[1]),label='Rx $t_1$')
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
        np.abs(cfar_mti_sc),
        label='MTI Signal')

line_th_cfar_mti_sc, = ax.plot(ranks/1e3,
                               np.abs(calc_th_mti_sc(init_gain_mti,cfar_mti_sc,ranks)[0]),
                               label='MTI Threshold')
ax.set_title('MTI SC')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
ax.grid(True)

ax = axes[3]
line_dif_th_mti_sc, = ax.plot(ranks/1e3,
                              np.abs(calc_th_mti_sc(init_gain_mti,cfar_mti_sc,ranks)[1])
                              )

targets_rank = calc_th_mti_sc(init_gain_mti,cfar_mti_sc,ranks)[2]

target_anotation = ax.annotate(text=f'{len(targets_rank)} Targets',
                               xy=(0.05,0.8),
                               xycoords='figure fraction',
                               bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
ax.set_title('MTI_SC signal')
ax.set_ylabel('Value')
ax.set_xlabel('Range [km]')
ax.grid(True)

# #%% sti simple cancelador (sti_sc)

# def calc_th_sti_sc(gain_sti,cfar_sti_sc):
#     # convolucionar y quitar colas de convolucion
#     th_cfar_sti_sc = gain_sti*fastconv(np.abs(cfar_sti_sc),h_cfar)
#     th_cfar_sti_sc = th_cfar_sti_sc[len(h_cfar)//2:len(cfar_sti_sc)+len(h_cfar)//2]
    
#     dif_th_sti_sc = np.sign(np.abs(cfar_sti_sc) - np.abs(th_cfar_sti_sc))
#     dif_th_sti_sc = np.diff(dif_th_sti_sc,n=1)
#     targets_rank = getRank(dif_th_sti_sc)
#     targets_rank = targets_rank[dif_th_sti_sc==2]
    
#     return (th_cfar_sti_sc,dif_th_sti_sc,targets_rank)

# # (4,30) 4.90625
# init_gain_sti = 2

# # Se resta la segunda fila a la primera

# sti_matrix_sc = np.array([1,1])
# cfar_sti_sc = np.inner(signal_comp[0:2].T,sti_matrix_sc)

# # Plot

# fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
# fig.suptitle('STI Simple Cancelador')
# fig.subplots_adjust(left=0.3)

# # vertical slider para la gain.

# axfreq = fig.add_axes([0.1, 0.1, 0.0225, 0.6])
# gain_slider = Slider(
#     ax=axfreq,
#     label='sti Gain',
#     valmin=1,
#     valmax=5,
#     valinit=init_gain_sti,
#     orientation="vertical"
# )

# # Update plot with slider

# def update(val):
#     global target_anotation
#     th_cfar_sti_sc = calc_th_sti_sc(gain_slider.val,cfar_sti_sc)[0]
#     dif_th_sti_sc = calc_th_sti_sc(gain_slider.val,cfar_sti_sc)[1]
#     targets_rank = calc_th_sti_sc(gain_slider.val,cfar_sti_sc)[2]
    
#     line_th_cfar_sti_sc.set_ydata(np.abs(th_cfar_sti_sc))
#     line_dif_th_sti_sc.set_ydata(np.abs(dif_th_sti_sc))
#     target_anotation.remove()
#     if(len(targets_rank) > 5 ):
    
#         target_anotation = ax.annotate(text=f'{len(targets_rank)} Targets',
#                                        xy=(0.05,0.8),
#                                        xycoords='figure fraction',
#                                        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
#     else:
#         target_anotation = ax.annotate(text='\n'.join([f'Target {ptr+1} : {targets_rank[ptr]:.2f} km' for ptr in range(len(targets_rank))]),
#                                        xy=(0.05,0.8),
#                                        xycoords='figure fraction',
#                                        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
    
#     #print(targets_rank)
#     fig.canvas.draw_idle()

# gain_slider.on_changed(update)

# # Plot

# ax = axes[0]
# ax.plot(getRank(rxSignal[0]),np.abs(rxSignal[0]),label='Rx $t_0$')
# ax.plot(getRank(rxSignal[0]),np.abs(rxSignal[1]),label='Rx $t_1$')
# ax.set_title('Rx Raw signals')
# ax.set_ylabel('Value')
# ax.legend(loc='upper right')
# ax.grid(True)

# ax = axes[1]
# ax.plot(getRank(signal_comp[0]),np.abs(signal_comp[0]),label='Comp $t_0$')
# ax.plot(getRank(signal_comp[1]),np.abs(signal_comp[1]),label='Comp $t_0$')
# ax.set_title('Rx compressed signals')
# ax.set_ylabel('Value')
# ax.legend(loc='upper right')
# ax.grid(True)

# ax = axes[2]
# ax.plot(getRank(cfar_sti_sc),
#         np.abs(cfar_sti_sc),
#         label='sti Signal')

# line_th_cfar_sti_sc, = ax.plot(getRank(calc_th_sti_sc(init_gain_sti,cfar_sti_sc)[0]),
#                                np.abs(calc_th_sti_sc(init_gain_sti,cfar_sti_sc)[0]),
#                                label='sti Threshold')
# ax.set_title('sti SC')
# ax.set_ylabel('Value')
# ax.legend(loc='upper right')
# ax.grid(True)

# ax = axes[3]
# line_dif_th_sti_sc, = ax.plot(getRank(calc_th_sti_sc(init_gain_sti,cfar_sti_sc)[1]),
#                               np.abs(calc_th_sti_sc(init_gain_sti,cfar_sti_sc)[1])
#                               )

# targets_rank = calc_th_sti_sc(init_gain_sti,cfar_sti_sc)[2]

# target_anotation = ax.annotate(text=f'{len(targets_rank)} Targets',
#                                xy=(0.05,0.8),
#                                xycoords='figure fraction',
#                                bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
# ax.set_title('sti_SC signal')
# ax.set_ylabel('Value')
# ax.set_xlabel('Range [km]')
# ax.grid(True)

# #%% mti doble cancelador (mti_dc)

# def calc_th_mti_dc(gain_mti,cfar_mti_dc):
#     # convolucionar y quitar colas de convolucion
#     th_cfar_mti_dc = gain_mti*fastconv(np.abs(cfar_mti_dc),h_cfar)
#     th_cfar_mti_dc = th_cfar_mti_dc[len(h_cfar)//2:len(cfar_mti_dc)+len(h_cfar)//2]
    
#     dif_th_mti_dc = np.sign(np.abs(cfar_mti_dc) - np.abs(th_cfar_mti_dc))
#     dif_th_mti_dc = np.diff(dif_th_mti_dc,n=1)
#     targets_rank = getRank(dif_th_mti_dc)
#     targets_rank = targets_rank[dif_th_mti_dc==2]
    
#     return (th_cfar_mti_dc,dif_th_mti_dc,targets_rank)

# # (4,30) 3.203125
# init_gain_mti = 2

# # Se resta la segunda fila a la primera

# mti_matrix_sc = np.array([1,-2,1])
# cfar_mti_dc = np.inner(signal_comp[0:3].T,mti_matrix_sc)

# # Plot

# fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
# fig.suptitle('MTI Doble Cancelador')
# fig.subplots_adjust(left=0.3)

# # vertical slider para la gain.

# axfreq = fig.add_axes([0.1, 0.1, 0.0225, 0.6])
# gain_slider = Slider(
#     ax=axfreq,
#     label='MTI Gain',
#     valmin=1,
#     valmax=5,
#     valinit=init_gain_mti,
#     orientation="vertical"
# )

# # Update plot with slider

# def update(val):
#     global target_anotation
#     th_cfar_mti_dc = calc_th_mti_dc(gain_slider.val,cfar_mti_dc)[0]
#     dif_th_mti_dc = calc_th_mti_dc(gain_slider.val,cfar_mti_dc)[1]
#     targets_rank = calc_th_mti_dc(gain_slider.val,cfar_mti_dc)[2]
    
#     line_th_cfar_mti_dc.set_ydata(np.abs(th_cfar_mti_dc))
#     line_dif_th_mti_dc.set_ydata(np.abs(dif_th_mti_dc))
#     target_anotation.remove()
#     if(len(targets_rank) > 5 ):
    
#         target_anotation = ax.annotate(text=f'{len(targets_rank)} Targets',
#                                        xy=(0.05,0.8),
#                                        xycoords='figure fraction',
#                                        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
#     else:
#         target_anotation = ax.annotate(text='\n'.join([f'Target {ptr+1} : {targets_rank[ptr]:.2f} km' for ptr in range(len(targets_rank))]),
#                                        xy=(0.05,0.8),
#                                        xycoords='figure fraction',
#                                        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
    
#     #print(targets_rank)
#     fig.canvas.draw_idle()

# gain_slider.on_changed(update)

# # Plot

# ax = axes[0]
# ax.plot(getRank(rxSignal[0]),np.abs(rxSignal[0]),label='Rx $t_0$')
# ax.plot(getRank(rxSignal[0]),np.abs(rxSignal[1]),label='Rx $t_1$')
# ax.set_title('Rx Raw signals')
# ax.set_ylabel('Value')
# ax.legend(loc='upper right')
# ax.grid(True)

# ax = axes[1]
# ax.plot(getRank(signal_comp[0]),np.abs(signal_comp[0]),label='Comp $t_0$')
# ax.plot(getRank(signal_comp[1]),np.abs(signal_comp[1]),label='Comp $t_0$')
# ax.set_title('Rx compressed signals')
# ax.set_ylabel('Value')
# ax.legend(loc='upper right')
# ax.grid(True)

# ax = axes[2]
# ax.plot(getRank(cfar_mti_dc),
#         np.abs(cfar_mti_dc),
#         label='MTI Signal')

# line_th_cfar_mti_dc, = ax.plot(getRank(calc_th_mti_dc(init_gain_mti,cfar_mti_dc)[0]),
#                                np.abs(calc_th_mti_dc(init_gain_mti,cfar_mti_dc)[0]),
#                                label='MTI Threshold')
# ax.set_title('MTI SC')
# ax.set_ylabel('Value')
# ax.legend(loc='upper right')
# ax.grid(True)

# ax = axes[3]
# line_dif_th_mti_dc, = ax.plot(getRank(calc_th_mti_dc(init_gain_mti,cfar_mti_dc)[1]),
#                               np.abs(calc_th_mti_dc(init_gain_mti,cfar_mti_dc)[1])
#                               )

# targets_rank = calc_th_mti_dc(init_gain_mti,cfar_mti_dc)[2]

# target_anotation = ax.annotate(text=f'{len(targets_rank)} Targets',
#                                xy=(0.05,0.8),
#                                xycoords='figure fraction',
#                                bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})
# ax.set_title('mti_dc signal')
# ax.set_ylabel('Value')
# ax.set_xlabel('Range [km]')
# ax.grid(True)

# #%% Doppler Analysis

# cfar_mti_doppler = np.zeros(signal_comp.shape,dtype=np.complex64)

# for t in range(1,Np):
#     cfar_mti_doppler[t] = signal_comp[t]-signal_comp[t-1]



# FD = np.exp(-1j*2*np.pi*np.outer(np.arange(1,Np+1),np.arange(1,Np+1).T)/(Np+1))
# #FD = np.exp(-1j*2*np.pi*np.outer(np.arange(1,Np+1),np.arange(1,Np+1).T)/Np)
# filtered = (cfar_mti_doppler.T@FD).T

# #FD = np.exp(-1j*2*np.pi*PRI*np.arange(Np))
# #filtered = (cfar_mti_doppler.T * FD).T

# #filtered = (signal_comp.T@FD).T

# from matplotlib import cm

# vel_vect = np.linspace(-vu/2,vu/2,Np,endpoint=True)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(16,9))
# fig.suptitle('Producto Doppler')
# x = vel_vect
# y = getRank(signal_comp[0])
# X, Y = np.meshgrid(x, y)
# Z = np.abs(fftshift(filtered,axes=0)).T
# surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
# ax.set_xlabel('Velocity')
# ax.set_ylabel('Range')
# ax.set_zlabel('Doppler Product')
# fig.colorbar(surf, shrink=0.5)

# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(16,9))
# # fig.suptitle('Deteccion Doppler')
# # x = vel_vect
# # y = getRank(detect_cfar_doppler[0])
# # X, Y = np.meshgrid(x, y)
# # Z = np.abs(fftshift(detect_cfar_doppler,axes=0)).T
# # surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
# # ax.set_xlabel('Velocity')
# # ax.set_ylabel('Range')
# # ax.set_zlabel('Doppler Detection')
# # fig.colorbar(surf, shrink=0.5)

# #%%


# # =============================================================================
# # signal_comp = []
# # for t in range(len(rxSignal)):
# #     # convolucionar y quitar colas de convolucion
# #     comp = fastconv(rxSignal[t],np.exp(-1j*2*np.pi*t*np.arange(len(rxSignal[t])) /Np))
# #     signal_comp.append(comp)
# # signal_comp = np.stack(signal_comp,axis=0)
# # 
# # 
# # signal_comp_doppler = np.insert(signal_comp,0,np.zeros(len(signal_comp[0]))).reshape(Np+1,-1)
# # 
# # # FFT en tiempo lento (en cada PRF)
# # dopplerProduct = fft(signal_comp_doppler,n=11,axis=0)
# # 
# # gain_doppler_prod = 10
# # 
# # th_cfar_doppler = []
# # for t in range(len(dopplerProduct)):
# #     doppler = gain_doppler_prod*fastconv(h_cfar,np.abs(signal_comp_doppler[t]))
# #     doppler = doppler[len(h_cfar)//2:len(doppler)-len(h_cfar)//2]
# #     th_cfar_doppler.append(doppler)
# # th_cfar_doppler = np.stack(th_cfar_doppler,axis=0).reshape(Np+1,-1)
# # 
# # detect_cfar_doppler = dopplerProduct - th_cfar_doppler
# # 
# # detect_cfar_doppler = np.where(np.real(detect_cfar_doppler)<0,0,detect_cfar_doppler)
# # 
# # vel_vect = np.linspace(-vu/2,vu/2,Np+1,endpoint=True)
# # 
# # #from mpl_toolkits.mplot3d import Axes3D
# # from matplotlib import cm
# # 
# # 
# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(16,9))
# # fig.suptitle('Producto Doppler')
# # x = vel_vect
# # y = getRank(signal_comp_doppler[0])
# # X, Y = np.meshgrid(x, y)
# # Z = np.abs(fftshift(dopplerProduct,axes=0)).T
# # surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
# # ax.set_xlabel('Velocity')
# # ax.set_ylabel('Range')
# # ax.set_zlabel('Doppler Product')
# # fig.colorbar(surf, shrink=0.5)
# # 
# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(16,9))
# # fig.suptitle('Deteccion Doppler')
# # x = vel_vect
# # y = getRank(detect_cfar_doppler[0])
# # X, Y = np.meshgrid(x, y)
# # Z = np.abs(fftshift(detect_cfar_doppler,axes=0)).T
# # surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
# # ax.set_xlabel('Velocity')
# # ax.set_ylabel('Range')
# # ax.set_zlabel('Doppler Detection')
# # fig.colorbar(surf, shrink=0.5)
# # =============================================================================




