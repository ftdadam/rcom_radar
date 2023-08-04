# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 22:29:58 2023

@author: fdadam
"""

import numpy as np

Npts = 10000
time = np.linspace(0,1,Npts)

voltage = (220*np.sqrt(2))*np.sin(2*np.pi*100e3*time)
rms_voltage = np.sqrt(1/len(time)*np.sum(voltage**2))
R = 100.0
instant_power = voltage**2/R
average_power = np.mean(instant_power)

print(f'Resistor = {R:.2f} Ohms')
print(f'RMS Voltage = {rms_voltage:.2f} V')
print(f'Average Power = {average_power:.2f} W')