# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:30:28 2022

@author: fdadam
"""

from pymatreader import read_mat
import numpy as np
import pandas as pd

mat = read_mat('radar_pulsado.mat')


y_og = mat['y'].transpose() 

y = y_og.flatten()

y_real = y.real
y_imag = y.imag

dfData = pd.DataFrame({'real':y.real,
                       'imag':y.imag
                       })

dfData_s = pd.DataFrame({'channel_real':mat['s'].real,
                         'channel_imag':mat['s'].imag
                         })

dfData.to_csv('signal.csv')

dfData_s.to_csv('channel.csv')
