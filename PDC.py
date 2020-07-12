#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 08:12:44 2020
@author: qiusb
"""
import numpy as np
import pandas as pd
from copy import deepcopy


data = np.loadtxt('./files/V4.txt')# N H log(g)
n_A = data[:,0]/max(data[:,0])
info = pd.DataFrame({'N_A':data[:,0], 'n_A':n_A, 'Energy':data[:,1], 'degeneracy':np.round(np.exp(data[:,2]))})
nm = 100
T_domain = range(100, 700, 100)
results = pd.DataFrame(columns=['mu', 'T', 'n_A'])
for T in T_domain:
    print(T)
    mu_domain = [-10, 10, -10, 10]# first(last) two are the range of mu let n=0(1)
    for t_split in range(20):
        # dichotomy
        mu_0 = sum(mu_domain[:2]) * 0.5
        mu_1 = sum(mu_domain[2:]) * 0.5
        H_enthalpy_0 = info['Energy'] - info['N_A'] * mu_0
        H_enthalpy_0 = H_enthalpy_0 -min(H_enthalpy_0) * np.ones(len(H_enthalpy_0))      
        H_enthalpy_1 = info['Energy'] - info['N_A'] * mu_1 
        H_enthalpy_1 = H_enthalpy_1 -min(H_enthalpy_1) * np.ones(len(H_enthalpy_1))
        k = 1.38e-23
        z_0 = info['degeneracy']*np.exp(-H_enthalpy_0*(1.602192e-19)/k/T)
        E_n_muT_0 =  np.dot(z_0, info['n_A']) / sum(z_0)
        z_1 = info['degeneracy']*np.exp(-H_enthalpy_1*(1.602192e-19)/k/T)
        E_n_muT_1 =  np.dot(z_1, info['n_A']) / sum(z_1)
        if E_n_muT_0 < 1e-2:
            mu_domain[0] = mu_0
        else:
            mu_domain[1] = mu_0
        if E_n_muT_1 > 1-1e-2:
            mu_domain[3] = mu_1
        else:
            mu_domain[2] = mu_1
    ms = sum(mu_domain[:2]) * 0.5
    me = sum(mu_domain[2:]) * 0.5
    mu_T_domain = np.linspace(ms, me, nm)
    for mu_T in mu_T_domain:
        H_enthalpy = info['Energy'] - info['N_A'] * mu_T
        H_enthalpy = H_enthalpy -min(H_enthalpy) * np.ones(len(H_enthalpy))
        k = 1.38e-23
        z = info['degeneracy']*np.exp(-H_enthalpy*(1.602192e-19)/k/T)
        E_n_muT =  np.dot(z, info['n_A']) / sum(z)
        results.loc[len(results['mu'])]=[mu_T, T, E_n_muT]

results.loc[len(results['mu'])] = [min(np.array(results['mu'])), min(np.array(results['T'])), 0]
results.loc[len(results['mu'])]=[max(np.array(results['mu'])), min(np.array(results['T'])), 1]



from scipy.interpolate import griddata
x = np.linspace(min(np.array(results['mu'])),max(np.array(results['mu'])),1000)
y = np.linspace(min(np.array(results['T'])),max(np.array(results['T'])),1000)
X, Y = np.meshgrid(x,y)

# Choose npts random point from the discrete domain of our model function
px, py = np.array(results['mu']), np.array(results['T'])
Ti = griddata((px, py), np.array(results['n_A']), (X, Y), method='linear')

import plotly.graph_objects as go

fig = go.Figure(data =
    go.Contour(
        z=Ti,
        x=x, # horizontal axis
        y=y # vertical axis
    ))
fig.show()