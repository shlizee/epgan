"""
ElectroPhysiomeGAN: Generation of Biophysical Neuron Model Parameters from Recorded Electrophysiological Responses
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

import numpy as np
import pandas as pd
import os
import torch

from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
from EPGAN import default_dir, neuron_params

exp_V_scaling = 10. # V -> 10V
exp_VI_scaling = 100. # pA -> 100pA

def load_exp_data_V(neuron_name, rec_type):

    os.chdir(default_dir + '/EPGAN/data/exp')

    if rec_type == 'multi':

        filename_0 = neuron_name + '_current_clamp_0.txt'
        filename_1 = neuron_name + '_current_clamp_1.txt'
        filename_2 = neuron_name + '_current_clamp_2.txt'

        df_neuron_0 = pd.read_csv(filename_0, delimiter = "\t")
        df_neuron_1 = pd.read_csv(filename_1, delimiter = "\t")
        df_neuron_2 = pd.read_csv(filename_2, delimiter = "\t")

        V0 = df_neuron_0.to_numpy()[2:22490-4800, 1:].astype('float')[::25][4:-4]
        V1 = df_neuron_1.to_numpy()[2:22490-4800, 1:].astype('float')[::25][4:-4]
        V2 = df_neuron_2.to_numpy()[2:22490-4800, 1:].astype('float')[::25][4:-4]

        V = np.stack([V0, V1, V2]).mean(axis = 0)
        V_scaled = torch.unsqueeze(torch.tensor(V), 0).float() * exp_V_scaling

    else:

        filename = neuron_name + '_current_clamp.txt'
        df_neuron = pd.read_csv(filename, delimiter = "\t")

        if neuron_name == 'AFD':

            V = df_neuron.to_numpy()[2:109950-39000, 1:].astype('float')[::100][5:-5]
            V_scaled = torch.unsqueeze(torch.tensor(V), 0).float() * exp_V_scaling

        else:

            V = df_neuron.to_numpy()[2:22490-4800, 1:].astype('float')[::25][4:-4]
            V_scaled = torch.unsqueeze(torch.tensor(V), 0).float() * exp_V_scaling

    V_scaled_75 = V_scaled[:, :, 3:]
    V_scaled_50 = V_scaled[:, :, 6:]
    V_scaled_25 = V_scaled[:, :, 9:]

    return V_scaled, V_scaled_75, V_scaled_50, V_scaled_25

def load_exp_data_IV(neuron_name, rec_type):

    os.chdir(default_dir + '/EPGAN/data/exp')

    if rec_type == 'multi':

        filename_0 = neuron_name + '_voltage_clamp_0.txt'
        filename_1 = neuron_name + '_voltage_clamp_1.txt'
        filename_2 = neuron_name + '_voltage_clamp_2.txt'

        VI_0 = pd.read_csv(filename_0, delimiter = "\t")
        VI_1 = pd.read_csv(filename_1, delimiter = "\t")
        VI_2 = pd.read_csv(filename_2, delimiter = "\t")

        VI_0 = (VI_0.to_numpy()[5000:-5000, 1:].astype('float') * 1e12).mean(axis = 0)
        VI_1 = (VI_1.to_numpy()[5000:-5000, 1:].astype('float') * 1e12).mean(axis = 0)
        VI_2 = (VI_2.to_numpy()[5000:-5000, 1:].astype('float') * 1e12).mean(axis = 0)
        VI = np.stack([VI_0, VI_1, VI_2]).T.mean(axis = 1)
        VI_sigma = np.stack([VI_0, VI_1, VI_2]).T.std(axis = 1)

    else:

        if neuron_name == 'RIM':

            VI = np.array([-18.34, -15.27, -12.2, -9.13, -6.57, -4.91, -3.57, -2.13, -0.807, 0.229, 1.46, 4.27, 7.46, 11.8, 17.2, 21.6, 27.1, 32.5])
            VI_sigma = np.array([2.39, 2.39, 2.39, 1.69, 1.21, 0.784, 0.527, 0.388, 0.392, 0.646, 0.926, 2.01, 2.99, 4.02, 5.9, 6.06, 6.93, 7.81])

        elif neuron_name == 'AIY':

            VI = np.array([-13.1, -10.4, -7.92, -5.89, -4.11, -2.69, -1.02, 0.0211, 1.17, 3.1, 7.32, 14.2, 22.4, 31.5, 43.2, 54.5, 69.5, 82.4])
            VI_sigma = np.array([2.88, 2.55, 1.47, 1.31, 1.04, 0.809, 0.7, 0.658, 0.638, 0.889, 1.94, 3.5, 5.36, 7.63, 10.6, 13.3, 16, 17.9])

        elif neuron_name == 'AFD':

            VI = np.array([-87.7, -68.6, -49.5, -18.2, -5.06, 2.19, 3.37, 2.52, 2.68, 5.97, 14.6, 33.4, 60.2, 85, 114, 152, 208, 254])
            VI_sigma = np.array([1, 1, 8.65, 0.636, 1.31, 1.83, 1.46, 0.814, 0.455, 0.613, 2.63, 7.71, 14.7, 22.3, 27.4, 44.1, 73.7, 97.6]) 

        else:

            filename = neuron_name + '_voltage_clamp.txt'
            VI_pd = pd.read_csv(filename, delimiter = "\t")
            VI = (VI_pd.to_numpy()[5000:-5000, 1:].astype('float') * 1e12).mean(axis = 0)
            VI_sigma = (VI_pd.to_numpy()[5000:-5000, 1:].astype('float') * 1e12).std(axis = 0)

    V_x_full = np.linspace(-120, 70, 20)
    V_x_75 = np.linspace(-70, 70, 15)
    V_x_50 = np.linspace(-30, 70, 11)
    V_x_25 = np.linspace(20, 70, 6)

    VI = torch.tensor(VI) / exp_VI_scaling
    VI = torch.unsqueeze(VI, 0)

    VI_75_interp = interpolate.interp1d(V_x_75, VI[0, -15:], fill_value = "extrapolate")
    VI_50_interp = interpolate.interp1d(V_x_50, VI[0, -11:], fill_value = "extrapolate")
    VI_25_interp = interpolate.interp1d(V_x_25, VI[0, -6:], fill_value = "extrapolate")

    VI_75 = torch.unsqueeze(torch.from_numpy(VI_75_interp(V_x_full))/ exp_VI_scaling, 0)
    VI_50 = torch.unsqueeze(torch.from_numpy(VI_50_interp(V_x_full))/ exp_VI_scaling, 0)
    VI_25 = torch.unsqueeze(torch.from_numpy(VI_25_interp(V_x_full))/ exp_VI_scaling, 0)

    return VI, VI_75, VI_50, VI_25, VI_sigma