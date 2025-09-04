"""
ElectroPhysiomeGAN: Generation of Biophysical Neuron Model Parameters from Recorded Electrophysiological Responses
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

import numpy as np
import pandas as pd
import os
import torch
from scipy.stats import skewnorm

from EPGAN import default_dir, neuron_params

default_param = np.array(list(neuron_params.generic_model_params.copy().values()))

g_inds = np.array([0, 19, 31, 38, 47, 60, 77, 90, 91, 100, 101, 110, 136, 152, 165, 166])

time_inds = np.array([5, 10, 11, 14, 15, 18, 24, 29, 30, 34, 37, 41, 46, 52, 55, 56, 59, 69, 72, 73, 76, 82, 85, 86, 89, 121, 124, 127, 129, 132, 135, 141, 145, 146, 149, 157, 160, 
                      161, 164])

V_neg_inds = np.array([1, 3, 12, 16, 20, 22, 32, 35, 39, 48, 50, 53, 57, 61, 63, 66, 70, 78, 80, 83, 87, 111, 113, 115, 130, 133, 137, 139, 
                       147, 150, 153, 155, 158, 162])

V_pos_inds = np.array([2, 4, 13, 17, 21, 23, 33, 36, 40, 49, 51, 54, 58, 62, 64, 67, 71, 74, 75, 79, 81, 84, 88, 112, 114, 116, 131, 134, 138, 140, 
                       148, 151, 154, 156, 159, 163])

VK_ind = np.array([171])
VCa_ind = np.array([172])
VNa_ind = np.array([173])
VLEAK_ind = np.array([174])
V0_ind = np.array([175])
a0_inds = np.array([176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
                    196, 197, 198, 199, 200])

special_constraint_inds = np.array([6, 7, 8, 9, 25, 26, 27, 28, 42, 43, 44, 45, 122, 123, 125, 126, 142, 143, 144])

identity_inds = np.array([65, 68, 92, 93, 94, 95, 96, 97, 98, 99, 102, 103, 104, 105, 106, 107, 108, 109, 117, 118, 119, 120, 128, 167, 168, 169, 170, 201])

C_ind = np.array([202])

def time_scaling(par):

    t_scaled_par = par.copy()

    ms_2_s_inds = np.array([5, 10, 11, 14, 15, 18, 24, 29, 30, 34, 37, 41, 46, 52, 55, 56, 59, 69, 72, 73, 76, 82, 85, 86, 89,
                            121, 124, 127, 129, 132, 135, 141, 145, 146, 149, 157, 160, 161, 164])

    msinv_2_sinv_inds = np.array([94, 95, 104, 105, 170])

    t_scaled_par[ms_2_s_inds] = par[ms_2_s_inds] * neuron_params.ms_2_s
    t_scaled_par[msinv_2_sinv_inds] = par[msinv_2_sinv_inds] * (1/neuron_params.ms_2_s)

    return t_scaled_par

def inv_time_scaling(par):

    t_scaled_par = par.copy()

    ms_2_s_inds = np.array([5, 10, 11, 14, 15, 18, 24, 29, 30, 34, 37, 41, 46, 52, 55, 56, 59, 69, 72, 73, 76, 82, 85, 86, 89,
                            121, 124, 127, 129, 132, 135, 141, 145, 146, 149, 157, 160, 161, 164])

    msinv_2_sinv_inds = np.array([94, 95, 104, 105, 170])

    t_scaled_par[ms_2_s_inds] = par[ms_2_s_inds] * (1/neuron_params.ms_2_s)
    t_scaled_par[msinv_2_sinv_inds] = par[msinv_2_sinv_inds] * neuron_params.ms_2_s

    return t_scaled_par