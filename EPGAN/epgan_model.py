"""
ElectroPhysiomeGAN: Generation of Biophysical Neuron Model Parameters from Recorded Electrophysiological Responses
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from EPGAN import default_dir, neuron_params, epgan_generate_data
import seaborn as sns

mv_scaling = 1e-2
g_C_scaling = 1e-1

original_par_inds = np.arange(len(epgan_generate_data.default_param))
nontrainable_inds = epgan_generate_data.identity_inds.copy()
trainable_inds = np.setdiff1d(original_par_inds, nontrainable_inds)

#################################################################################################################################################################
# NN Archs ######################################################################################################################################################
#################################################################################################################################################################

class Generator(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, SS_size, p_enc, p_gen):
        
        super(Generator, self).__init__()

        self.rnn = torch.nn.GRU(input_size = input_size, hidden_size = hidden_size, 
                                  num_layers = num_layers, 
                                  batch_first = True, dropout = p_enc, bidirectional = True)
        
        self.fc_decoder1 = torch.nn.Linear(hidden_size * 2 + SS_size, 1536)
        self.fc_decoder2 = torch.nn.Linear(1536, 768)
        self.fc_decoder3 = torch.nn.Linear(768, 384)
        self.fc_decoder4 = torch.nn.Linear(384, output_size)
        #self.fc_dropout1 = torch.nn.Dropout(p = p_gen)
        #self.fc_dropout2 = torch.nn.Dropout(p = p_gen)
        self.LN1 = torch.nn.LayerNorm(1536)
        self.LN2 = torch.nn.LayerNorm(768)

    def forward(self, input_seq, hidden_state, SS):
        
        encoded_voltage, hidden = self.rnn(input_seq, hidden_state)
        combined = torch.cat([encoded_voltage[:, -1, :], SS], axis = 1)
        
        output1 = torch.nn.functional.relu(self.LN1(self.fc_decoder1(combined)))
        #output1 = self.fc_dropout1(output1)
        output2 = torch.nn.functional.relu(self.LN2(self.fc_decoder2(output1)))
        #output2 = self.fc_dropout2(output2)
        output3 = torch.nn.functional.relu(self.fc_decoder3(output2))
        param_pred = torch.tanh(self.fc_decoder4(output3))
        
        return param_pred

#################################################################################################################################################################
# Scaling Functions #############################################################################################################################################
#################################################################################################################################################################

def nn_scaling(pars):

    nn_scaled_pars = pars.copy()

    downscale_inds = np.union1d(epgan_generate_data.V_neg_inds, epgan_generate_data.V_pos_inds)
    downscale_inds = np.union1d(downscale_inds, np.array([171, 172, 173, 174, 175]))
    downscale_inds = np.union1d(downscale_inds, epgan_generate_data.special_constraint_inds)
    upscale_inds = np.array([92, 93, 102, 103])

    nn_scaled_pars[:, downscale_inds] = pars[:, downscale_inds] * mv_scaling
    nn_scaled_pars[:, upscale_inds] = pars[:, upscale_inds] * (1/mv_scaling)

    g_C_inds = np.union1d(epgan_generate_data.g_inds, epgan_generate_data.C_ind)
    nn_scaled_pars[:, g_C_inds] = pars[:, g_C_inds] * g_C_scaling

    return nn_scaled_pars

def inv_nn_scaling(pars):

    inv_nn_scaled_pars = pars.copy()

    downscale_inds = np.union1d(epgan_generate_data.V_neg_inds, epgan_generate_data.V_pos_inds)
    downscale_inds = np.union1d(downscale_inds, np.array([171, 172, 173, 174, 175]))
    downscale_inds = np.union1d(downscale_inds, epgan_generate_data.special_constraint_inds)
    upscale_inds = np.array([92, 93, 102, 103])

    inv_nn_scaled_pars[:, downscale_inds] = pars[:, downscale_inds] * (1/mv_scaling)
    inv_nn_scaled_pars[:, upscale_inds] = pars[:, upscale_inds] * mv_scaling

    g_C_inds = np.union1d(epgan_generate_data.g_inds, epgan_generate_data.C_ind)
    inv_nn_scaled_pars[:, g_C_inds] = pars[:, g_C_inds] * (1/g_C_scaling)

    return inv_nn_scaled_pars

def full_scaling(pars):

    pars_scaled = pars.copy()

    for param_k in range(len(pars_scaled)):

        pars_scaled[param_k] = epgan_generate_data.time_scaling(pars_scaled[param_k])

    pars_scaled = nn_scaling(pars_scaled)

    return pars_scaled

def inv_full_scaling(pars):

    pars_scaled = pars.copy()

    for param_k in range(len(pars_scaled)):

        pars_scaled[param_k] = epgan_generate_data.inv_time_scaling(pars_scaled[param_k])

    pars_scaled = inv_nn_scaling(pars_scaled)

    return pars_scaled

scaled_nontrainable = torch.from_numpy(full_scaling(epgan_generate_data.default_param[np.newaxis, :])[0, nontrainable_inds]).float()

#################################################################################################################################################################
# Processing functions ##########################################################################################################################################
#################################################################################################################################################################

def construct_Iext(num, current_clamp_list):

    input_mat_list = []

    for iext in current_clamp_list:

        input_mat = torch.zeros((750, 1))       
        input_mat[250:500, 0] = iext
        input_mat_list.append(input_mat)

    input_mat_list = torch.hstack(input_mat_list)
    input_mat_list = input_mat_list.unsqueeze(0).repeat_interleave(num, axis = 0)

    return input_mat_list

def rescale_params(pars, p_min, p_max):
    
    pars_recovered = ((pars + 1) * (p_max - p_min))/2 + p_min
    
    return pars_recovered

def recover_params(rescaled_pars):

    recovered_pars = torch.zeros((len(rescaled_pars), len(epgan_generate_data.default_param)), dtype = rescaled_pars.dtype)
    recovered_pars[:, trainable_inds] = rescaled_pars
    recovered_pars[:, nontrainable_inds] = scaled_nontrainable
    
    return recovered_pars

#################################################################################################################################################################
# Training QOL ##################################################################################################################################################
#################################################################################################################################################################

def test_exp_inputs(Generator_NN, exp_batches_features_V, exp_SS, Pars_min, Pars_max, v_initcond):

    pars_gen_exp = Generator_NN(exp_batches_features_V, None, exp_SS)
    pars_gen_exp_vc = recover_params(rescale_params(pars_gen_exp, Pars_min, Pars_max))
    pars_gen_exp_vc[:, 175] = torch.tensor(v_initcond, device = pars_gen_exp_vc.device) # Match initial conditions

    pars_gen_exp_vc_scaled = inv_full_scaling(pars_gen_exp_vc.cpu().numpy())

    return pars_gen_exp_vc_scaled