# -*- coding: utf-8 -*-


import torch

def encoder(encoder_inputs_size):
    """
    Create encoder model for membership inference attack.
    Individual attack input components are concatenated and passed to encoder.
    """
    encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_inputs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.ReLU()       
            )
    return encoder
    

    