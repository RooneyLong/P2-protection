# -*- coding: utf-8 -*-

import os

import math
import torch
from torch import nn
import torch.nn.functional as F


import numpy as np

def sanity_check(layers, layers_to_exploit):
    """
    Basic sanity check for layers and gradients to exploit based on model layers
    """
    if layers_to_exploit and len(layers_to_exploit):
        assert np.max(layers_to_exploit) <= len(layers),\
            "layer index greater than the last layer"

def time_taken(self, start_time, end_time):
    """
    Calculates difference between 2 times
    """
    delta = end_time - start_time
    hours = int(delta / 3600)
    delta -= hours * 3600
    minutes = int(delta / 60)
    delta -= minutes * 60
    seconds = delta
    return hours, minutes, np.int(seconds)            

class attack_utils():
    """
    Utilities required for conducting membership inference attack
    """

    def __init__(self, directory_name='latest'):
        self.root_dir = os.path.abspath(os.path.join(
                                        os.path.dirname(__file__),
                                        ))
        self.log_dir = os.path.join(self.root_dir, "logs")
#        self.aprefix = os.path.join(self.log_dir,
#                                    directory_name,
#                                    "attack",
#                                    "model_checkpoints")
#        self.dataset_directory = os.path.join(self.root_dir, "datasets")

#        if not os.path.exists(self.aprefix):
#            os.makedirs(self.aprefix)
#        if not os.path.exists(self.dataset_directory):
#            os.makedirs(self.dataset_directory)
            
    def get_gradshape(self, layer):
        """
        Returns the shape of gradient matrices
        Args:
        -----
        model: model to attack 
        """
        row = layer[0].size()[0]
        col = layer[0].size()[1]
        gradshape = (row,col)
        return gradshape          

    def split(self, x):
        """
        Splits the array into number of elements equal to the
        size of the array. This is required for per example
        computation.
        """
        split_x = torch.split(x, len(x.numpy()))
        
        return split_x
    def createOHE(self, num_output_classes):
            """
            creates one hot encoding matrix of all the vectors
            in a given range of 0 to number of output classes.
            """
            return F.one_hot(torch.arange(0, num_output_classes),
                             num_output_classes)

    def one_hot_encoding(self, labels, ohencoding):
            """
            Creates a one hot encoding of the labels used for 
            inference model's sub neural network
            Args: 
            ------
            zero_index: `True` implies labels start from 0
            """
            labels = (labels.clone().detach().to(torch.int64)).numpy()
            result = np.stack(list(map(lambda x: ohencoding[x], labels)))
            result = np.squeeze(result,axis=1)
            result = torch.tensor(result,dtype=torch.float32)        
            return result

