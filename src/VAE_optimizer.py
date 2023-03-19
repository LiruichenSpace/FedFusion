import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as DIS 
import torch.autograd as autograd
from torch.autograd import Variable

import numpy as np
import copy
import os
from utils import process_gradients
from models import CNNMnist, CNNFashion_Mnist, CNNCifar

SMALL = 1e-10
def kumaraswamy(a, b):
    u = torch.Tensor(a.shape).uniform_(1e-3, 1. - 1e-3)
    return torch.exp(torch.log(1. - torch.exp(torch.log(u) / (b+SMALL)) + SMALL) / (a+SMALL))

def logit(x):
    return torch.log(x + SMALL) - torch.log(1. - x + SMALL)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, categorical_dim, output_dim):
        pass



class VAE_optimizer():
    def __init__(self, params, num_users=20, component_num=8, vae_step=1):
        pass


