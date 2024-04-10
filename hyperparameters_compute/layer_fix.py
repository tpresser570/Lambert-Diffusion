#%% imports and config setup
import torch
import argparse
import pandas as pd
from yaml import load, dump
from yaml import Loader, Dumper
from models.ncsnv2 import NCSNv2
import os

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace




with open(os.path.join('configs', 'unconditional_lambert.yml'), 'r') as f:
    config = load(f,Loader=Loader)

    #%%
new_config = dict2namespace(config)
# %% add device


#%%
test_data = torch.rand(32,256,4,16)
#conv_result = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
           #     output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.

x = test_data
out = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2])
# %%

output = test_data
# Determine the new shape for mean pooling
batch_size, channels, height, width = output.shape
height_pool = 2 if height % 2 == 0 else 1
width_pool = 2 if width % 2 == 0 else 1

# Reshape and permute for pooling
output = output.view(batch_size, channels, height // height_pool, height_pool, width // width_pool, width_pool)
out2 = output.mean(-1).mean(-2)




# %%
