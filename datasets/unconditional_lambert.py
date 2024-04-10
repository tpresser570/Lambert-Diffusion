from __future__ import print_function, division
import pandas as pd
import torch 
import math
from torch.utils.data import Dataset
import os

from torch import nn



class LambertDataset(Dataset):

    def __init__(self,csv_file,root_dir):
        """ Arguments:
            csv_file (string): path to the CSV file of ICS
            csv_scalars (string): path to the csv file with the scalars data
            root_dir     (string): Directory with all the trajectory files
            scale_factors (list): list of 3 floats used to scale position, velocity, and time
            dim (integer): either 2 or 3 depending if you want 2D or 3D data
            
        """
        self.intial_conditions_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.intial_conditions_frame)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #trajecotries
        traj_name = os.path.join(self.root_dir,
                                self.intial_conditions_frame.iloc[idx,-1])
        
        trajectory_tensor = torch.load(traj_name) #[t,x,y,vx,vy]



        x = trajectory_tensor[1,:]
        y = trajectory_tensor[2,:]
        #z = trajectory_tensor[3,:]
        vx = trajectory_tensor[3,:]
        vy = trajectory_tensor[4,:]
        #vz = trajectory_tensor[6,:]
        t = trajectory_tensor[0,:]

        padding = torch.zeros_like(x)

        #scaled trajectory tensor
        #3d uncomment
        #((t.unsqueeze(0),x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0),vx.unsqueeze(0),vy.unsqueeze(0),vz.unsqueeze(0),padding.unsqueeze(0)),axis = 0)
        trajectory_tensor_scaled = torch.cat((t.unsqueeze(0),x.unsqueeze(0),y.unsqueeze(0),vx.unsqueeze(0),vy.unsqueeze(0),padding.unsqueeze(0)),axis = 0)

        #single channel img
        img = torch.unsqueeze(trajectory_tensor_scaled,dim = 0)
        target = torch.zeros([1])


        #returns the intial conditions for the tensors and the scaled trajectory for each
        return img.type(torch.float),target


class LambertDataset_unscaled(Dataset):

    def __init__(self,csv_file,root_dir):
        """ Arguments:
            csv_file (string): path to the CSV file
            root_dir     (string): Directory with all the trajectory files
            scale_factors (list): list of 3 floats used to scale position, velocity, and time
            dim (integer): either 2 or 3 depending if you want 2D or 3D data
            
        """
        self.min_rx = -42137.17003845432
        self.max_rx = 41943.28515089416
        self.min_ry = -42160.6731312146
        self.max_ry = 42050.87579047758


        self.max_vx = 10.831376811559116
        self.min_vx = -10.85739254084226
        self.max_vy = 10.863773216670156
        self.min_vy = -10.858455417146548


        self.max_t = 85325.21689499712
        self.min_t = 0
    
        self.intial_conditions_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.intial_conditions_frame)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #trajecotries
        traj_name = os.path.join(self.root_dir,
                                self.intial_conditions_frame.iloc[idx,-1])
        
        trajectory_tensor = torch.load(traj_name) #[t,x,y,vx,vy]



        x = trajectory_tensor[1,:]
        y = trajectory_tensor[2,:]
        vx = trajectory_tensor[4,:]
        vy = trajectory_tensor[5,:]
        t = trajectory_tensor[0,:]

        x_scaled = x
        y_scaled = y
        vx_scaled = vx
        vy_scaled = vy
        t_scaled =  t
        padding = torch.zeros_like(x_scaled)

        #scaled trajectory tensor
        trajectory_tensor_scaled = torch.cat((t_scaled.unsqueeze(0),x_scaled.unsqueeze(0),y_scaled.unsqueeze(0),vx_scaled.unsqueeze(0),vy_scaled.unsqueeze(0),padding.unsqueeze(0)),axis = 0)
        #single channel img
        img = torch.unsqueeze(trajectory_tensor_scaled,dim = 0)
        target = torch.zeros([1])


        #returns the intial conditions for the tensors and the scaled trajectory for each
        return img,target

