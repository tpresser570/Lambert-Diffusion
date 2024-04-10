import torch 
import numpy as np 

def img2traj(img,ranges_array):
    t = ranges_array[0:2]
    x = ranges_array[4:6]
    y = ranges_array[8:10]
    z = ranges_array[12:14]
    vx = ranges_array[16:18]
    vy = ranges_array[20:22]
    vz = ranges_array[24:36]

    traj = np.array(img[0,:,:]) #remove color channels from the image

    t_real = traj[0,:]*t[1]
    x_real = traj[1,:] * (x[1] - x[0]) + x[0]
    y_real = traj[2,:] * (y[1] - y[0]) + y[0]
    vx_real = traj[3,:] * (vx[1] - vx[0]) + vx[0]
    vy_real = traj[4,:] * (vy[1] - vy[0]) + vy[0]


    real_traj = np.vstack([t_real,x_real,y_real,vx_real,vy_real])

    return real_traj