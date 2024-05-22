from scipy.io import netcdf, loadmat
import os
import numpy as np
import pickle
from tqdm import tqdm
import time
from multiprocessing import Pool

from interpolator import Interpolator

grid_path = "/Volumes/Seagate Backup Plus Drive/gmx1km_grd.nc"
grid_z_mat_path = '/Volumes/Seagate Backup Plus Drive/50_5_1.mat'
flow_data_folder_path = "/Volumes/Seagate Backup Plus Drive/CROCO_DATA/2016_remain"
traj_path =  "/Volumes/Seagate Backup Plus Drive/ptraj_result/2016output"


def read_flow_field_at_time(time):
    with open("time_to_filename.pkl", "rb") as f:
        time_to_filename = pickle.load(f)
    file_name = time_to_filename[time]
    file_path = os.path.join(flow_data_folder_path, file_name)
    nc_data = netcdf.NetCDFFile(file_path,'r')

    # read velocity and temperature
    u, v, w = nc_data.variables['u'][0].copy(), nc_data.variables['v'][0].copy(), nc_data.variables['w'][0].copy()
    temperature = nc_data.variables['temp'][0].copy()[:, 1:, 1:]
    scrum_time = nc_data.variables['scrum_time'][:]
    assert scrum_time == time

    # calibrate u, v, w
    u_cal = (u[:,1:, :] + u[:,:-1, :])/2
    v_cal = (v[:,:, 1:] + v[:,:, :-1])/2
    w_cal = w[:, 1:, 1:]
    return u_cal, v_cal, w_cal, temperature


def process_traj_file(traj_file_path):
    # read grid and build interpolator
    grid = netcdf.NetCDFFile(grid_path,'r')
    lon_psi, lat_psi = grid.variables['lon_psi'][:], grid.variables['lat_psi'][:]
    z_mat = loadmat(grid_z_mat_path)
    z_cal = z_mat['z'][:, 1:, 1:] #到水面的距离, m
    dist_to_bottom = z_cal - z_cal[0]
    lon_psi_3d = np.tile(lon_psi, (z_cal.shape[0], 1,1))
    lat_psi_3d = np.tile(lat_psi, (z_cal.shape[0], 1,1))
    interpolator = Interpolator(lon_psi_3d, lat_psi_3d, z_cal)

    # process trajectory
    traj = netcdf.NetCDFFile(traj_file_path, 'r')
    lon = traj.variables['lon'][:].copy() # (time_step, pid) # 时间步30min
    lat = traj.variables['lat'][:].copy()
    depth = traj.variables['depth'][:].copy() #到水面距离
    ptime = traj.variables['time'][:].copy() #
    state = {"lon_m":[], "lat_m":[], "depth":[], "d2b":[], "u":[], "v":[], "w":[], "temp":[], "age":[]}
    for i_time in tqdm(range(len(ptime))):
        t_cur = ptime[i_time]
        t1 = time.time()
        u, v, w, temperature = read_flow_field_at_time(t_cur)
        t2 = time.time()
        lon_m, lat_m = interpolator.transformer.transform(lon[i_time], lat[i_time])
        t3 = time.time()
        u_inter, v_inter, w_inter, temp_inter, d2b_inter = interpolator.interpolate([u, v, w, temperature, dist_to_bottom], lon[i_time], lat[i_time], depth[i_time])
        t4 = time.time()
        print(t2 - t1, t3 - t2, t4 - t3)
        age = np.array([t_cur - ptime[0]]*len(u_inter))
        
        state['lon_m'].append(lon_m)
        state['lat_m'].append(lat_m)
        state['depth'].append(depth[i_time])
        state['d2b'].append(d2b_inter)
        
        state['u'].append(u_inter)
        state['v'].append(v_inter)
        state['w'].append(w_inter)
        state['temp'].append(temp_inter)
        state['age'].append(age)
    for key in state:
        state[key] = np.array(state[key])

    file_name = traj_file_path.split("/")[-1]
    with open(os.path.join("processed_data", file_name[:-3] + ".pkl"), "wb") as f:
        pickle.dump(state, f)

if __name__ == "__main__":
    # trajectory files with uvw starts from 20160821
    options = os.listdir(traj_path)
    file_pathes = []
    for option in options:
        for name in os.listdir(os.path.join(traj_path, option)):
            if ".nc" not in name:
                continue
            if name[:-3].split("_")[-1] < "20160821": #skip files with no flow field
                continue
            file_pathes.append(os.path.join(traj_path, option, name))
    print(len(file_pathes))
    process_traj_file(file_pathes[0])