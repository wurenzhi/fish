from scipy.io import netcdf, loadmat
import numpy as np

path = "/Volumes/Seagate Backup Plus Drive/CROCO_DATA/2016Jun/croco_avg.00761.nc"
grid_path = "/Volumes/Seagate Backup Plus Drive/gmx1km_grd.nc"
mat = loadmat('/Volumes/Seagate Backup Plus Drive/50_5_1.mat')
traj_path = "/Volumes/Seagate Backup Plus Drive/ptraj_result/2016output/croco-buoy/crocobuoy_DanceBooker_20160501.nc"

file2read = netcdf.NetCDFFile(path,'r')

# read velocity and temp
u, v, w = file2read.variables['u'][0], file2read.variables['v'][0], file2read.variables['w'][0] #
temp = file2read.variables['temp'][0][:, 1:, 1:]
time = file2read.variables['scrum_time'] # in seconds
#u (time, s_rho, eta_rho, xi_u), s_rho: z, eta_rho: y, xi_u: x
u_inter = (u[:,1:, :] + u[:,:-1, :])/2
v_inter = (v[:,:, 1:] + v[:,:, :-1])/2
w_inter = w[:, 1:, 1:]

# read grid
grid = netcdf.NetCDFFile(grid_path,'r')
lon_psi, lat_psi = grid.variables['lon_psi'], grid.variables['lat_psi']
z_inter = mat['z'][:, 1:, 1:] #到水面的距离, m


# read trajectory
traj = netcdf.NetCDFFile(traj_path, 'r')
lon = traj.variables['lon'] # (time_step, pid) # 时间步30min
lat = traj.variables['lat']
depth = traj.variables['depth'] #到水面距离
ptime = traj.variables['time'] #

#reward: 26-30到达终点

#1： 海底深度范围：z[0] in [-15, -64]
#2： 粒子到海底距离：|depth - z[0] | < 5
#3： 粒子所在位子温度： 21 < temp < 25

print()
