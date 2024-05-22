"""create filename_to_time.pkl that maps a time to the name of the file that stores the flow field at that time.
"""
import os
from scipy.io import netcdf
import pickle
from tqdm import tqdm

time_to_filename = {}
dir1 = "/Volumes/Seagate Backup Plus Drive/CROCO_DATA/2016Jun"
dir2 = "/Volumes/Seagate Backup Plus Drive/CROCO_DATA/2016_remain"

try:
    for dir in [dir1, dir2]:
        for name in tqdm(os.listdir(dir)):
            if ".nc" in name and "avg" in name:
                file_path = os.path.join(dir, name)
                file2read = netcdf.NetCDFFile(file_path,'r')
                time = file2read.variables['scrum_time'][0]
                if time < 1:
                    continue
                time_to_filename[time] = name
    with open("time_to_filename.pkl", "wb") as f:
        pickle.dump(time_to_filename, f)
except:
    print("file not found")

