import numpy as np
import pandas as pd
import os
from pyntcloud import PyntCloud
import matplotlib
import matplotlib.pyplot as plt
import math
import subprocess
import argparse
import glob
import time

#Arg parser
parser = argparse.ArgumentParser(description='Select depth output file')
parser.add_argument('file', type=str, help='defines the source depth output file')
parser.add_argument('width', type=int, help='defines the width of the source depth output file')
#parser.add_argument('--filter', type=str, help='defines filter type (SOR, ROR)')
#parser.add_argument('--offset', type=str, help='defines the axis ofset (x, y or z)')
#parser.add_argument('--val_1', type=float, help='first filter value')
#parser.add_argument('--val_2', type=float, help='second filter value')
args = parser.parse_args()

# FUNCTIONS:

# Convert sensor data to planar data
def map_to_plane(x_index, y_index, tof_depth, width, height):
    # Compute d
    focal_length = 451.19988773439741
    x_center = (width - 1) / 2
    y_center = (height - 1) / 2
    delta_x = (x_index - x_center)
    delta_y = (y_index - y_center)
    d = math.sqrt(delta_x**2 + delta_y**2)
    
    # Compute lens angle and plane angle
    theta = math.atan(d/focal_length)
    phi = math.atan(delta_x/delta_y)
    
    # Compute planar coords
    planar_z = math.cos(theta) * tof_depth
    scaling_factor = (-1) * planar_z / focal_length
    planar_x = scaling_factor * delta_x
    planar_y = scaling_factor * delta_y
    
    return planar_x, planar_y, planar_z

# Fill points ndarray
def raw_to_ndarray(depth_file, width):
    # Read file into ndarray
    depth_list = np.fromfile(depth_file, dtype='float32')
    # Reshape ndarray
    depth_map = np.reshape(depth_list, (-1, width))
    height = depth_map.size / width
    # Allocate points ndarray
    np_points = np.zeros(shape=(depth_list.size, 3))
    gen_pos = 0
    # Fill points ndarray
    for index, depth in np.ndenumerate(depth_map):
        x_pos, y_pos, z_pos = map_to_plane(index[1],
                                           index[0],
                                           depth,
                                           width,
                                           height)
        np_points[gen_pos] = [x_pos, z_pos, y_pos]
        gen_pos += 1
    
    np_points, discarded_points = np.vsplit(np_points, np.array([480 * width]))
    return np_points

# Plot a point cloud from an ndarray of x,y,z coords
def get_point_cloud(np_points):
    # Convert to pandas data frame
    pandas_points = pd.DataFrame(np_points, columns=['x', 'y', 'z'])
    # Construct cloud from data frame
    cloud = PyntCloud(pandas_points)
    return cloud

# Get input file, output file and points ndarray
def get_io(file_path, width):
    base_name = os.path.basename(file_path)
    file_name, ext = os.path.splitext(base_name)
    np_points = raw_to_ndarray(file_path, width)
    return np_points

# ––––––––––––––––––––––– MAIN –––––––––––––––––––––––
np_points = get_io(args.file, args.width)
cloud = get_point_cloud(np_points)

cloud.plot(backend="threejs", initial_point_size=10, use_as_color="z", cmap="viridis_r") #, polylines=lines)

# Move files into folder: threejs_plot
if os.path.isdir('threejs_tmp'):
    subprocess.check_call(['rm', '-rf', 'threejs_tmp'])
subprocess.check_call(['mkdir', 'threejs_tmp'])
files = glob.glob('pyntcloud_plot*')
for file in files:
    subprocess.check_call(['mv', file, 'threejs_tmp/.'])

# Open html file:
subprocess.check_call(['open', 'threejs_tmp/pyntcloud_plot.html'])

# Clean up
time.sleep(5)
subprocess.check_call(['rm', '-rf', 'threejs_tmp'])