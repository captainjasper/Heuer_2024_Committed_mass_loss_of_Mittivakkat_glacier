# -*- coding: utf-8 -*-
"""
created on: 2025-03-25
@author:    Jasper Heuer
use:        1) reproject Landsat GEE imagery to common grid
            2) adjust GeoTransform to line up grids exactly
"""

# import packages ==================================================================================

import os
import glob
import time
import shutil
import numpy as np
from osgeo import gdal

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

start_time = time.time() # set start time

# create new directories:
path_reprojected = "./Landsat/Reprojected"
path_resampled = "./Landsat/Resampled"

if not os.path.exists(path_reprojected):
    os.makedirs(path_reprojected)
if not os.path.exists(path_resampled):
    os.makedirs(path_resampled)

# create file list:
file_list = glob.glob("./Landsat/GEE_imagery/" + "*.tif", recursive=True)

# define variables =================================================================================

fn_mask_raster = "./Masks/mask.tif"
dst_crs = "EPSG:32624" # destination coordinate system
res = 30 # pixel size in meters

# get corner coordinates for mask:
mask = gdal.Open(fn_mask_raster)
xmin, ymax = mask.GetGeoTransform()[0], mask.GetGeoTransform()[3]

mask = None # set to none

# batch reproject ==================================================================================

# create list of dates:
date_list = []

for i in range(0, np.size(file_list)):
    if file_list[i].split("\\")[1][0:4] == "LT05":
        date_list.append("LT05_" + file_list[i].split("\\")[1][12:20])
    elif file_list[i].split("\\")[1][0:4] == "LE07":
        date_list.append("LE07_" + file_list[i].split("\\")[1][12:20])
    elif file_list[i].split("\\")[1][0:4] == "LC08":
        date_list.append("LC08_" + file_list[i].split("\\")[1][12:20])
    elif file_list[i].split("\\")[1][0:4] == "LC09":
        date_list.append("LC09_" + file_list[i].split("\\")[1][12:20])
    else:
        print("Could not determine sensor for file: " + file_list[i].split("\\")[1])
        pass

# run reprojection loop:
for i in range(0, np.size(file_list)):
    print("Reprojecting: " + str(date_list[i]))
    
    fn_in = file_list[i]
    fn_out = "./Landsat/Reprojected/" + str(date_list[i]) + "_reprojected.tif"

    ds = gdal.Open(fn_in) # read dataset
    # inspect projection by typing ds.GetProjection() in the console- NOT the editor

    # reproject to common grid:
    ds_reproj = gdal.Warp(fn_out, ds, dstSRS=dst_crs,
                          xRes=res, yRes=-res,
                          cutlineDSName="./Masks/mask_UTM-24N.shp", # cut by extend of mask
                          cropToCutline=True, 
                          outputType=gdal.GDT_Float32,  # comment this one out if UInt16 is wanted
                          dstNodata=np.nan)

    # set data to none:
    ds = None
    ds_reproj = None
    
    print("Done!")
    
# batch adjust GeoTransform ========================================================================

reproj_file_list = glob.glob("./Landsat/Reprojected/" + "*.tif", recursive=True)

# create list of dates:
reproj_date_list = []

for i in range(0, np.size(reproj_file_list)):
    reproj_date_list.append(reproj_file_list[i].split("\\")[1][0:13])

# run GeoTransform adjustment loop:
for i in range(0, np.size(reproj_file_list)):
    print("Adjust GeoTransform: " + str(reproj_date_list[i]))
    
    ds2 = gdal.Open(reproj_file_list[i])
    ds2.SetGeoTransform([xmin, res, 0.0, ymax, 0.0, -res]) # adjust GeoTransform
    
    # save copy to drive
    driver = gdal.GetDriverByName("GTiff")
    moved_ds = driver.CreateCopy("./Landsat/Resampled/" + str(reproj_date_list[i]) + 
                                 "_resample.tif", ds2)
    
    # set data to none:
    ds2 = None
    moved_ds = None
    
    print("Done!")

# print duration:
print(f"Duration: {time.time() - start_time} seconds")

# clean-up drive ===================================================================================

shutil.rmtree("./Landsat/Reprojected") # remove folder
