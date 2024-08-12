# -*- coding: utf-8 -*-
"""
created on: 2025-03-25
@author:    Jasper Heuer
use:        1) reproject Landsat GEE imagery to common grid
            2) adjust GeoTransform to line up grids
"""

# import packages ==================================================================================

import os
import glob
import time
import numpy as np
from osgeo import gdal

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

file_list = glob.glob("./Landsat/Landsat_05/GEE_imagery/" + "*.tif", recursive=True)

# define variables =================================================================================

fn_mask_raster = "./Masks/mask.tif"
dst_crs = "EPSG:32624" # destination coordinate system
res = 30 # pixel size in meters

# get corner coordinates for mask:
mask = gdal.Open(fn_mask_raster)
xmin, ymax = mask.GetGeoTransform()[0], mask.GetGeoTransform()[3]

mask = None # set to none

# batch reproject ==================================================================================

start_time = time.time()

# create list of dates:
date_list = []
for i in range(0, np.size(file_list)):
    date_list.append("LT05_" + file_list[i].split("\\")[1][12:20])

# run batch reprojection loop:
for i in range(0, np.size(file_list)):
    print("Reprojecting: " + str(date_list[i]))
    
    fn_in = file_list[i]
    fn_out = "./Landsat/Landsat_05/Reprojected/" + str(date_list[i]) + "_reprojected.tif"

    ds = gdal.Open(fn_in) # read dataset
    # inspect projection by typing ds.GetProjection() in the console- NOT the editor

    # reproject to common grid:
    ds_reproj = gdal.Warp(fn_out, ds, dstSRS=dst_crs,
                          xRes=res, yRes=-res,
                          cutlineDSName="./Masks/mask_UTM-24N.shp", # cut by extend of mask
                          cropToCutline=True, dstNodata=np.nan)

    # set data to none:
    ds = None
    ds_reproj = None
    
    print("Done!")
    
# batch adjust GeoTransform ========================================================================

reproj_file_list = glob.glob("./Landsat/Landsat_05/Reprojected/" + "*.tif", recursive=True)

reproj_date_list = []

# create list of dates:
for i in range(0, np.size(reproj_file_list)):
    reproj_date_list.append(reproj_file_list[i].split("\\")[1][0:13])

# run GeoTransform adjustment loop:
for i in range(0, np.size(reproj_file_list)):
    print("Adjust GeoTransform: " + str(reproj_date_list[i]))
    
    ds2 = gdal.Open(reproj_file_list[i])
    ds2.SetGeoTransform([xmin, res, 0.0, ymax, 0.0, -res]) # adjust GeoTransform
    
    # save copy to drive
    driver = gdal.GetDriverByName("GTiff")
    moved_ds = driver.CreateCopy("./Landsat//Landsat_05/Resampled/" + str(reproj_date_list[i]) + 
                                 "_resample.tif",ds2)
    
    # set data to none:
    ds2 = None
    moved_ds = None
    
    print("Done!")

# print duration:
print(f"Duration: {time.time() - start_time} seconds")
