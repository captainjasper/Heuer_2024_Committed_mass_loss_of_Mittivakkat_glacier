# -*- coding: utf-8 -*-
"""
created on: 2024-03-22
@author:    Jasper Heuer
use:        reproject/resample Landsat imagery to common grid
"""

# import packages ==================================================================================

import os
from osgeo import gdal
import numpy as np

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)
           
# warp raster with gdal ============================================================================

dst_crs = "EPSG:32624" # define destination CRS
res = 30 # define resolution in meters

fn_in = "./Landsat/LT05_231014_19880831.tif"
fn_out = "./Landsat/test_reproj.tif"
fn_mask_raster = "./Masks/mask.tif"

# get corner coordinates for mask:
mask = gdal.Open(fn_mask_raster)
xmin, ymax = mask.GetGeoTransform()[0], mask.GetGeoTransform()[3]

mask = None # set to none

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

# read reprojected data and adjust GeoTransform to exactly align with common grid:
ds2 = gdal.Open(fn_out)
print(ds2.GetGeoTransform())

ds2.SetGeoTransform([xmin, res, 0.0, ymax, 0.0, -res]) # change GeoTransform
print(ds2.GetGeoTransform())

# write to disk:
driver = gdal.GetDriverByName("GTiff")
moved = driver.CreateCopy("./Landsat/LT05_231014_19880831_resample.tif", ds2)

# set data to none:
ds2 = None
moved = None