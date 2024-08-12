# -*- coding: utf-8 -*-
"""
created on: 2024-03-26
@author:    Jasper Heuer
use:        1) crop imagery to glacier extent        
            2) classify glacier surface into snow/ice/bedrock based on simple threshold values
            3) detect border pixels between classes
"""

# import packages ==================================================================================

import os
import glob
import rasterio
import numpy as np
from osgeo import gdal
from rasterio.plot import show

# define variables =================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

fn_mask = "./Masks/mittivakkat_outline.shp"
dst_crs = "EPSG:32624"
res = 30

# batch crop to Mittivakkat mask ===================================================================

file_list = glob.glob("./Landsat/Landsat_05/Resampled/" + "*.tif", recursive=True)

# create list of dates:
date_list = []
for i in range(0, np.size(file_list)):
    date_list.append(file_list[i].split("\\")[1][0:13])

# batch crop imagery
for i in range(0, np.size(file_list)):
    img_in = file_list[i]
    img_out = "./Landsat/Landsat_05/Cropped/" + str(date_list[i]) + "_cropped.tif"
    
    img_ds = gdal.Open(img_in)
    
    img_crop = gdal.Warp(img_out, img_ds,
                         cutlineDSName=fn_mask,
                         cropToCutline=True) 
    
    img_ds = None
    img_crop = None
    img_out = None

# batch classify surface ===========================================================================

file_list2 = glob.glob("./Landsat/Landsat_05/Cropped/" + "*.tif")

# create list of dates:
date_list2 = []
for i in range(0, np.size(file_list2)):
    date_list2.append(file_list2[i].split("\\")[1][0:13])
    
# batch reclassify imagery/detect pixels:
for i in range(0, np.size(file_list2)):  
    img_ds = rasterio.open(file_list2[i])
    img_full = img_ds.read()
    band_01 = img_ds.read(1)
    
    # relassify:
    reclass = np.where((band_01 >= 60000), 3,
                       np.where((band_01 >= 15000), 2,
                                np.where((band_01 > 0), 1, 0)))
    
    show(reclass) # plot reclass array
    
    # detect pixels:
    borders = np.zeros_like(band_01, dtype=int)
    x_size, y_size = np.shape(borders)[0], np.shape(borders)[1]
    
    for k in range(x_size):
        for j in range(y_size):
            # select class 1 (bedrock):
            if reclass[k,j] == 1:
                slice = reclass[k-1:k+2,j-1:j+2] # get k1 neighborhood
                if np.any(slice == 2):
                    borders[k,j] = 4 # update border pixel value
                if np.any(slice == 3):
                    borders[k,j] = 5
            # select class 2 (ice):
            if reclass[k,j] == 2:
                slice = reclass[k-1:k+2, j-1:j+2]
                if np.any(slice == 3):
                    borders[k,j] = 6
                if np.any(slice == 1):
                    borders[k,j] = 7
            # select class 3 (snow):
            if reclass[k,j] == 3:
                slice = reclass[k-1:k+2, j-1:j+2]
                if np.any(slice == 2):
                    borders[k,j] = 8
                if np.any(slice == 1):
                    borders[k,j] = 9
                    
    show(borders) # plot border array

    # save reclass array as GeoTiff:
    with rasterio.open(
        "./Landsat/Landsat_05/Reclassified/" + str(date_list[i]) + "_reclass.tif",
        mode="w",
        driver="GTiff",
        height=img_full.shape[1],
        width=img_full.shape[2],
        count=img_full.shape[0],
        dtype=img_full.dtype,
        crs=img_ds.crs,
        transform=img_ds.transform
        ) as dst:
            dst.write(reclass, 1)
            
    # save border array as GeoTiff:
    with rasterio.open(
        "./Landsat/Landsat_05/Borders/" + str(date_list[i]) + "_borders.tif",
        mode="w",
        driver="GTiff",
        height=img_full.shape[1],
        width=img_full.shape[2],
        count=img_full.shape[0],
        dtype=img_full.dtype,
        crs=img_ds.crs,
        transform=img_ds.transform
        ) as dst:
            dst.write(borders, 1)