# -*- coding: utf-8 -*-
"""
created on: 2024-03-11
@author:    Jasper Heuer
use:        select pixels on class border
"""

# import packages ==================================================================================

import os
import numpy as np
import rasterio
from rasterio.plot import show

base_path = "C:/Jasper/Master/Thesis/Data/Landsat/"
os.chdir(base_path)

# import data ======================================================================================

img = rasterio.open("./1985-09-08/1985_reclass.tif")
show(img)

full_img = img.read()
band1 = img.read(1)

# locate border pixels =============================================================================

borders = np.zeros_like(band1, dtype=int) # create new array for to write borders
x_size, y_size = np.shape(borders)[0],np.shape(borders)[1] # get extent

# Iterate over pixels and classify based on border selection:
for i in range(x_size):
    for j in range(y_size):
        # select class 1 (bedrock):
        if band1[i,j] == 1:
            slice = band1[i-1:i+2,j-1:j+2] # get k1 neighborhood
            if np.any(slice == 2):
                borders[i,j] = 4 # update border pixel value
            if np.any(slice == 3):
                borders[i,j] = 5
        # select class 2 (ice):
        if band1[i,j] == 2:
            slice = band1[i-1:i+2, j-1:j+2]
            if np.any(slice == 3):
                borders[i,j] = 6
            if np.any(slice == 1):
                borders[i,j] = 7
        # select class 3 (snow):
        if band1[i,j] == 3:
            slice = band1[i-1:i+2, j-1:j+2]
            if np.any(slice == 2):
                borders[i,j] = 8
            if np.any(slice == 1):
                borders[i,j] = 9
                
show(borders) # plot border array

# save border array as GeoTiff:
with rasterio.open(
    "./1985-09-08/1985_borders.tif",
    mode="w",
    driver="GTiff",
    height=full_img.shape[1],
    width=full_img.shape[2],
    count=full_img.shape[0],
    dtype=full_img.dtype,
    crs=img.crs,
    transform=img.transform
    ) as dst:
        dst.write(borders, 1)
    