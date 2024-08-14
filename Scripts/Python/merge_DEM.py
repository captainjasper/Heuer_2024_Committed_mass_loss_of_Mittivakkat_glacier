# -*- coding: utf-8 -*-
"""
created on: 2024-03-18
@author:    Jasper Heuer
use:        merge raster datasets into single file
"""

# import packages ==================================================================================

import os
import glob
from osgeo import gdal

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/Arctic_DEM/"
os.chdir(base_path)

file_list = glob.glob("./raw/*.tif")

# merge rasters ====================================================================================

vrt = gdal.BuildVRT("./merged.vrt", file_list) # build virtual raster from file list
gdal.Translate("./merged_DEM.tif", vrt, xRes=2, yRes=-2) # specify resolution of output raster

# set data to none:
vrt = None 

# clean drive ======================================================================================

os.remove("./merged.vrt")
