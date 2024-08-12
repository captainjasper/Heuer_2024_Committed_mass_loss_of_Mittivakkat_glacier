# -*- coding: utf-8 -*-
"""
created on: 2024-03-20
@author:    Jasper Heuer
use:        classify glacier surface using simple threshold
"""

# import packages ==================================================================================

import os
import rasterio
import numpy as np
from osgeo import gdal
from rasterio.plot import show

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data"
os.chdir(base_path)

dst_crs = "EPSG:32624"
res = 30

# crop Landsat image ===============================================================================

img_in = "./Landsat/1985-09-08/LT05_L2SP_231014_19850908_20200918_02_T1_SR_B1.TIF"
img_out = "./Landsat/1985-09-08/Band_01_crop.tif"

img_ds = gdal.Open(img_in)

img_crop = gdal.Warp(img_out, img_ds,
                       cutlineDSName="./Masks/mittivakkat_outline.shp", # cut by extend of mask
                       cropToCutline=True, dstNodata=np.nan) 

img_ds = None
img_crop = None
img_out = None

# classfiy surface =================================================================================

img2_in = "./Landsat/1985-09-08/Band_01_crop.tif"

band_01_ds = rasterio.open(img2_in)
band_01_full = band_01_ds.read()
band_01 = band_01_ds.read(1)

reclass = np.where((band_01 >= 60000), 3,
                   np.where((band_01 >= 15000), 2,
                            np.where((band_01 > 0), 1, 0)))
show(reclass)

# save border array as GeoTiff:
with rasterio.open(
    "./Landsat/1985-09-08/1985_reclass.tif",
    mode="w",
    driver="GTiff",
    height=band_01_full.shape[1],
    width=band_01_full.shape[2],
    count=band_01_full.shape[0],
    dtype=band_01_full.dtype,
    crs=band_01_ds.crs,
    transform=band_01_ds.transform
    ) as dst:
        dst.write(reclass, 1)
