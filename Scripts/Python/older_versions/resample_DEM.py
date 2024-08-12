# -*- coding: utf-8 -*-
"""
created on: 2024-03-15
@author:    Jasper Heuer
use:        1) reproject input-DEM to fit Landsat data grid using gdal.Warp
            2) reproject input-DEM to fit Landsat data grid using rasterio (not needed)
            3) crop DEM to Mittivakkat extent
"""

# import packages ==================================================================================

import os
from osgeo import gdal
import numpy as np
# import rasterio
# from rasterio.warp import calculate_default_transform, reproject, Resampling

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)
           
# warp raster with gdal ============================================================================

dst_crs = "EPSG:32624" # define destination CRS
res = 30 # define resolution in meters

fn_in = "./Arctic_DEM/merged_DEM.tif"
fn_out = "./Arctic_DEM/DEM_reproj.tif"
fn_mask = "./Masks/mask.tif"

# get corner coordinates for mask:
mask = gdal.Open(fn_mask)
xmin, ymax = mask.GetGeoTransform()[0], mask.GetGeoTransform()[3] # get top left coordinates

mask = None # set to none

ds = gdal.Open(fn_in) # read dataset
# inspect projection by typing ds.GetProjection() in the console- NOT the editor

# reproject to common grid:
ds_reproj = gdal.Warp(fn_out, ds, dstSRS=dst_crs,
                      xRes=res, yRes=-res, # specify resolution
                      cutlineDSName="./Masks/mask_UTM-24N.shp", # cut by extend of mask
                      cropToCutline=True, dstNodata=np.nan) 
# if error with mask, try mask_WGS84.shp instead - projection issues not fully understood yet

# set data to none:
ds = None
ds_reproj = None

# read reprojected data and adjust GeoTransform to exactly align with common grid:
ds2 = gdal.Open(fn_out)
print(ds2.GetGeoTransform())

ds2.SetGeoTransform([xmin, res, 0.0, ymax, 0.0, -res]) # change top left coordinates
print(ds2.GetGeoTransform())

# write to disk:
driver = gdal.GetDriverByName("GTiff")
moved = driver.CreateCopy("./Arctic_DEM/DEM_final.tif", ds2)

# set data to none:
ds2 = None
moved = None

# old version with rasterio ========================================================================

"""
dst_crs = "EPSG:32624"

with rasterio.open("./Arctic_DEM/14_44_1_1_2m_v4.1/14_44_1_1_2m_v4.1_browse.tif") as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height
    })
    
    with rasterio.open("./Arctic_DEM/14_44_1_1_2m_v4.1/14_44_1_1_2m_v4.1_browse_resample.tif", "w", **kwargs) as dst:
        for i in range(1, src.count+1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
"""

# crop DEM =========================================================================================

dem_in = "./Arctic_DEM/DEM_final.tif"
dem_out = "./Arctic_DEM/DEM_crop.tif"

dem_ds = gdal.Open(dem_in)

dem_crop = gdal.Warp(dem_out, dem_ds,
                     cutlineDSName="./Masks/mittivakkat_outline.shp",
                     cropToCutline=True, dstNodata=np.nan)

dem_ds = None
dem_crop = None