# -*- coding: utf-8 -*-
"""
created on: 2024-05-24
@author:    Jasper Heuer
use:        create mask for MAR resample and cropping of raw data
"""

# import packages ==================================================================================

import os
import numpy as np
from osgeo import gdal, osr, ogr

# set working directory ============================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

# create raster mask ===============================================================================

driver = gdal.GetDriverByName("GTiff")

# set coordinates (by adding 5000m * np.sqrt(2) in each direction to get all MAR pixels):
xmin = 547515 - (5000 * np.sqrt(2)) # = length of diagonal of 5000m pixel
xmax = 558795 + (5000 * np.sqrt(2))
ymin = 7283505 - (5000 * np.sqrt(2))
ymax = 7290015 + (5000 * np.sqrt(2))

# set metadata:
outfn = "./Masks/MAR_mask.tif"
nbands = 1
xres = 30
yres = -30
dtype = gdal.GDT_Int16 

# calculate raster height/width in pixel:
xsize = abs(int((xmax-xmin) / xres))
ysize = abs(int((ymax-ymin) / yres))

# create new raster:
ds = driver.Create(outfn, xsize, ysize, nbands, dtype)
ds.SetProjection("EPSG:32624")
ds.SetGeoTransform([xmin, xres, 0, ymax, 0, yres])
ds.GetRasterBand(1).Fill(1) # value of raster (and later on shapefile) mask
ds.GetRasterBand(1).SetNoDataValue(np.nan)

# FlushCache to write to disk and set data to none:
ds.FlushCache()
ds = None

# polygonize raster mask for WGS84 =================================================================

# read mask file:
src = gdal.Open(outfn) # open mask raster
srcband = src.GetRasterBand(1) # get first (and only) band

# define driver:
shape_driver = ogr.GetDriverByName("ESRI Shapefile")
dst = shape_driver.CreateDataSource("./Masks/MAR_mask_UTM-24N.shp")

# set CRS:
sp_ref = osr.SpatialReference()
sp_ref.SetFromUserInput('EPSG:32624')

# create new layer:
dst_layername = "mask"
dst_layer = dst.CreateLayer(dst_layername, srs = sp_ref)

# create field in attribute table:
fld = ogr.FieldDefn("mask", ogr.OFTInteger)
dst_layer.CreateField(fld)
dst_field = dst_layer.GetLayerDefn().GetFieldIndex("mask")

# polygonize raster to shapefile: 
gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None)

# set data to none:
dst.FlushCache()
src = None
dst = None
