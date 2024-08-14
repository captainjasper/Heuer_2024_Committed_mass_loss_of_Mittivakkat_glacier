# -*- coding: utf-8 -*-
"""
created on: 2024-03-19
@author:    Jasper Heuer
use:        1) create raster mask for glacier extent
            2) polygonize raster to WGS-84 shapefile
            3) polygonize raster to UTM-24N shapefile
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

# set coordinates:
xmin = 547515 
xmax = 558795
ymin = 7283505
ymax = 7290015  

# set metadata:
outfn = "./Masks/mask.tif"
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

src = gdal.Open(outfn) # open mask raster
srcband = src.GetRasterBand(1) # get first (and only) band

shape_driver = ogr.GetDriverByName("ESRI Shapefile")
dst = shape_driver.CreateDataSource("./Masks/mask_WGS84.shp")

sp_ref = osr.SpatialReference()
sp_ref.SetFromUserInput('EPSG:4326')

dst_layername = "mask"
dst_layer = dst.CreateLayer(dst_layername, srs = sp_ref)

# create field in attribute table:
fld = ogr.FieldDefn("mask", ogr.OFTInteger)
dst_layer.CreateField(fld)
dst_field = dst_layer.GetLayerDefn().GetFieldIndex("mask")

gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None)

# set data to none:
dst.FlushCache()
src = None
dst = None

# polygonize raster mask for UTM-24N ===============================================================

src2 = gdal.Open(outfn) # open mask raster
srcband2 = src2.GetRasterBand(1) # get first (and only) band

shape_driver = ogr.GetDriverByName("ESRI Shapefile")
dst2 = shape_driver.CreateDataSource("./Masks/mask_UTM-24N.shp")

sp_ref2 = osr.SpatialReference()
sp_ref2.SetFromUserInput('EPSG:32624')

dst_layername = "mask"
dst_layer = dst2.CreateLayer(dst_layername, srs = sp_ref2)

# create field in attribute table:
fld = ogr.FieldDefn("mask", ogr.OFTInteger)
dst_layer.CreateField(fld)
dst_field = dst_layer.GetLayerDefn().GetFieldIndex("mask")

gdal.Polygonize(srcband2, None, dst_layer, dst_field, [], callback=None)

# set data to none:
dst2.FlushCache()
src2 = None
dst2 = None
