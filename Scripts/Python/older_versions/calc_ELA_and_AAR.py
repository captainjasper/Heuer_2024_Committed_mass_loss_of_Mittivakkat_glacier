# -*- coding: utf-8 -*-
"""
created on: 2024-03-20
@author:    Jasper Heuer
use:        calculate ELA and AAR
"""

# import packages ==================================================================================

import os
import rasterio
import numpy as np
from osgeo import gdal

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data"
os.chdir(base_path)

border_fn = "./Landsat/1985-09-08/1985_borders.tif"
dem_fn = "./Arctic_DEM/DEM_crop.tif"
reclass_fn = "./Landsat/1985-09-08/1985_reclass.tif"

dem_ds = rasterio.open(dem_fn)
dem = dem_ds.read(1)

border_ds = rasterio.open(border_fn)
border = border_ds.read(1)

reclass_ds = rasterio.open(reclass_fn)
reclass = reclass_ds.read(1)

# calculate ELA ====================================================================================
    
heights = []
x_coords = []
y_coords = []
    
for i in range(0, border.shape[0]):
    for j in range(0, border.shape[1]):
        if border[i,j] == 8:
            heights.append(dem[i,j])
            x_coords.append(j)
            y_coords.append(i)
            
ELA_array = np.array((x_coords, y_coords, heights)).T
ELA = np.mean(ELA_array[:, 2])
print("ELA: " + str(int(ELA)) + "m")

# calculate AAR ====================================================================================

classes, counts = np.unique(reclass.reshape(-1, 1),
                            return_counts=True,
                            axis=0)

AAR = counts[3]/(counts[2]+counts[3])
print("AAR (ice vs. snow): " + str("%0.2f" % round(AAR*100, 2)) + "%")

AAR_glacier_outline = counts[3]/(counts[1]+counts[2]+counts[3])
print("AAR (glacier outline: " + str("%0.2f" % round(AAR_glacier_outline*100, 2)) + "%")
