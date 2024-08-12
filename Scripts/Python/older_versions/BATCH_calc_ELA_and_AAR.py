# -*- coding: utf-8 -*-
"""
created on: 2024-03-26
@author:    Jasper Heuer
use:        1) calculate ELA and AAR for glacier
            2) export data as CSV table
"""

# import packages ==================================================================================

import os
import glob
import rasterio
import numpy as np
import pandas as pd

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

# import DEM:
dem_fn = "./Arctic_DEM/DEM_crop.tif"
dem_ds = rasterio.open(dem_fn)
dem = dem_ds.read(1)

# create list of reclass and border rasters:
reclass_list = glob.glob("./Landsat/Landsat_05/Reclassified/" + "*.tif", recursive=True)
border_list = glob.glob("./Landsat/Landsat_05/Borders/" + "*.tif", recursive=True)

# calculate ELA and AAR ============================================================================

year_list = []
month_list = []
day_list = []
ELA_list = []
AAR_list = []

# loop over dates:
for i in range(0, np.size(reclass_list)):
    border_ds = rasterio.open(border_list[i])
    border = border_ds.read(1)
    
    reclass_ds = rasterio.open(reclass_list[i])
    reclass = reclass_ds.read(1)
    
    year_list.append(border_list[i].split("\\")[1][5:9])
    month_list.append(border_list[i].split("\\")[1][9:11])
    day_list.append(border_list[i].split("\\")[1][11:13])

    # check that we are using the same date for ELA and AAR:
    if border_list[i].split("\\")[1][0:13] == reclass_list[i].split("\\")[1][0:13]:
        heights = []
        x_coords = []
        y_coords = []
            
        # calculate ELA:
        for k in range(0, border.shape[0]):
            for j in range(0, border.shape[1]):
                if border[k,j] == 8:
                    heights.append(dem[k,j])
                    x_coords.append(j)
                    y_coords.append(k)
                    
        ELA_array = np.array((x_coords, y_coords, heights)).T
        ELA = np.mean(ELA_array[:, 2])
        
        # create dictionaray with pixel counts for each class:
        classes, counts = np.unique(reclass, return_counts=True)
        classes_dict = dict(zip(classes, counts))
        
        # calculate AAR (and handle special cases):
        if 2 in classes_dict: # 2 = snow pixels
            if 3 in classes_dict: # 3 = ice pixels
                total_area = classes_dict[2] + classes_dict[3]
                accumulation_area = classes_dict[3]
                AAR = accumulation_area/total_area
                print("AAR calculated at date: " 
                      + str(year_list[i]) + str(month_list[i]) + str(day_list[i]))
            else:
                AAR = 0
                print("No ice pixels identfied at date: "
                      + str(year_list[i]) + str(month_list[i]) + str(day_list[i]))
        elif 3 in classes_dict:
            AAR = 1
            print("No snow pixels identified at date: " 
                  + str(year_list[i]) + str(month_list[i]) + str(day_list[i]))
        else:
            AAR = np.nan
            print("No snow or ice pixels identified at date: " 
                  + str(year_list[i]) + str(month_list[i]) + str(day_list[i]))
            pass 

        # create output lists:
        AAR_list.append(AAR)
        ELA_list.append(ELA)
    
    else:
        print("Border and reclass rasters not from the same date!")
        pass
        
# create output array and write to disk:
data_array = np.array((year_list, month_list, day_list, ELA_list, AAR_list)).T
df = pd.DataFrame(data=data_array, columns=["year", "month", "day", "ELA", "AAR"])
df.to_csv("./Landsat/Landsat_05/ELA_AAR_table.csv", sep=",")
