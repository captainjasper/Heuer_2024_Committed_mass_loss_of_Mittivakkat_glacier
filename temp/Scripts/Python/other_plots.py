# -*- coding: utf-8 -*-
"""
created on: 2024-06-21
@author:    Jasper Heuer
use:        create plots for thesis
"""

# import packages ==================================================================================

import os
import rasterio
import numpy as np
import pandas as pd
import statsmodels.api as sma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

df = pd.read_csv("./CSV/complete_table_latest.csv", sep=",")
df = df.drop("Unnamed: 0", axis=1)

# plot accumulation and ablation area ==============================================================

df["Total_area"] = df["Accumulation_area"] + df["Ablation_area"]

# define X and y:
X = sma.add_constant(df[["Year"]].drop([9, 23, 25])) # double brackets needed for LinearRegression
y = df[["Total_area"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

fig, ax1 = plt.subplots()

ax1.bar(df["Year"], df["Accumulation_area"], color="paleturquoise", label="Accumulation area")
ax1.bar(df["Year"], df["Ablation_area"], bottom=df["Accumulation_area"], 
        color="coral", label="Ablation area")
ax1.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
ax1.plot(X["Year"], upper, color="black", linestyle="dotted")
ax1.plot(X["Year"], lower, color="black", label="95% confidence interval", linestyle="dotted")
ax1.plot(df["Year"], df["Total_area"], color="black", label="Glacier area")

ax2 = ax1.twinx()

ax2.plot(df["Year"], df["AAR"], color="green", label="AAR")
ax2.set_ylim(0, 2)
ax2.yaxis.set_major_locator(ticker.FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1]))
ax2.set_ylabel("AAR")
ax2.yaxis.set_label_coords(1.09, 0.25)

plt.title("Accumulation and ablation area over time")
ax1.set_ylabel("Area in km²")
fig.legend(loc="lower center", bbox_to_anchor = (0.01, -0.15, 1, 1), ncols=2)
plt.xlim(1984, 2024)
plt.savefig("./Plots/accumulation_ablation_area.png", bbox_inches="tight", dpi=300)
plt.show()

np.nanmean(df["Total_area"][11:39])

# check assumptions of regression ==================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["Year"], res)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (AAR, linear regression)")
plt.legend()
plt.show()

# check independence using the Durbin-Watson test:
durbin_stat = durbin_watson(res) # if value outside of 1.5 - 2.5 interval -> autocorrelation
if durbin_stat < 2.5 and durbin_stat > 1.5:
    print("No autocorrelation of residuals detected")
else:
    print("Autocorrelation of residuals detected")

# check homoscedasticity of residuals:
plt.scatter(model.predict(X), res)
plt.xlabel("Predicted AAR")
plt.ylabel("Residual AAR")
plt.title("Residuals vs. model predicitons (AAR, linear regression)")
plt.show()

bp_stat = het_breuschpagan(res, X)
if bp_stat[1] > 0.05:
    print("Homoscedasticity of residuals")
else:
    print("Heteroscedasticity of residuals")

# check normal distribution of residuals:
plt.hist(res)
plt.xlabel("Residual value")
plt.ylabel("Number of instances")
plt.title("Residuals histogram (AAR, linear regression)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution") 
    
# check RMSE:
print("RMSE: " + str(np.sqrt(np.mean(sum(res**2)))))

# elevation-binned area ============================================================================

# import DEM:
dem_fn = "./Arctic_DEM/DEM_crop.tif"
dem_ds = rasterio.open(dem_fn)
dem = dem_ds.read(1)

# list of bin ranges:
bins_list = np.arange(150, 1100, 50)

# plot histogram:
plt.hist(dem.flatten(), bins=bins_list, rwidth=0.8, color="steelblue")
plt.title("Elevation histogram")
plt.xlabel("Elevation in meters")
labels=(0, 0.5, 1.0, 1.5, 2.0)
plt.yticks((0, 555.55, 1111.11, 1666.66, 2222.22), labels) # convert pixel count to km²
plt.ylabel("Area in km²")
plt.show()

dem_df = pd.DataFrame(dem.flatten(), columns=["DEM"]).dropna()
quantile = dem_df["DEM"].quantile(q=0.26)

np.nanmean(df["ELA"])

# Landsat imagery counts ===========================================================================

satellites = ("Landsat 5", "Landsat 6", "Landsat 7", "Landsat 8", "Landsat 9", "Total")
counts = {
    'No filter': (70, 0, 77, 83, 15, 245),
    'CF Mask': (48, 0, 52, 62, 11, 173),
    'Cloud cover': (10, 0, 32, 26, 7, 75),
}

x = np.arange(len(satellites))  # the label locations
width = 0.3  # the width of the bars
multiplier = 0

text_str = "RIP Landsat 6"

fig, ax = plt.subplots(layout='constrained')

color_list = ["lightcoral", "cornflowerblue", "orchid"]

i = 0
for filter_type, measurement in counts.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=filter_type, color=color_list[i])
    ax.bar_label(rects, padding=3)
    multiplier += 1
    i += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scene counts')
ax.set_title('Number of retrieved scenes per satellite')
ax.set_xticks(x + width, satellites)
ax.legend(loc="lower center", bbox_to_anchor = (0.01, -0.22, 1, 1), ncols=3)
plt.text(1.25, 35, text_str, rotation="vertical")
ax.set_ylim(0, 270)

plt.savefig("./Plots/image_counts.png", bbox_inches="tight", dpi=300)
plt.show()