# -*- coding: utf-8 -*-
"""
created on: 2024-06-14
@author:    Jasper Heuer
use:        analyze AAR-SMB relationship through time
"""

# import packages ==================================================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sma
import matplotlib.pyplot as plt

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

# read data:
df = pd.read_csv("./CSV/complete_table_latest.csv", sep=",") # latest version by default
df = df.drop("Unnamed: 0", axis=1)

# AAR-SMB over time (original data) ================================================================

year_list = []
model_list = []
coef_list = []
intercept_list = []
AAR0_list = []
r2_list = []

for i in range(0, len(df) - 10):
    y = df["AAR"][i:i+10].dropna()
    index = y.index.to_list() 
    
    X = sma.add_constant(df["SMB"][index])
    
    # create linear regression models:
    model = sma.OLS(y, X).fit()
    print(model.summary())
    
    # save results:
    coef = model.params["SMB"]
    intercept = model.params["const"]
    r2 = model.rsquared
    
    # add to lists:
    year_list.append(df["Year"][i+10])
    model_list.append(model)
    coef_list.append(coef)
    intercept_list.append(intercept)
    AAR0_list.append(float(model.predict(np.array([1, 0]))))
    r2_list.append(r2)
    
aar0_df = pd.DataFrame(np.array([year_list, coef_list, intercept_list, AAR0_list, r2_list]).T,
                       columns=["Year", "Coefficient", "Intercept", "AAR0", "r2"])
    
# AAR-SMB over time (WGMS data) ====================================================================

wgms_year_list = []
wgms_model_list = []
wgms_coef_list = []
wgms_intercept_list = []
wgms_AAR0_list = []
wgms_r2_list = []

for i in range(10, len(df) - 10):
    X = sma.add_constant(df[["WGMS_SMB"]][i:i+10]).dropna()
    y = df["WGMS_AAR"][i:i+10].dropna()
    
    # create linear regression models:
    model = sma.OLS(y, X).fit()
    print(model.summary())
    
    # save results:
    coef = model.params["WGMS_SMB"]
    intercept = model.params["const"]
    r2 = model.rsquared
    
    # add to lists:+
    wgms_year_list.append(df["Year"][i+10])
    wgms_model_list.append(model)
    wgms_coef_list.append(coef)
    wgms_intercept_list.append(intercept)
    wgms_AAR0_list.append(float(model.predict(np.array([1, 0]))))
    wgms_r2_list.append(r2)
    
wgms_aar0_df = pd.DataFrame(np.array([wgms_year_list, wgms_coef_list, wgms_intercept_list, 
                            wgms_AAR0_list, wgms_r2_list]).T, 
                            columns=["Year", "Coefficient", "Intercept", "AAR0", "r2"])

# plotting results =================================================================================

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# AAR0 comparison:
plot_01 = ax1.plot(aar0_df["Year"], aar0_df["AAR0"], color="black",
                   label="$AAR_{0}$" + " (original data)")
plot_02 = ax1.plot(wgms_aar0_df["Year"], wgms_aar0_df["AAR0"], color="black", linestyle="--",
                   label="$AAR_{0}$" + " (WGMS data)")
ax1.set_ylabel("$AAR_{0}$")

# R² comparison:
plot_03 = ax2.plot(aar0_df["Year"], aar0_df["r2"], color="red", label="R² (original data)")
plot_04 = ax2.plot(wgms_aar0_df["Year"], wgms_aar0_df["r2"], color="red", label="R² (WGMS data)",
         linestyle="--")
ax2.set_ylabel("R²", color="red")
ax2.tick_params(axis="y", colors="red")

plt.title("$AAR_{0}$" + " and R² over time")

lns = plot_01 + plot_02 + plot_03 + plot_04
fig.legend(bbox_to_anchor = (0.85, 0.05), ncols=2)
fig.savefig("./Plots/aar0_moving_window.png", bbox_inches="tight", dpi=300)
plt.show()

# lowest and highest AAR0 plot (original data):
plt.scatter(df["SMB"][20:30], df["AAR"][20:30], label="Low " + "$AAR_{0}$" +" data", color="blue")
plt.scatter(df["SMB"][22:32], df["AAR"][22:32], label="High " + "$AAR_{0}$" +" data", color="red")
plt.scatter(df["SMB"][22:30], df["AAR"][22:30], label="Overlap data", color="purple")
plt.plot([-5000, 200], model_list[19].predict([[1, -5000], [1, 200]]), 
         label="Low " + "$AAR_{0}$" +" trend", linestyle="--", color="blue")
plt.plot([-5000, 200], model_list[21].predict([[1, -5000], [1, 200]]), 
         label="High " + "$AAR_{0}$" +" data", linestyle="--", color="red")

# label individual points:
years = df["Year"][20:32]

for (xi, yi, year) in zip(df["SMB"][20:32], df["AAR"][20:32], years):
    plt.text(xi, yi, year, va="top", ha="left", size=9)

r2_str = "R² (high " + "$AAR_{0}$" + " model): " + str(round(r2_list[21], 3))\
    + "\n" + "R² (low " + "$AAR_{0}$" + " model): " + str(round(r2_list[19], 3))

plt.xlim(-4800, 200)
plt.ylim(-0.05, 1.1)
plt.axvline(x=0, color="black", linestyle="--")
plt.title("Comparsion of highest and lowest " + "$AAR_{0}$" + " 10-year windows")
plt.xlabel("SMB in mm w.eq.")
plt.ylabel("AAR")
plt.text(-2900, 0.92, r2_str)
plt.legend()
plt.savefig("./Plots/lowest_highest_aar0.png", dpi=300)
plt.show()

# lowest and highest AAR0 plot (WGMS):
plt.scatter(df["WGMS_SMB"][20:30], df["WGMS_AAR"][20:30], 
            label="Low " + "$AAR_{0}$" +" data", color="blue")
plt.scatter(df["WGMS_SMB"][11:21], df["WGMS_AAR"][11:21], 
            label="High " + "$AAR_{0}$" +" data", color="red")
# plt.scatter(df["WGMS_SMB"][11:30], df["AAR"][11:30], label="Overlap data", color="purple")
plt.plot([-2600, 500], wgms_model_list[9].predict([[1, -2600], [1, 500]]), 
         label="Low " + "$AAR_{0}$" +" trend", linestyle="--", color="blue")
plt.plot([-2600, 500], wgms_model_list[0].predict([[1, -2600], [1, 500]]), 
         label="High " + "$AAR_{0}$" +" trend", linestyle="--", color="red")

# label individual points:
tl_list = [11, 13, 14, 17, 18, 19, 20, 24, 25, 29]
br_list = [12, 15, 16, 21, 22]
c_list = [28, 26]
bl_list = [23, 27]
   
for (xi, yi, tl_list) in zip(df["WGMS_SMB"][tl_list], df["WGMS_AAR"][tl_list], df["Year"][tl_list]): 
    plt.text(xi, yi, tl_list, va="bottom", ha="right", size=9)
for (xi, yi, br_list) in zip(df["WGMS_SMB"][br_list], df["WGMS_AAR"][br_list], df["Year"][br_list]): 
    plt.text(xi, yi, br_list, va="top", ha="left", size=9)
for (xi, yi, c_list) in zip(df["WGMS_SMB"][c_list], df["WGMS_AAR"][c_list], df["Year"][c_list]): 
    plt.text(xi, yi, c_list, va="top", ha="center", size=9)  
for (xi, yi, bl_list) in zip(df["WGMS_SMB"][bl_list], df["WGMS_AAR"][bl_list], df["Year"][bl_list]): 
    plt.text(xi, yi, bl_list, va="top", ha="right", size=9)
    
r2_str = "R² (high " + "$AAR_{0}$" + " model): " + str(round(wgms_r2_list[0], 3))\
    + "\n" + "R² (low " + "$AAR_{0}$" + " model): " + str(round(wgms_r2_list[9], 3))
    
# plt.xlim(-3000, 200)
# plt.ylim(-0.05, 0.95)
plt.axvline(x=0, color="black", linestyle="--")
plt.title("Comparsion of highest and lowest " + "$AAR_{0}$" + " 10-year windows (WGMS)")
plt.xlabel("SMB in mm w.eq.")
plt.ylabel("AAR")
plt.legend()
plt.xlim(-2599, 499)
plt.ylim(-0.45, 0.9)
plt.text(-1400, -0.38, r2_str)
plt.savefig("./Plots/lowest_and_highest_aar0_wgms.png", dpi=300)
plt.show()
