# -*- coding: utf-8 -*-
"""
created on: 2024-06-05
@author:    Jasper Heuer
use:        investigate correlation between variables
"""

# import packages ==================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/"
os.chdir(base_path)

df = pd.read_csv("./Data/CSV/complete_table_latest.csv")
df = df.drop("Unnamed: 0", axis=1)

# test for normal distribution =====================================================================

for i in range(0, np.size(df, axis=1)):
    res = stats.shapiro(df.iloc[:, i], nan_policy="omit")
    print("p-value for column " + str(df.keys()[i]) + " = " + str(res[1]))
    if res[1] > 0.05:
        print("Data normally distributed")
    else:
        print("Data not normally distributed")
              
# correlation scatterplots =========================================================================

corr_df = df[11:38].drop([23, 25])

# SMB correlation:
X = corr_df["SMB"]
y = corr_df["WGMS_SMB"]

corr_coef, p_value = stats.pearsonr(X, y)
corr_str = "Correlation coefficient: " + str(round(corr_coef, 3))\
            + "\n" + "p-Value: " + str(round(p_value, 4))
            
plt.scatter(X, y)
plt.plot(np.arange(-3500, 1001, 1), np.arange(-3500, 1001, 1), linestyle="--", color="red")
plt.xlim(-3500, 600)
plt.ylim(-3500, 600)
plt.gca().set_aspect("equal")
plt.xlabel("SMB in mm w.eq.")
plt.ylabel("WGMS SMB in mm w.eq.")
# plt.text(-3800, -4700, corr_str)
plt.title("SMB correlation")
plt.savefig("./Data/Plots/smb_correlation.png", bbox_inches="tight", dpi=300)
plt.show()

# ELA correlation:
X = corr_df["ELA"]
y = corr_df["WGMS_ELA"]

corr_coef, p_value = stats.kendalltau(X, y)
corr_str = "Correlation coefficient: " + str(round(corr_coef, 3))\
            + "\n" + "p-Value: " + str(round(p_value, 4))
            
plt.scatter(X, y)
plt.plot(np.arange(300, 1001, 1), np.arange(300, 1001, 1), linestyle="--", color="red")
plt.xlim(350, 950)
plt.ylim(350, 950)
plt.gca().set_aspect("equal")
plt.xlabel("ELA in meters")
plt.ylabel("WGMS ELA in meters")
# plt.text(380, 760, corr_str)
plt.title("ELA correlation")
plt.savefig("./Data/Plots/ela_correlation.png", bbox_inches="tight", dpi=300)
plt.show()

# AAR correlation:
X = corr_df["AAR"]
y = corr_df["WGMS_AAR"]

corr_coef, p_value = stats.kendalltau(X, y)
corr_str = "Correlation coefficient: " + str(round(corr_coef, 3))\
            + "\n" + "p-Value: " + str(round(p_value, 4))
            
plt.scatter(X, y)
plt.plot(np.arange(-0.2, 1.1, 0.1), np.arange(-0.2, 1.1, 0.1), linestyle="--", color="red")
plt.xlim(-0.1, 1)
plt.ylim(-0.1, 1)
plt.gca().set_aspect("equal")
plt.xlabel("AAR")
plt.ylabel("WGMS AAR")
# plt.text(0, 0.8, corr_str)
plt.title("AAR correlation")
plt.savefig("./Data/Plots/aar_correlation.png", bbox_inches="tight", dpi=300)
plt.show()

# hydrological year correlation with SMB:
fig = plt.figure(figsize=(6,6))
    
X = df["Hydro_year"]
y = df["SMB"]

corr_coef, p_value = stats.pearsonr(X, y)
corr_str = "Correlation coefficient: " + str(round(corr_coef, 3))\
            + "\n" + "p-Value: " + str(round(p_value, 4))
            
plt.scatter(X, y)
# plt.plot(np.arange(320, 420, 1), np.arange(500, 3500, 1), linestyle="--", color="red")
plt.xlabel("Length of hydrological year in days")
plt.ylabel("SMB in mm w.eq.")
# plt.text(325, -3500, corr_str)
plt.title("Correlation of SMB with length of hydrological year")
plt.savefig("./Data/Plots/smb_hydro_year_correlation.png", bbox_inches="tight", dpi=300)
fig.show()

# ELA-AAR correlation:
X = corr_df["AAR"]
y = corr_df["ELA"]

corr_coef, p_value = stats.kendalltau(X, y)
corr_str = "Correlation coefficient: " + str(round(corr_coef, 3))\
            + "\n" + "p-Value: " + str(round(p_value, 4))
            
# ELA-AAR (WGMS) correlation:
X = corr_df["WGMS_AAR"]
y = corr_df["WGMS_ELA"]

corr_coef, p_value = stats.kendalltau(X, y)
corr_str = "Correlation coefficient: " + str(round(corr_coef, 3))\
            + "\n" + "p-Value: " + str(round(p_value, 4))
            
plt.scatter(df["WGMS_AAR"], df["WGMS_ELA"])
