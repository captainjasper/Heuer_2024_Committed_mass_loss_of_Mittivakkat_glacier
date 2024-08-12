# -*- coding: utf-8 -*-
"""
created on: 2024-05-29
@author:    Jasper Heuer
use:        investigate statistical relationships between AAR and SMB
"""

# import packages ==================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/"
os.chdir(base_path)

df = pd.read_csv("./Data/CSV/complete_table_latest.csv")
df = df.drop("Unnamed: 0", axis=1)

# melt season length ===============================================================================
plt.plot(df["Year"], df["Melt_season"], "o-")
plt.title("Melt season length (Mittivakkat)")
plt.ylabel("Season length in days")
plt.show()

# ELA comparison ===================================================================================

fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("ELA in meters", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["ELA"], "o-", color="black", label="ELA")
plot_02 = ax1.plot(df["Year"], df["WGMS_ELA"], "o--", color="red", label="ELA (WGMS)")

# create legend:
lns = plot_01 + plot_02
labels = [l.get_label() for l in lns]
plt.legend(lns, labels) # loc=0, bbox_to_anchor = (1, -0.15), ncols=4)

# finish plot and write to disk:
plt.title("ELA comparison (Mittivakkat)")
plt.show()

# AAR comparison ===================================================================================

fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["AAR"], "o-", color="black", label="AAR")
plot_02 = ax1.plot(df["Year"], df["WGMS_AAR"], "o--", color="blue", label="AAR (WGMS)")

# create legend:
lns = plot_01 + plot_02
labels = [l.get_label() for l in lns]
plt.legend(lns, labels) # loc=0, bbox_to_anchor = (1, -0.15), ncols=4)

# finish plot and write to disk:
plt.title("AAR comparison (Mittivakkat)")
plt.show()

# SMB comparison ===================================================================================

fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("SMB in mm w.e.", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["SMB"], "o-", color="black", label="SMB")
plot_02 = ax1.plot(df["Year"], df["WGMS_SMB"], "o--", color="green", label="SMB (WGMS)")

# create legend:
lns = plot_01 + plot_02
labels = [l.get_label() for l in lns]
plt.legend(lns, labels) # loc=0, bbox_to_anchor = (1, -0.15), ncols=4)

# finish plot and write to disk:
plt.title("SMB comparison (Mittivakkat)")
plt.show()

# visual correlation between original data and WGMS ================================================

# AAR scatterplot:
plt.scatter(df["AAR"], df["WGMS_AAR"])
plt.title("AAR (WGMS) vs. AAR")
plt.xlabel("AAR")
plt.ylabel("AAR (WGMS)")
plt.show()

# ELA scatterplot
plt.scatter(df["ELA"], df["WGMS_ELA"])
plt.title("ELA (WGMS) vs. ELA")
plt.xlabel("ELA in meters")
plt.ylabel("ELA (WGMS) in meters")
plt.show()

# SMB scatterplot
plt.scatter(df["SMB"], df["WGMS_SMB"])
plt.title("SMB (WGMS) vs. SMB")
plt.xlabel("SMB in mm w.e.")
plt.ylabel("SMB (WGMS) in mm w.e.")
plt.show()

# histograms to visually assess normal distribution ================================================

# AAR histogram:
plt.hist([df["AAR"], df["WGMS_AAR"]], color=["black", "blue"], bins = 10)
plt.xlabel("AAR")
plt.ylabel("Number of instances")
plt.legend(["AAR", "AAR (WGMS)"])
plt.show()

# ELA histogram:
plt.hist([df["ELA"], df["WGMS_ELA"]], color=["black", "red"], bins = 10)
plt.xlabel("ELA in meters")
plt.ylabel("Number of instances")
plt.legend(["ELA", "ELA (WGMS)"])
plt.show()

# SMB histogram:
plt.hist([df["SMB"], df["WGMS_SMB"]], color=["black", "green"], bins = 10)
plt.xlabel("SMB in mm w.e.")
plt.ylabel("Number of instances")
plt.legend(["SMB", "SMB (WGMS)"])
plt.show()

# test for normal distribution =====================================================================

for i in range(0, np.size(df, axis=1)):
    res = stats.shapiro(df.iloc[:, i], nan_policy="omit")
    print("p-value for column " + str(df.keys()[i]) + " = " + str(res[1]))
    if res[1] > 0.05:
        print("Data normally distributed")
    else:
        print("Data not normally distributed")

# AAR trend analysis ===============================================================================

X = df[["Year"]] # double brackets needed for LinearRegression
y_aar = df[["AAR"]].interpolate(method="linear")
y_aar_wgms = df[["WGMS_AAR"]].interpolate(method="linear").dropna()

# create linear regression models:
aar_model = LinearRegression().fit(X, y_aar)
aar_wgms_model = LinearRegression().fit(X[11:39], y_aar_wgms)

# save results:
AAR_coef = aar_model.coef_
AAR_intercept = aar_model.intercept_
AAR_WGMS_coef = aar_wgms_model.coef_
AAR_WGMS_intercept = aar_wgms_model.intercept_

# compute r2 score of models:
AAR_r2 = r2_score(y_true=y_aar, y_pred=aar_model.predict(X))
AAR_WGMS_r2 = r2_score(y_true=y_aar_wgms, y_pred=aar_wgms_model.predict(X[11:39]))

# plot original AAR with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["AAR"], "o-", color="black", label="AAR", alpha=0.7)
plot_02 = ax1.plot(df["Year"], aar_model.predict(X), color="black", 
                   label="Trendline", linestyle="--")
plot_03 = ax1.plot(df["Year"], df["WGMS_AAR"], "o-", color="blue", label="AAR (WGMS)", alpha=0.7)
plot_04 = ax1.plot(df["Year"][11:39], aar_wgms_model.predict(X [11:39]), color="blue", 
                   label="Trendline (WGMS)", linestyle="--")

# create legend:
lns = plot_01 + plot_02 + plot_03 + plot_04
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc="center", bbox_to_anchor = (0.5, -0.2), ncols=4)

# finish plot and write to disk:
plt.title("AAR with trendline (Mittivakkat)")
plt.show()

# ELA trend analysis ===============================================================================

X = df[["Year"]] # double brackets needed for LinearRegression
y_ela = df[["ELA"]].interpolate(method="linear")
y_ela_wgms = df[["WGMS_ELA"]].interpolate(method="linear").dropna()

# create linear regression models:
ela_model = LinearRegression().fit(X, y_ela)
ela_wgms_model = LinearRegression().fit(X[11:39], y_ela_wgms)

# save results:
ELA_coef = ela_model.coef_
ELA_intercept = ela_model.intercept_
ELA_WGMS_coef = ela_wgms_model.coef_
ELA_WGMS_intercept = ela_wgms_model.intercept_

# compute r2 score of models:
ELA_r2 = r2_score(y_true=y_ela, y_pred=ela_model.predict(X))
ELA_WGMS_r2 = r2_score(y_true=y_ela_wgms, y_pred=ela_wgms_model.predict(X[11:39]))

# plot original ELA with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("ELA in meters", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["ELA"], "o-", color="black", label="ELA", alpha=0.7)
plot_02 = ax1.plot(df["Year"], ela_model.predict(X), color="black", 
                   label="Trendline", linestyle="--")
plot_03 = ax1.plot(df["Year"], df["WGMS_ELA"], "o-", color="red", label="ELA (WGMS)", alpha=0.7)
plot_04 = ax1.plot(df["Year"][11:39], ela_wgms_model.predict(X [11:39]), color="red", 
                   label="Trendline (WGMS)", linestyle="--")

# create legend:
lns = plot_01 + plot_02 + plot_03 + plot_04
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc="center", bbox_to_anchor = (0.5, -0.2), ncols=4)

# finish plot and write to disk:
plt.title("ELA with trendline (Mittivakkat)")
plt.show()

# SMB trend analysis ===============================================================================

X = df[["Year"]] # double brackets needed for LinearRegression
y_smb = df[["SMB"]].interpolate(method="linear")
y_smb_wgms = df[["WGMS_SMB"]].interpolate(method="linear").dropna()

# create linear regression models:
smb_model = LinearRegression().fit(X, y_smb)
smb_wgms_model = LinearRegression().fit(X[11:39], y_smb_wgms)

# save results:
SMB_coef = smb_model.coef_
SMB_intercept = smb_model.intercept_
SMB_WGMS_coef = smb_wgms_model.coef_
SMB_WGMS_intercept = smb_wgms_model.intercept_

# compute r2 score of models:
SMB_r2 = r2_score(y_true=y_smb, y_pred=smb_model.predict(X))
SMB_WGMS_r2 = r2_score(y_true=y_smb_wgms, y_pred=smb_wgms_model.predict(X[11:39]))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("SMB in mm w.e.", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["SMB"], "o-", color="black", label="SMB", alpha=0.7)
plot_02 = ax1.plot(df["Year"], smb_model.predict(X), color="black", 
                   label="Trendline", linestyle="--")
plot_03 = ax1.plot(df["Year"], df["WGMS_SMB"], "o-", color="green", label="SMB (WGMS)", alpha=0.7)
plot_04 = ax1.plot(df["Year"][11:39], smb_wgms_model.predict(X [11:39]), color="green", 
                   label="Trendline (WGMS)", linestyle="--")

# create legend:
lns = plot_01 + plot_02 + plot_03 + plot_04
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc="center", bbox_to_anchor = (0.5, -0.2), ncols=4)

# finish plot and write to disk:
plt.title("SMB with trendline (Mittivakkat)")
plt.show()

# Melt season trend analysis =======================================================================

X = df[["Year"]] # double brackets needed for LinearRegression
y_melt = df[["Melt_season"]].interpolate(method="linear")

# create linear regression models:
melt_model = LinearRegression().fit(X, y_melt)

# save results:
melt_coef = melt_model.coef_
melt_intercept = melt_model.intercept_

# compute r2 score of models:
SMB_r2 = r2_score(y_true=y_melt, y_pred=melt_model.predict(X))

# plot original Melt season with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("Melt season length in days", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["Melt_season"], "o-", color="tab:blue", 
                   label="Melt season length", alpha=0.7)
plot_02 = ax1.plot(df["Year"], melt_model.predict(X), color="tab:blue", 
                   label="Trendline", linestyle="--")

# create legend:
lns = plot_01 + plot_02
labels = [l.get_label() for l in lns]
plt.legend(lns, labels) #, loc="center", bbox_to_anchor = (0.5, -0.2), ncols=4)

# finish plot and write to disk:
plt.title("Melt season length with trendline (Mittivakkat)")
plt.show()

# AAR-SMB relationship =============================================================================

# linear regression:
sorted_df = df.sort_values(by="SMB", axis=0, ascending=True)
X = sorted_df[["SMB"]].dropna() # double brackets needed for LinearRegression
y = sorted_df[["AAR"]].dropna()

# create linear regression models:
model = LinearRegression().fit(X, y)

# save results:
coef = model.coef_
intercept = model.intercept_

# compute r2 score of models:
r2 = r2_score(y_true=y, y_pred=model.predict(X))

model_str = "r2: " + str(round(r2, 3)) + "\n" + "coef: " + str(round(coef[0, 0], 6))\
    + "\n" "intercept: " + str(round(intercept[0], 3))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(sorted_df["SMB"], sorted_df["AAR"], color="black")
plot_02 = ax1.plot(X, model.predict(X), color="red", label="Trendline", linestyle="--")
plt.xlabel("Annual SMB in mm w.e.")
plt.title("AAR vs. Annual SMB (Mittivakkat)")
plt.axvline(x=0, color="black", linestyle="--")
plt.text(-4500, 0.5, model_str)
plt.show()

np.std(df["AAR"])

# polynomial regression:
X_poly = np.array(X)
y_poly = np.array(y)

poly = PolynomialFeatures(degree=2, include_bias=False)   

# create new array with X² features:
poly_features = poly.fit_transform(X_poly.reshape(-1, 1))

poly_model = LinearRegression()
poly_model.fit(poly_features, y_poly)

# calcuate model predicitons for y:
y_pred = poly_model.predict(poly_features)

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X_poly, y_poly, color="black")
plot_02 = ax1.plot(X, y_pred, color="red", label="Trendline", linestyle="--")
plt.xlabel("Annual SMB in mm w.e.")
plt.title("AAR vs. Annual SMB (Mittivakkat)")
plt.axvline(x=0, color="black", linestyle="--")
# plt.text(-4500, 0.5, model_str)
plt.show()

print(poly_model.coef_)
print(poly_model.intercept_)

np.array(X).reshape(-1)

# exponential regression:
sorted_no_zero_df = df[["Year", "SMB", "AAR"]].sort_values(by="SMB", axis=0, ascending=True).where(df["AAR"] > 0).dropna()
sorted_no_zero_df = np.array(sorted_no_zero_df)
exp_model = np.polyfit(sorted_no_zero_df[:,1], np.log(sorted_no_zero_df[:,2]), 1)
print(exp_model)


X_exp = np.arange(-5000, 500, 50)
y_exp = np.exp(exp_model[1]) * (np.exp(exp_model[0])**(X_exp))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(sorted_no_zero_df[:,1], sorted_no_zero_df[:,2], color="black")
plot_02 = ax1.plot(X_exp, y_exp, color="red", label="Trendline", linestyle="--")
plt.xlabel("Annual SMB in mm w.e.")
plt.title("AAR vs. Annual SMB (Mittivakkat)")
plt.axvline(x=0, color="black", linestyle="--")
# plt.text(-4500, 0.5, model_str)
plt.show()
    
# AAR-SMB relationship for WGMS data ===============================================================

X = df[["WGMS_SMB"]].dropna() # double brackets needed for LinearRegression
y = df[["WGMS_AAR"]].dropna()

# create linear regression models:
model = LinearRegression().fit(X, y)

# save results:
coef = model.coef_
intercept = model.intercept_

# compute r2 score of models:
r2 = r2_score(y_true=y, y_pred=model.predict(X))

model_str = "r2: " + str(round(r2, 3)) + "\n" + "coef: " + str(round(coef[0, 0], 6))\
    + "\n" "intercept: " + str(round(intercept[0], 3))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(df["WGMS_SMB"], df["WGMS_AAR"], color="black")
plot_02 = ax1.plot(X, model.predict(X), color="red", label="Trendline", linestyle="--")
plt.xlabel("Annual SMB in mm w.e.")
plt.title("AAR vs. Annual SMB (Mittivakkat) - WGMS version")
plt.axvline(x=0, color="black", linestyle="--")
plt.text(-2250, 0.5, model_str)
plt.show()

# SMB-SMB (WGMS) relationship ======================================================================

X = df[["SMB"]].interpolate(method="linear")[11:39] # double brackets needed for LinearRegression
y = df[["WGMS_SMB"]].interpolate(method="linear").dropna()

# create linear regression models:
model = LinearRegression().fit(X, y)

# save results:
coef = model.coef_
intercept = model.intercept_

# compute r2 score of models:
r2 = r2_score(y_true=y, y_pred=model.predict(X))

model_str = "r2: " + str(round(r2, 3)) + "\n" + "coef: " + str(round(coef[0, 0], 3))\
    + "\n" "intercept: " + str(round(intercept[0], 3)) 

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("SMB (WGMS)", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X, y, color="black")
plot_02 = ax1.plot(X, model.predict(X), color="red", label="Trendline", linestyle="--")
plt.xlabel("SMB (MAR)")
plt.title("SMB (WGMS) vs. SMB (MAR) (Mittivakkat)")
plt.text(-4500, -250, model_str)
plt.show()

