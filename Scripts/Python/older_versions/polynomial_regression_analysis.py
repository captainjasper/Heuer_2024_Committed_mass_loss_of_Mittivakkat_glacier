# -*- coding: utf-8 -*-
"""
Created on: 2024-06-03
@author:    Jasper Heuer
use:        investigate relationship between AAR and SMB using polynomial linear regression (OLS)
"""

# import packages ==================================================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sma
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.stattools import durbin_watson
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/"
os.chdir(base_path)

# read data:
df = pd.read_csv("./Data/CSV/complete_table_latest.csv")
df = df.drop("Unnamed: 0", axis=1) # drop unwanted axis
df = df.drop([7]) # drop 1992 value
df = df.sort_values(by="SMB", axis=0, ascending=True) # sort values
df["SMB"] = df["SMB"] / 1000 # get m w.e. instead of mm

# polynomial regression (raw data) =================================================================

# define X and y:
X = np.array(df["SMB"].drop([9, 23, 25]))
y = np.array(df["AAR"].dropna())

# define polynomial model:
poly = PolynomialFeatures(degree=2, include_bias=False)   

# create new array with X² features:
poly_features = poly.fit_transform(X.reshape(-1, 1))

# add constant term to X for sma.OLS regression and redefine X and y as DataFrame:
X = pd.DataFrame(sma.add_constant(poly_features), columns=["const", "x", "x2"]) 
y = pd.DataFrame(y)

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef1 = model.params["x"]
coef2 = model.params["x2"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef1, 6))\
    + "\n" + "x²-coefficient: " + str(round(coef2, 6))\
    + "\n" + "Intercept: " + str(round(intercept, 3))\
    + "\n" + "Mean standard error: " + str(np.mean(round(std_err, 3)))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X["x"], y, color="black", label="Original data")
plot_02 = ax1.plot(X["x"], model.predict(X), color="red", label="Trendline", linestyle="--")
plot_03 = ax1.plot(X["x"], upper, color="red", label="95% confidence interval", linestyle="dotted")
plot_04 = ax1.plot(X["x"], lower, color="red", linestyle="dotted")
plt.xlabel("Annual SMB in m w.e.")
plt.title("Second-order polynomial AAR-SMB regression")
plt.axvline(x=0, color="black", linestyle="--")
plt.legend(bbox_to_anchor=(-0.1, -0.3, 1, 1), loc="lower left", ncols=3)
plt.savefig("./Data/Plots/aar_smb_poly_regression.png", bbox_inches="tight", dpi=300)
plt.show()

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["x"], res)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Annual SMB in m w.e.")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (raw data)")
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
plt.ylabel("AAR")
plt.title("Residuals vs. model predicitons (raw data)")
plt.show()

# check normal distribution of residuals:
plt.hist(res)
plt.xlabel("Residual value")
plt.ylabel("Number of instances")
plt.title("Residuals histogram (raw data)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution") 

# polynomial regression (square root transform) ====================================================

# define X and y:
X = np.array(df["SMB"].drop([9,23,25])
y = np.array(np.sqrt(df["AAR"].dropna()))

# define polynomial model:
poly = PolynomialFeatures(degree=2, include_bias=False)   

# create new array with X² features:
poly_features = poly.fit_transform(X.reshape(-1, 1))

# add constant term to X for sma.OLS regression and redefine X and y as DataFrame:
X = pd.DataFrame(sma.add_constant(poly_features), columns=["const", "x", "x2"])
y = pd.DataFrame(y)

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef1 = model.params["x"]
coef2 = model.params["x2"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef1, 6))\
    + "\n" + "x²-coefficient: " + str(round(coef2, 6))\
    + "\n" + "Intercept: " + str(round(intercept, 3))\
    + "\n" + "Mean standard error: " + str(np.mean(round(std_err, 3)))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X["x"], y, color="black")
plot_02 = ax1.plot(X["x"], model.predict(X), color="red", label="Trendline", linestyle="--")
plot_03 = ax1.plot(X["x"], upper, color="red", label="Upper", linestyle="dotted")
plot_04 = ax1.plot(X["x"], lower, color="red", label="Lower", linestyle="dotted")
plt.xlabel("Annual SMB in m w.e.")
plt.title("Polynomial regression (square root transformation)")
plt.axvline(x=0, color="black", linestyle="--")
plt.text(-4.5, 0.75, model_str, fontsize=9)
plt.show()

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["x"], res)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Annual SMB in m w.e.")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (square root transform)")
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
plt.ylabel("AAR")
plt.title("Residuals vs. model predicitons (square root transform)")
plt.show()

# check normal distribution of residuals:
plt.hist(res)
plt.xlabel("Residual value")
plt.ylabel("Number of instances")
plt.title("Residuals histogram (square root transform)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution") 

# polynomial regression (remove AAR = 0) ===========================================================

# define X and y:
X = np.array(df[["SMB"]].where(df["AAR"] > 0).dropna()) 
y = np.array(df[["AAR"]].where(df["AAR"] > 0).dropna())

# define polynomial model:
poly = PolynomialFeatures(degree=2, include_bias=False)   

# create new array with X² features:
poly_features = poly.fit_transform(X.reshape(-1, 1))

# add constant term to X for sma.OLS regression and redefine X and y as DataFrame:
X = pd.DataFrame(sma.add_constant(poly_features), columns=["const", "x", "x2"])
y = pd.DataFrame(y)

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef1 = model.params["x"]
coef2 = model.params["x2"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef1, 6))\
    + "\n" + "x²-coefficient: " + str(round(coef2, 6))\
    + "\n" + "Intercept: " + str(round(intercept, 3))\
    + "\n" + "Mean standard error: " + str(np.mean(round(std_err, 3)))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X["x"], y, color="black")
plot_02 = ax1.plot(X["x"], model.predict(X), color="red", label="Trendline", linestyle="--")
plot_03 = ax1.plot(X["x"], upper, color="red", label="Upper", linestyle="dotted")
plot_04 = ax1.plot(X["x"], lower, color="red", label="Lower", linestyle="dotted")
plt.xlabel("Annual SMB in m w.e.")
plt.title("Polynomial regression (AAR > 0)")
plt.axvline(x=0, color="black", linestyle="--")
plt.text(-4.5, 0.6, model_str, fontsize=9)
plt.show()

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["x"], res)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Annual SMB in m w.e.")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (AAR > 0)")
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
plt.ylabel("AAR")
plt.title("Residuals vs. model predicitons (AAR > 0)")
plt.show()

# check normal distribution of residuals:
plt.hist(res)
plt.xlabel("Residual value")
plt.ylabel("Number of instances")
plt.title("Residuals histogram (AAR > 0)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution") 