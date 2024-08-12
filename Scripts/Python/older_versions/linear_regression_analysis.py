# -*- coding: utf-8 -*-
"""
created on: 2024-06-03
@author:    Jasper Heuer
use:        investigate relationship between SMB and AAR using linear regression (OLS)
"""

# import packages ==================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sma
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/"
os.chdir(base_path)

df = pd.read_csv("./Data/CSV/complete_table_latest.csv")
df = df.drop("Unnamed: 0", axis=1)
df = df.sort_values(by="SMB", axis=0, ascending=True)
df["SMB"] = df["SMB"] / 1000 # get m w.e. instead of mm

# Linear regression (raw data) =====================================================================

# define X and y:
X = sma.add_constant(df[["SMB"]].dropna()) # double brackets needed for LinearRegression
y = df[["AAR"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# save results:
coef = model.params["SMB"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "r2: " + str(round(r2, 3)) + "\n" + "coef: " + str(round(coef, 6))\
    + "\n" "intercept: " + str(round(intercept, 3))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X["SMB"], y, color="black")
plot_02 = ax1.plot(X["SMB"], model.predict(X), color="red", label="Trendline", linestyle="--")
plt.xlabel("Annual SMB in mm w.e.")
plt.title("Linear regression (raw data)")
plt.axvline(x=0, color="black", linestyle="--")
plt.text(-4.5, 0.5, model_str)
plt.show()

# Large condition number if SMB is given in mm w.e., but that that is a scaling issue and does
# not need to be addressed

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["SMB"], res)
plt.axhline(y=(2/np.sqrt(len(X))), color="black", linestyle="--")
plt.axhline(y=(-2/np.sqrt(len(X))), color="black", linestyle="--")
plt.title("Distribution of residuals (raw data)")
plt.show()

# check using the Durbin-Watson test:
print(durbin_watson(res)) # if value outside of 1.5 - 2.5 interval -> autocorrelation

# homoscedasticity:
plt.scatter(model.predict(X), res)
plt.title("Residuals vs. model predicitons (raw data)")
plt.show()

# normal distribution:
plt.hist(res)
plt.title("Residuals histogram (raw data)")
plt.show()

stats.shapiro(res) # p-value < 0.05 indicated non-normal distribution

# Linear regression (sqrt-transform) ================================================================

# define X and y:
X = sma.add_constant(df[["SMB"]].dropna()) # double brackets needed for LinearRegression
y = np.sqrt(df[["AAR"]].dropna())

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# save results:
coef = model.params["SMB"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "r2: " + str(round(r2, 3)) + "\n" + "coef: " + str(round(coef, 6))\
    + "\n" "intercept: " + str(round(intercept, 3))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X["SMB"], y, color="black")
plot_02 = ax1.plot(X["SMB"], model.predict(X), color="red", label="Trendline", linestyle="--")
plt.xlabel("Annual SMB in mm w.e.")
plt.title("Linear regression (square root transform)")
# plt.axvline(x=0, color="black", linestyle="--")
plt.text(-4.5, 0.5, model_str)
plt.show()

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["SMB"], res)
plt.axhline(y=(2/np.sqrt(len(X))), color="black", linestyle="--")
plt.axhline(y=(-2/np.sqrt(len(X))), color="black", linestyle="--")
plt.title("Distribution of residuals (square-root transform)")
plt.show()

# check using the Durbin-Watson test:
print(durbin_watson(res)) # if value outside of 1.5 - 2.5 interval -> autocorrelation

# homoscedasticity:
plt.scatter(model.predict(X), res)
plt.title("Residuals vs. model predicitons (square-root transform)")
plt.show()

# normal distribution:
plt.hist(res)
plt.title("Residuals histogram (square-root transform)")
plt.show()

stats.shapiro(res) # p-value < 0.05 indicated non-normal distribution

# Linear regression (remove AAR = 0) ================================================================

# define X and y:
X = sma.add_constant(df[["SMB"]].where(df["AAR"] > 0).dropna()) # double brackets needed for LinearRegression
y = df[["AAR"]].where(df["AAR"] > 0).dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# save results:
coef = model.params["SMB"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "r2: " + str(round(r2, 3)) + "\n" + "coef: " + str(round(coef, 6))\
    + "\n" "intercept: " + str(round(intercept, 3))
    
# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X["SMB"], y, color="black")
plot_02 = ax1.plot(X["SMB"], model.predict(X), color="red", label="Trendline", linestyle="--")
plt.xlabel("Annual SMB in mm w.e.")
plt.title("Linear regression (AAR > 0)")
plt.axvline(x=0, color="black", linestyle="--")
plt.text(-4.5, 0.5, model_str)
plt.show()

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["SMB"], res)
plt.axhline(y=(2/np.sqrt(len(X))), color="black", linestyle="--")
plt.axhline(y=(-2/np.sqrt(len(X))), color="black", linestyle="--")
plt.title("Distribution of residuals (AAR > 0)")
plt.show()

# check using the Durbin-Watson test:
print(durbin_watson(res)) # if value outside of 1.5 - 2.5 interval -> autocorrelation

# homoscedasticity:
plt.scatter(model.predict(X), res)
plt.title("Residuals vs. model predicitons (AAR > 0)")
plt.show()

# normal distribution:
plt.hist(res)
plt.title("Residuals histogram (AAR > 0)")
plt.show()

stats.shapiro(res) # p-value < 0.05 indicated non-normal distribution
