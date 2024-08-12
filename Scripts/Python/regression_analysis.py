# -*- coding: utf-8 -*-
"""
created on: 2024-03-27
@author:    Jasper Heuer
use:        create AAR and ELA plot
"""

# import packages ==================================================================================

import os
import math
import numpy as np
import pandas as pd
import datetime as datetime
import statsmodels.api as sma
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

# read data:
df = pd.read_csv("./CSV/complete_table_latest.csv", sep=",") # latest version by default
df = df.drop("Unnamed: 0", axis=1)

np.nanmean(df["ELA"][11:39])

# linear regression AAR ============================================================================

# define X and y:
X = sma.add_constant(df[["Year"]].drop([9, 23, 25])) # double brackets needed for LinearRegression
y = df[["AAR"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef = model.params["Year"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef, 6))\
    + "\n" "Intercept: " + str(round(intercept, 3))
    
# define X and y (WGMS):
wgms_X = sma.add_constant(df[["Year"]][11:39]) # double brackets needed for LinearRegression
wgms_y = df[["WGMS_AAR"]].dropna()

# create linear regression models:
wgms_model = sma.OLS(wgms_y, wgms_X).fit()
print(wgms_model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
wgms_std_err, wgms_upper, wgms_lower =  wls_prediction_std(wgms_model, alpha=0.05) # tool is valid for OLS
wgms_std_err = np.mean(wgms_std_err[:])

# save results:
wgms_coef = model.params["Year"]
wgms_intercept = model.params["const"]
wgms_r2 = model.rsquared

# plot original AAR with trend line:
fig, ax1 = plt.subplots()

ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["AAR"], "o-", color="blue", label="AAR")
plot_02 = ax1.fill_between(df["Year"], df["AAR (0.5)"], df["AAR (0.6)"], color="cornflowerblue",
                           label="Uncertainty envelope")
plot_03 = ax1.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
plot_04 = ax1.plot(X["Year"], upper, color="black", linestyle="dotted")
plot_05 = ax1.plot(X["Year"], lower, color="black", label="95% confidence interval", 
                   linestyle="dotted")
plt.title("Accumulation area ratio (AAR) over time")
# plt.text(1985, -0.2, model_str)
plt.legend(loc="lower left", mode="expand", ncols=2, bbox_to_anchor=(0, -0.3, 1, 1))
plt.savefig("./Plots/aar_trend.png", bbox_inches="tight", dpi=300)
plt.show()

# plot original and WGMS data:
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["AAR"], "o-", color="blue", label="Original data")
plot_02 = ax1.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
plot_03 = ax1.plot(X["Year"], upper, color="black", linestyle="dotted")
plot_04 = ax1.plot(X["Year"], lower, color="black", label="95% confidence interval", 
                   linestyle="dotted")
plot_05 = ax1.plot(wgms_X["Year"], df["WGMS_AAR"][11:39], "o-", color="skyblue", alpha=0.7, 
                   label="WGMS data")
plot_06 = ax1.plot(wgms_X["Year"], wgms_model.predict(wgms_X), color="grey", 
                   label="Trendline (WGMS)", linestyle="--")
plot_07 = ax1.plot(wgms_X["Year"], wgms_upper, color="grey", linestyle="dotted")
plot_08 = ax1.plot(wgms_X["Year"], wgms_lower, color="grey", label="95% confidence interval (WGMS)",
                   linestyle="dotted")
plt.title("Comparison of AAR time series")
# plt.text(1985, -0.2, model_str)
plt.legend(loc=0, bbox_to_anchor = (1, -0.10), ncols=2)
# plt.legend(loc="lower left", mode="expand", ncols=2, bbox_to_anchor=(0, -0.3, 1, 1))
plt.savefig("./Plots/aar_comparison.png", bbox_inches="tight", dpi=300)
plt.show()

np.nanmean(np.abs(df["AAR"] - df["WGMS_AAR"]))
np.nanmean(df["AAR"][10:39])

# investigate assumptions ==========================================================================

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

# linear regression ELA ============================================================================

# define X and y:
X = sma.add_constant(df[["Year"]].drop([9, 23, 25])) # double brackets needed for LinearRegression
y = df[["ELA"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef = model.params["Year"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef, 6))\
    + "\n" "Intercept: " + str(round(intercept, 3))
    
# define X and y (WGMS):
wgms_X = sma.add_constant(df[["Year"]][11:39]) # double brackets needed for LinearRegression
wgms_y = df[["WGMS_ELA"]].dropna()

# create linear regression models:
wgms_model = sma.OLS(wgms_y, wgms_X).fit()
print(wgms_model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
wgms_std_err, wgms_upper, wgms_lower =  wls_prediction_std(wgms_model, alpha=0.05) # tool is valid for OLS
wgms_std_err = np.mean(wgms_std_err[:])

# save results:
wgms_coef = model.params["Year"]
wgms_intercept = model.params["const"]
wgms_r2 = model.rsquared

# plot original ELA with trend line:
fig, ax1 = plt.subplots()

ax1.set_ylabel("ELA in meters", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["ELA"], "o-", color="red", label="ELA")
plot_02 = ax1.fill_between(df["Year"], df["ELA (0.5)"], df["ELA (0.6)"], color="pink",
                           label="Uncertainty envelope")
plot_03 = ax1.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
plot_04 = ax1.plot(X["Year"], upper, color="black", linestyle="dotted")
plot_05 = ax1.plot(X["Year"], lower, color="black", label="95% confidence interval", 
                   linestyle="dotted")
plt.title("Equilibrium-line altitude (ELA) over time")
# plt.text(1985, 810, model_str)
plt.legend(bbox_to_anchor=(0, -0.3, 1, 1), loc="lower left", mode="expand", ncols=2)
plt.savefig("./Plots/ela_trend.png", bbox_inches="tight", dpi=300)
plt.show()

# format first y-axis:
fig, ax1 = plt.subplots()

ax1.set_ylabel("ELA in m", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["ELA"], "o-", color="red", label="Original data")
plot_02 = ax1.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
plot_03 = ax1.plot(X["Year"], upper, color="black", linestyle="dotted")
plot_04 = ax1.plot(X["Year"], lower, color="black", label="95% confidence interval",
                   linestyle="dotted")
plot_05 = ax1.plot(wgms_X["Year"], df["WGMS_ELA"][11:39], "o-", color="lightcoral", alpha=0.7,
                   label="WGMS data")
plot_06 = ax1.plot(wgms_X["Year"], wgms_model.predict(wgms_X), color="grey", 
                   label="Trendline (WGMS)", linestyle="--")
plot_07 = ax1.plot(wgms_X["Year"], wgms_upper, color="grey", linestyle="dotted")
plot_08 = ax1.plot(wgms_X["Year"], wgms_lower, color="grey", label="95% confidence interval (WGMS)",
                   linestyle="dotted")
plt.title("Comparison of ELA time series")
# plt.text(1985, 1100, model_str)
plt.legend(loc=0, bbox_to_anchor = (1, -0.10), ncols=2)
plt.savefig("./Plots/ela_comparison.png", bbox_inches="tight", dpi=300)
plt.show()

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["Year"], res)
plt.ylim(-170, 170)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (ELA, linear regression)")
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
plt.xlabel("Predicted ELA in m")
plt.ylabel("Residual ELA in m")
plt.title("Residuals vs. model predicitons (ELA, linear regression)")
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
plt.title("Residuals histogram (ELA, linear regression)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution")
    
# check RMSE:
print("RMSE: " + str(np.sqrt(np.mean(sum(res**2)))))
    
# linear regression SMB ============================================================================

# define X and y:
X = sma.add_constant(df[["Year"]])
y = df[["SMB"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef = model.params["Year"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef, 6))\
    + "\n" "Intercept: " + str(round(intercept, 3))
    
# define X and y (WGMS):
wgms_X = sma.add_constant(df[["Year"]][11:39]) # double brackets needed for LinearRegression
wgms_y = df[["WGMS_SMB"]].dropna()

# create linear regression models:
wgms_model = sma.OLS(wgms_y, wgms_X).fit()
print(wgms_model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
wgms_std_err, wgms_upper, wgms_lower =  wls_prediction_std(wgms_model, alpha=0.05) # tool is valid for OLS
wgms_std_err = np.mean(wgms_std_err[:])

# save results:
wgms_coef = model.params["Year"]
wgms_intercept = model.params["const"]
wgms_r2 = model.rsquared

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("SMB in mm w.eq.", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(X["Year"], y, "o-", color="green", label="SMB")
plot_02 = ax1.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
plot_03 = ax1.plot(X["Year"], upper, color="black", linestyle="dotted")
plot_04 = ax1.plot(X["Year"], lower, color="black", label="95% confidence interval",
                   linestyle="dotted")
plt.title("Surface mass balance (SMB) over time")
# plt.text(1985, -3700, model_str)
plt.axhline(y=0, linestyle="--", color="grey")
plt.legend(bbox_to_anchor=(0, -0.225, 1, 1), loc="lower left", mode="expand", ncols=3)
plt.savefig("./Plots/smb_trend.png", bbox_inches="tight", dpi=300)
plt.show()

# format first y-axis:
fig, ax1 = plt.subplots()
    
ax1.set_ylabel("SMB in mm w.eq.", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(X["Year"], y, "o-", color="green", label="Original data")
plot_02 = ax1.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
plot_03 = ax1.plot(X["Year"], upper, color="black", linestyle="dotted")
plot_04 = ax1.plot(X["Year"], lower, color="black", label="95% confidence interval",
                   linestyle="dotted")
plot_05 = ax1.plot(wgms_X["Year"], df["WGMS_SMB"][11:39], "o-", color="lightgreen", alpha=0.7,
                   label="WGMS data")
plot_06 = ax1.plot(wgms_X["Year"], wgms_model.predict(wgms_X), color="grey", 
                   label="Trendline (WGMS)", linestyle="--")
plot_07 = ax1.plot(wgms_X["Year"], wgms_upper, color="grey", linestyle="dotted")
plot_08 = ax1.plot(wgms_X["Year"], wgms_lower, color="grey", label="95% confidence interval (WGMS)",
                   linestyle="dotted")
plt.title("Comparison of SMB time series")
# plt.text(1985, -3700, model_str)
plt.axhline(y=0, linestyle="--", color="grey")
plt.legend(loc=0, bbox_to_anchor = (1, -0.15), ncols=2)
plt.savefig("./Plots/smb_comparison.png", bbox_inches="tight", dpi=300)
plt.show()

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["Year"], res)
plt.ylim(-3000, 2000)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (SMB, linear regression)")
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
plt.xlabel("Predicted SMB in mm w.eq.")
plt.ylabel("Residual SMB in mm w.eq.")
plt.title("Residuals vs. model predicitons (SMB, linear regression)")
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
plt.title("Residuals histogram (SMB, linear regression)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution")
    
# check RMSE:
print("RMSE: " + str(np.sqrt(np.mean(sum(res**2)))))
    
# linear regression hydrological year length =======================================================

# define X and y:
X = sma.add_constant(df[["Year"]]) # double brackets needed for LinearRegression
y = df[["Hydro_year"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef = model.params["Year"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef, 6))\
    + "\n" "Intercept: " + str(round(intercept, 3))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("Year length in days", color="black")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.plot(df["Year"], df["Hydro_year"], "o-", color="purple", label="Year length")
plot_02 = ax1.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
plot_03 = ax1.plot(X["Year"], upper, color="black", linestyle="dotted")
plot_04 = ax1.plot(X["Year"], lower, color="black", label="95% confidence interval",
                   linestyle="dotted")
plt.title("Length of hydrological year over time")
# plt.text(1985, 310, model_str)
plt.legend(bbox_to_anchor=(0, -0.225, 1, 1), loc="lower left", mode="expand", ncols=3)
plt.savefig("./Plots/hydro_year.png", bbox_inches="tight", dpi=300)
plt.show()

# linear regression of season end ==================================================================

jul_list = []

for i in range(0, len(df)):   
    if math.isnan(df["Month"][i]):
        pass
    else:
        year = df["Year"][i].astype(int)
        month = df["Month"][i].astype(int)
        day = df["Day"][i].astype(int)
    
        dt = datetime.datetime(year, month, day)
        jul = int(dt.strftime("%j"))
        jul_list.append(jul)
    
plt.plot(df["Year"].drop([9, 23, 25]), jul_list, "-o")

# define X and y:
X = sma.add_constant(df[["Year"]].drop([9, 23, 25])) # double brackets needed for LinearRegression
y = jul_list

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["Year"], res)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (SMB, linear regression)")
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
plt.xlabel("Predicted SMB in mm w.eq.")
plt.ylabel("Residual SMB in mm w.eq.")
plt.title("Residuals vs. model predicitons (ELA, linear regression)")
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
plt.title("Residuals histogram (SMB, linear regression)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution")

# check RMSE:
print("RMSE: " + str(np.sqrt(np.mean(sum(res**2)))))
    
# linear regression AAR-SMB ========================================================================

# define X and y:
X = sma.add_constant(df[["SMB"]]).drop([9, 23, 25]) # double brackets needed for LinearRegression
y = df[["AAR"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef = model.params["SMB"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef, 6))\
    + "\n" "Intercept: " + str(round(intercept, 3))

# plot original SMB with trend line:
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("AAR", color="black")
ax1.set_xlabel("SMB in mm w.eq.")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X["SMB"], y, color="black", label="Original data")
plot_02 = ax1.plot([-4000, 500], model.predict([[1, -4000], [1, 500]]), color="red", label="Trendline", linestyle="--")
plot_03 = ax1.plot(df["SMB"][[7, 31]], upper[[7, 31]], color="red", linestyle="dotted")
plot_04 = ax1.plot(df["SMB"][[7, 31]], lower[[7, 31]], color="red", label="95% confidence interval",
                   linestyle="dotted")
plt.title("AAR-SMB regression")
plt.xlim(-3400, 450)
# plt.text(-2000, -0.4, model_str)
plt.axvline(x=0, color="black", linestyle="--")
plt.legend(bbox_to_anchor=(-0.1, -0.3, 1, 1), loc="lower left", ncols=3)
plt.savefig("./Plots/aar_smb_regression.png", bbox_inches="tight", dpi=300)
plt.show()

AAR0 = model.predict([1, 0])
AAR_mean = np.mean(df["AAR"])
AAR_reg_mean = model.predict([1, np.mean(df["SMB"])])

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["SMB"], res)
plt.ylim(-0.5, 0.6)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (SMB-AAR, linear regression)")
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
plt.title("Residuals vs. model predicitons (AAR-SMB, linear regression)")
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
plt.title("Residuals histogram (AAR-SMB, linear regression)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution")
    
# check RMSE:
print("RMSE: " + str(np.sqrt(np.mean(sum(res**2)))))
    
# linear regression AAR-SMB (WGMS data) ============================================================

# define X and y:
X = sma.add_constant(df[["WGMS_SMB"]][df["WGMS_AAR"] > 0]).dropna() # double brackets needed for LinearRegression
y = df[["WGMS_AAR"]][df["WGMS_AAR"] > 0].dropna()

# exclude AAR = 0:
# X = sma.add_constant(df["WGMS_SMB"][df["WGMS_AAR"] > 0]).dropna()
# y = df["WGMS_AAR"][df["WGMS_AAR"] > 0].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# save results:
coef = model.params["WGMS_SMB"]
intercept = model.params["const"]
r2 = model.rsquared

model_str = "R²-value: " + str(round(r2, 3)) + "\n" + "x-coefficient: " + str(round(coef, 6))\
    + "\n" "Intercept: " + str(round(intercept, 3))

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("WGMS ARR", color="black")
ax1.set_xlabel("WGMS SMB in mm w.eq.")
ax1.tick_params(axis="y", colors="black")
plot_01 = ax1.scatter(X["WGMS_SMB"], y, color="black", label="WGMS data")
plot_02 = ax1.plot([-1300, 500], model.predict([[1, -1300], [1, 500]]), color="red", label="Trendline", linestyle="--")
plot_03 = ax1.plot(df["WGMS_SMB"][[29, 18]], upper[[29, 18]], color="red", linestyle="dotted")
plot_04 = ax1.plot(df["WGMS_SMB"][[29, 18]], lower[[29, 18]], color="red", label="95% confidence interval",
                   linestyle="dotted")
plt.title("AAR-SMB regression (WGMS data)")
# plt.text(-2500, 0.35, model_str) # full data set
# plt.text(-1250, -0.4, model_str) # AAR > 0 
plt.axvline(x=0, color="black", linestyle="--")
plt.xlim(-1260, 420)
plt.legend(bbox_to_anchor=(-0.1, -0.3, 1, 1), loc="lower left", ncols=3)
plt.savefig("./Plots/aar_smb_wgms_regression.png", bbox_inches="tight", dpi=300)
plt.show()

# investigate assumptions ==========================================================================

# calculate residuals:
res = model.resid

# independece of residuals - should mostly fall withing 95% confidence interval
plt.scatter(X["WGMS_SMB"], res)
plt.ylim(-0.3, 0.3)
plt.axhline(y=np.std(res)*1.96, color="red", linestyle="--", label="95% confidence interval")
plt.axhline(y=np.std(res)*-1.96, color="red", linestyle="--")
plt.axhline(y=0, color="black", linestyle="--")
plt.ylabel("Residual value")
plt.title("Distribution of residuals (SMB-AAR (WGMS), linear regression)")
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
plt.title("Residuals vs. model predicitons (AAR-SMB (WGMS), linear regression)")
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
plt.title("Residuals histogram (AAR-SMB (WGMS), linear regression)")
plt.show()

# check normal distribution of residuals:
shapiro_stat = stats.shapiro(res)[1]
if shapiro_stat > 0.05:
    print("Residuals follow a normal distribution")
else:
    print("Residuals don't follow a normal distribution")
    
# check RMSE:
print("RMSE: " + str(np.sqrt(np.mean(sum(res**2)))))

# linear regression AAR-SMB (long vs.short) ===========================================================

# define X and y for short period:
X = sma.add_constant(df[["SMB"]][11:39]).drop([23, 25]) # double brackets needed for LinearRegression
y = df[["AAR"]][11:39].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input
print(model.params) 

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# define X and y for whole period:
X_w = sma.add_constant(df[["SMB"]]).drop([9, 23, 25]) # double brackets needed for LinearRegression
y_w = df[["AAR"]].dropna()

# create linear regression models:
model_w = sma.OLS(y_w, X_w).fit()
print(model_w.summary()) # large condition number only due to scale of X input
print(model_w.params)

# compute confidence intervals:
std_err_w, upper_w, lower_w =  wls_prediction_std(model_w, alpha=0.05) # tool is valid for OLS
std_err_w = np.mean(std_err_w[:])

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# define X and y for WGMS:
X_wgms = sma.add_constant(df[["WGMS_SMB"]][df["WGMS_AAR"] > 0]).dropna() # double brackets needed for LinearRegression
y_wgms = df[["WGMS_AAR"]][df["WGMS_AAR"] > 0].dropna()

# exclude AAR = 0:
# X = sma.add_constant(df["WGMS_SMB"][df["WGMS_AAR"] > 0]).dropna()
# y = df["WGMS_AAR"][df["WGMS_AAR"] > 0].dropna()

# create linear regression model for WGMS:
model_wgms = sma.OLS(y_wgms, X_wgms).fit()
print(model_wgms.summary()) # large condition number only due to scale of X input
print(model_wgms.params)
# compute confidence intervals for WGMS:
std_err_wgms, upper_wgms, lower_wgms =  wls_prediction_std(model_wgms, alpha=0.05) # tool is valid for OLS
std_err_wgms = np.mean(std_err_wgms[:])

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("ARR", color="black")
ax1.set_xlabel("SMB in mm w.eq.")
ax1.tick_params(axis="y", colors="black")

plot_05 = ax1.scatter(X_w["SMB"], y_w, color="black", label="1985-1994")
plot_06 = ax1.plot([-4000, 500], model_w.predict([[1, -4000], [1, 500]]), color="red", label="Trendline (1985-2023)", linestyle="--")
plot_07 = ax1.plot(df["SMB"][[7, 31]], upper_w[[7, 31]], color="red", linestyle="dotted")
plot_08 = ax1.plot(df["SMB"][[7, 31]], lower_w[[7, 31]], color="red", label="95% confidence interval (1985-2023)",
                   linestyle="dotted") 

plot_01 = ax1.scatter(X["SMB"], y, color="cornflowerblue", label="1995-2023")
plot_02 = ax1.plot([-4000, 500], model.predict([[1, -4000], [1, 500]]), color="skyblue", label="Trendline (1995-2023)", linestyle="--")
plot_03 = ax1.plot(df["SMB"][[30, 31]], upper[[30, 31]], color="skyblue", linestyle="dotted")
plot_04 = ax1.plot(df["SMB"][[30, 31]], lower[[30, 31]], color="skyblue", label="95% confidence interval (1995-2023)",
                   linestyle="dotted")

plt.title("Comparison of regression periods")
plt.xlim(-3400, 499)
plt.axvline(x=0, color="black", linestyle="--")
plt.legend(bbox_to_anchor=(-0.23, -0.42, 1, 1), loc="lower left", ncols=2)
plt.savefig("./Plots/smb_aar_regression_comparison.png", bbox_inches="tight", dpi=300)
plt.show()

# linear regression AAR-SMB (comparison) ===========================================================

# define X and y:
X = sma.add_constant(df[["SMB"]]).drop([9, 23, 25]) # double brackets needed for LinearRegression
y = df[["AAR"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input

# compute confidence intervals:
std_err, upper, lower =  wls_prediction_std(model, alpha=0.05) # tool is valid for OLS
std_err = np.mean(std_err[:])

# define X and y for WGMS:
X_wgms = sma.add_constant(df[["WGMS_SMB"]][df["WGMS_AAR"] > 0]).dropna() # double brackets needed for LinearRegression
y_wgms = df[["WGMS_AAR"]][df["WGMS_AAR"] > 0].dropna()

# create linear regression model for WGMS:
model_wgms = sma.OLS(y_wgms, X_wgms).fit()
print(model_wgms.summary()) # large condition number only due to scale of X input

# compute confidence intervals for WGMS:
std_err_wgms, upper_wgms, lower_wgms =  wls_prediction_std(model_wgms, alpha=0.05) # tool is valid for OLS
std_err_wgms = np.mean(std_err_wgms[:])

# plot original SMB with trend line
fig, ax1 = plt.subplots()

# format first y-axis:
ax1.set_ylabel("ARR", color="black")
ax1.set_xlabel("SMB in mm w.eq.")
ax1.tick_params(axis="y", colors="black")

plot_01 = ax1.scatter(X["SMB"], y, color="black", label="1985-1994")
plot_02 = ax1.plot([-4000, 500], model.predict([[1, -4000], [1, 500]]), color="red", label="Trendline", linestyle="--")
plot_03 = ax1.plot(df["SMB"][[7, 31]], upper[[7, 31]], color="red", linestyle="dotted")
plot_04 = ax1.plot(df["SMB"][[7, 31]], lower[[7, 31]], color="red", label="95% confidence interval",
                   linestyle="dotted")
plot_05 = ax1.scatter(X_wgms["WGMS_SMB"], y_wgms, color="darkorchid", label="WGMS data")
plot_06 = ax1.plot([-1300, 500], model_wgms.predict([[1, -1300], [1, 500]]), color="orchid", label="Trendline (WGMS)", linestyle="--")
plot_07 = ax1.plot(df["WGMS_SMB"][[29, 18]], upper_wgms[[29, 18]], color="orchid", linestyle="dotted")
plot_08 = ax1.plot(df["WGMS_SMB"][[29, 18]], lower_wgms[[29, 18]], color="orchid", label="95% confidence interval (WGMS)",
                   linestyle="dotted")
plt.title("Comparison of AAR-SMB regressions")
plt.xlim(-3400, 499)
plt.axvline(x=0, color="black", linestyle="--")
plt.legend(bbox_to_anchor=(-0.1, -0.42, 1, 1), loc="lower left", ncols=2)
plt.savefig("./Plots/smb_aar_regression_comparison_wgms.png", bbox_inches="tight", dpi=300)
plt.show()
