# -*- coding: utf-8 -*-
"""
Created on: 2024-07-25
@author:    Jasper Heuer
use:        calculate statistics for annual climate station data in Tasiilaq
"""

# import packages ==================================================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy import stats

# import data ======================================================================================

basepath = "C:/Jasper/Master/Thesis/Data/"
os.chdir(basepath)

temp = pd.read_csv("./Other/Climate_stations/Tasiilaq/annual_temp.csv", sep=",")
precip = pd.read_csv("./Other/Climate_stations/Tasiilaq/precipitation.csv", sep=",")

amo = pd.read_csv("./Other/Climate_indices/amo_monthly.csv", sep=";", header=None) 
nao = pd.read_csv("./Other/Climate_indices/nao_monthly.csv", sep=";")

analysis_df = pd.read_csv("./CSV/complete_table_latest.csv")

years = np.arange(1958, 2024, 1)

df = temp.merge(precip, on="DateTime")[0:66]
df["Year"] = years

plt.plot(df["Middel"])

# PROMIE station:     
site='MIT'

timeframe='month'

url = "https://thredds.geus.dk/thredds/fileServer/aws_l3_station_csv/level_3/{}/{}_{}.csv".format(site,site,timeframe)
promice = pd.read_csv(url)
promice["time"] = pd.to_datetime(promice['time'])
promice.index = pd.to_datetime(promice["time"])

# calculate annual mean temperature:
year_mean = []

for i in range(8, len(promice)-9, 12):
    mean_temp = np.mean(promice["t_u"][i:i+12])
    year_mean.append(mean_temp)
    
promice_df = pd.DataFrame([year_mean, np.arange(2010, 2023, 1)]).T
promice_df = promice_df.rename(columns={0:"Promice_T", 1:"Year"})

# calculate annual total precipitation:
year_sum = []

for i in range(8, len(promice)-9, 12):
    total_precip = np.sum(promice["precip_u_cor"][i:i+12])
    year_sum.append(total_precip)
    
promice_df = pd.DataFrame([year_mean, year_sum, np.arange(2010, 2023, 1)]).T
promice_df = promice_df.rename(columns={0:"Promice_T", 1:"Precipitation", 2:"Year"})

# calculate annual mean nao:
nao_mean = []

for i in range(0, len(nao), 12):
    mean_nao = np.mean(nao["INDEX"][i:i+12])
    nao_mean.append(mean_nao)
    
nao_df = pd.DataFrame([nao_mean, np.arange(1950, 2025, 1)]).T
nao_df = nao_df.rename(columns={0:"NAO", 1:"Year"})

# calculate annual mean amo:
amo_mean = []

for i in range(0, len(amo), 12):
    mean_amo = np.mean(amo[1][i:i+12])
    amo_mean.append(mean_amo)
    
amo_df = pd.DataFrame([amo_mean, np.arange(1948, 2025, 1)]).T
amo_df = amo_df.rename(columns={0:"AMO", 1:"Year"})

# merge final dataframe:
df = df.merge(promice_df, on="Year", how="outer")
df = df.merge(nao_df, on="Year", how="outer")
df = df.merge(amo_df, on="Year", how="outer")
df = df.merge(analysis_df, on="Year", how="outer")


# clean up dataframe:
df_long = df
df = df.drop(["Højeste 10 min. middel", "Højeste vindstød", "DateTime"], axis=1)[37:76]
df = df.reset_index()

# correlation analysis =============================================================================

for i in range(0, np.size(df, axis=1)):
    res = stats.shapiro(df.iloc[:, i], nan_policy="omit")
    print("p-value for column " + str(df.keys()[i]) + " = " + str(res[1]))
    if res[1] > 0.05:
        print("Data normally distributed")
    else:
        print("Data not normally distributed")

corr_df = df.drop([9, 23, 25])

# SMB correlation:
X = df_long["Nedbør"][10:76]
y = df_long["Middel"][10:76]
corr_coef, p_value = stats.pearsonr(X, y)
corr_str = "Correlation coefficient: " + str(round(corr_coef, 3))\
            + "\n" + "p-Value: " + str(round(p_value, 4))
print(corr_str)

# SMB correlation:
X = df["WGMS_ELA"].dropna()
y = df["NAO"][11:39]#.drop([9, 23, 25])
corr_coef, p_value = stats.kendalltau(X, y)
corr_str = "Correlation coefficient: " + str(round(corr_coef, 3))\
            + "\n" + "p-Value: " + str(round(p_value, 4))
print(corr_str)
            
plt.scatter(X, y)
# plt.plot(np.arange(-3500, 1001, 1), np.arange(-3500, 1001, 1), linestyle="--", color="red")
# plt.xlim(-3500, 600)
# plt.ylim(-3500, 600)
# plt.gca().set_aspect("equal")
plt.xlabel("SMB in mm w.eq.")
plt.ylabel("AMO")
# plt.text(-3800, -4700, corr_str)
plt.title("SMB correlation")
# plt.savefig("./Data/Plots/smb_correlation.png", bbox_inches="tight", dpi=300)
plt.show()

# calculate statistics =============================================================================

# plot temperature:
fig, ax1 = plt.subplots()

plot_01 = ax1.bar(df["Year"], df["Nedbør"], color="tab:blue", label="Precipitation (Tasiilaq)")
ax1.set_ylabel("Total precipitation in mm w.eq.")

ax1.set_ylim(0, 3000)
ax1.yaxis.set_major_locator(ticker.FixedLocator([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]))
ax1.yaxis.set_label_coords(-0.12, 0.35)

ax2 = ax1.twinx()

plot_02 = ax2.plot(df["Year"], df["Middel"], color="red", label="Temperature (Tasiilaq)")
plot_03 = ax2.plot(df["Year"], df["Promice_T"], color="purple", label="Temperature (Glacier)")
ax2.set_ylabel("Air temperature (2m) in °C")

ax2.set_ylim(-10, 3)
ax2.yaxis.set_major_locator(ticker.FixedLocator([-4, -2, 0, 2]))
ax2.yaxis.set_label_coords(1.09, 0.7)

plt.title("Precipitation and temperature (1985-2023)")

lns = plot_02 + plot_03
fig.legend(bbox_to_anchor = (0.875, 0.05), ncols=2)
plt.savefig("./Plots/climate_overview.png", bbox_inches="tight", dpi=300)
plt.show()

# plot precipitation:
fig, ax = plt.subplots()

plot_01 = ax.bar(df["Year"], df["Nedbør"], color="tab:blue", label="Station Tasiilaq", width=0.7)

plt.title("Total annual precipitation (Tasiilaq)")
plt.ylabel("Total annual precipitation in mm")
plt.show()

np.mean(df["Middel"][8:27])
np.mean(df["Nedbør"][14:22])

# regression analysis ==============================================================================

# define X and y:
X = sma.add_constant(df[["Year"]][25:38]) # double brackets needed for LinearRegression
y = df[["Promice_T"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary())

# long-term plots ==================================================================================

mean_list = []

for i in range(0, len(df_long[10:76])-10):
    mean_10y = np.nanmean(df_long["Middel"][i+10:i+20])
    mean_list.append(mean_10y)
    
precip_list = []

for i in range(0, len(df_long[10:76])-10):
    mean_10y = np.nanmean(df_long["Nedbør"][i+10:i+20])
    precip_list.append(mean_10y)
    
amo_list = []

for i in range(0, len(df_long[10:76])-10):
    mean_10y = np.nanmean(df_long["AMO"][i+10:i+20])
    amo_list.append(mean_10y)
    
nao_list = []

for i in range(0, len(df_long[10:76])-10):
    mean_10y = np.nanmean(df_long["NAO"][i+10:i+20])
    nao_list.append(mean_10y)
        
# plot anomalies:
fig, ax1 = plt.subplots()

plot_01 = ax1.plot(df_long["Year"][10:76], df_long["Middel"][10:76] - np.nanmean(df_long["Middel"]),
                   label="Raw temperature", color="coral")
plot_02 = ax1.plot(df_long["Year"][10:66]+5, mean_list - np.nanmean(df_long["Middel"]), 
                   label="10-year running mean temperature", color="red")
ax1.set_ylabel("Mean air temperature anomaly in °C", color="coral")
ax1.tick_params(axis="y", colors="coral")

ax2 = ax1.twinx()

plot_03 = ax2.plot(df_long["Year"][10:76], df_long["Nedbør"][10:76]- np.nanmean(df_long["Nedbør"]),
                   label="Raw precipitation", color="tab:blue")   
plot_04 = ax2.plot(df_long["Year"][10:66]+5, precip_list - np.nanmean(df_long["Nedbør"]),
                   label="10-year running mean precipitation", color="blue")
ax2.set_ylabel("Total precipitation in mm w.eq.", color="tab:blue")
ax2.tick_params(axis="y", colors="tab:blue")

plt.title("Long-term climate anomalies in Tasiilaq (1958-2023)")
plt.savefig("./Plots/climate_anomalies.png", bbox_inches="tight", dpi=300)
plt.show()

# plot indices:
fig, ax1 = plt.subplots()

plot_01 = ax1.plot(df_long["Year"][10:76], df_long["NAO"][10:76],
                   label="NAO", color="coral")
plot_02 = ax1.plot(df_long["Year"][10:66]+5, nao_list, 
                   label="10-year running mean NAO", color="red")
ax1.set_ylabel("NAO index", color="coral")
ax1.tick_params(axis="y", colors="coral")

ax2 = ax1.twinx()

plot_03 = ax2.plot(df_long["Year"][10:76], df_long["AMO"][10:76],
                   label="AMO", color="tab:blue")   
plot_04 = ax2.plot(df_long["Year"][10:66]+5, amo_list,
                   label="10-year running mean AMO", color="blue")
ax2.set_ylabel("AMO index", color="tab:blue")
ax2.tick_params(axis="y", colors="tab:blue")

plt.title("AMO and NAO indices (1958-2023)")
plt.savefig("./Plots/AMO_NAO_indices.png", bbox_inches="tight", dpi=300)
plt.show()    

# precipitation comparison:
fig, ax1 = plt.subplots()
p1 = ax1.plot(df["Year"], df["Middel"])

ax2 = ax1.twinx()
p2 = ax2.plot(df["Year"], df["SMB"], color="orange")
plt.show()
