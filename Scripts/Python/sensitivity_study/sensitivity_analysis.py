# -*- coding: utf-8 -*-
"""
created on: 2024-07-08
@author:    Jasper Heuer
use:        create AAR and ELA plot
"""

# import packages ==================================================================================

import os
import pandas as pd
import statsmodels.api as sma
import matplotlib.pyplot as plt

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

# read data:
df = pd.read_excel("./Landsat/Sensitivity_study/sensitivity_table.xlsx") # latest version by default
df = df.drop("Unnamed: 0", axis=1)
df["ELA_uncertainty"] = df["ELA (0.6)"] - df["ELA (0.5)"]
df["AAR_uncertainty"] = df["AAR (0.5)"] - df["AAR (0.6)"]

df.to_csv("./CSV/sensitivity_table.csv")


# plot ELA =========================================================================================

# define X and y:
X = sma.add_constant(df[["Year"]].drop([9, 23, 25])) # double brackets needed for LinearRegression
y = df[["ELA"]].dropna()

# create linear regression models:
model = sma.OLS(y, X).fit()
print(model.summary()) # large condition number only due to scale of X input

# plot ELA:
fig, ax = plt.subplots()

plot_01 = ax.plot(df["Year"], df["ELA"], "-o", label="ELA", color="red")
plot_02 = ax.fill_between(df["Year"], df["ELA (0.5)"], df["ELA (0.6)"], color="pink", label="Uncertainty envelope")
plot_03 = ax.plot(X["Year"], model.predict(X), color="black", label="Trendline", linestyle="--")
ax.set_ylabel("ELA in meters")
plt.title("ELA with uncertainty envelope")
plt.ylim(500, 900)
plt.legend()
plt.savefig("./Plots/ELA_sensitivity.png", dpi=300)
plt.show()

# plot AAR =========================================================================================

fig, ax = plt.subplots()

plot_01 = ax.plot(df["Year"], df["AAR"], "-o", label="Threshold = 0.55", color="blue")
plot_02 = ax.fill_between(df["Year"], df["AAR (0.5)"], df["AAR (0.6)"], color="cornflowerblue")
ax.set_ylabel("AAR")
plt.title("AAR with uncertainty envelope")
plt.savefig("./Plots/AAR_sensitivity.png", dpi=300)
plt.show()
