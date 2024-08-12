# -*- coding: utf-8 -*-
"""
created on: 2024-03-27
@author:    Jasper Heuer
use:        create AAR and ELA plot
"""

# import packages ==================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data ======================================================================================

base_path = "C:/Jasper/Master/Thesis/Data/"
os.chdir(base_path)

df = pd.read_csv("./Landsat/Landsat_05/ELA_AAR_table.csv", sep=",")

# create ELA/AAR plot ==============================================================================

fig, ax1 = plt.subplots()

# add first y-axis:
ax1.set_xlabel("Year")
ax1.set_ylabel("ELA in meters")
plot_01 = ax1.plot(df["year"], df["ELA"], color="red", label="ELA")

# add second y-axis:
ax2 = ax1.twinx()
ax2.set_ylabel("AAR")
plot_02 = ax2.plot(df["year"], df["AAR"], color="blue", label="AAR", linestyle="--")

# create legend:
lns = plot_01 + plot_02
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=0)

# finish plot and write to disk:
plt.title("ELA/AAR over time")
plt.show()
fig.savefig("./Landsat/Landsat_05/Plots/ELA_and_AAR_plot.png", dpi=300)


