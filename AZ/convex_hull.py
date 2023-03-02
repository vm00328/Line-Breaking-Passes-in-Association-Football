# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:17:05 2023

@author: Vlado
"""
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen
from matplotlib import animation
import numpy as np
import pandas as pd


df_AZ_VIT_cleaned = pd.read_csv("C:/Users/Vlado/Desktop/AZ/AZ/data/AZ_VIT_tracking_cleaned.csv")
df_tracking_home = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==0]
df_tracking_away = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==1]
df_tracking_ball = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team'].isna()]
df_tracking_ball.drop(['team', 'jersey_no'], axis = 1, inplace = True) #if inplace is False, the ball doesn't appear

parser = Sbopen()
df, related, freeze, tactics = parser.event(7478)

df = df[(df_tracking_away.jersey_no == '12').copy()

pitch = Pitch(pitch_type='tracab')
fig, ax = pitch.draw(figsize=(8, 6))
hull = pitch.convexhull(df.x, df.y)
poly = pitch.polygon(hull, ax=ax, edgecolor='cornflowerblue', facecolor='cornflowerblue', alpha=0.3)
scatter = pitch.scatter(df.x, df.y, ax=ax, edgecolor='black', facecolor='cornflowerblue')
plt.show()  # if you are not using a Jupyter notebook this is necessary to show the plot