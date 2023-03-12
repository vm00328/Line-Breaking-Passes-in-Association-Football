# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:54:50 2023

@author: Vladislav Manolov
"""
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib import animation
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import imageio
import ffmpeg
import matplotlib
import os
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, MovieWriter
from matplotlib.text import Text
from scipy.spatial.distance import cdist
from matplotlib.patches import FancyArrowPatch, ArrowStyle, FancyArrow

df_AZ_VIT_cleaned = pd.read_csv("C:/Users/Vlado/Desktop/AZ/AZ/data/AZ_VIT_tracking_cleaned.csv")
df_tracking_home = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==0]
df_tracking_away = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==1]

df_tracking_away = df_tracking_away[(df_tracking_away['frame'] >= 1690842) & (df_tracking_away['frame'] < 1690942)].copy()
df_tracking_home = df_tracking_home[(df_tracking_home['frame'] >= 1690842) & (df_tracking_home['frame'] < 1690942)].copy()

df_tracking_ball = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team'].isna() & (df_AZ_VIT_cleaned['frame'] >=1690842) & (df_AZ_VIT_cleaned['frame'] < 1690942)].copy()
#df_tracking_ball.drop(['team', 'jersey_no'], axis = 1, inplace = True) #if inplace is False, the ball doesn't appear

pitch = Pitch(pitch_type='tracab', goal_type='line', pitch_length=105, pitch_width=68,
              axis=True, label=True, corner_arcs=True, line_color='white', pitch_color = "grass", stripe =True)
fig, ax = pitch.draw(figsize=(16, 10.4))

marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}                 
ball, = ax.plot([], [], ms=8, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms=11, markerfacecolor='#0000FF', **marker_kwargs) #blue
home, = ax.plot([], [], ms=11, markerfacecolor='#ff0000', **marker_kwargs) #red
player_labels={}
def animate(i):
    #global frame
    """ Function to animate the data. Each frame it sets the data for the players and the ball."""
    ball.set_data(df_tracking_ball.iloc[i, 5], df_tracking_ball.iloc[i, 6])
    
    frame = df_tracking_ball.iloc[i, 2] # get the frame id for the ith frame
    # set the player data using the frame id .set_data takes takes two one-dimensional arrays represneting x and y values to plot
    away.set_data(df_tracking_away.loc[df_tracking_away.frame == frame, 'x'],
                  df_tracking_away.loc[df_tracking_away.frame == frame, 'y'])
    home.set_data(df_tracking_home.loc[df_tracking_home.frame == frame, 'x'],
                  df_tracking_home.loc[df_tracking_home.frame == frame, 'y'])
    
   # create Text objects for each player's jersey number
    for i, row in df_tracking_home[df_tracking_home['frame']==frame].iterrows():
        x, y = row['x'], row['y']
        vx, vy = row['vx'], row['vy']
        
        p = pitch.arrows(x, y, vx, vy, alpha=0.8, cmap='coolwarm_r',
                          headaxislength=3, headlength=3, headwidth=4, width=2, ax=ax, zorder = 10)
        jersey_no = int(row['jersey_no'])
        # if the player already has a label, move it to their current position
        if i in player_labels:
            label = player_labels[i]
            label.set_text(str(jersey_no))
            label.xy = (x, y)
        # otherwise, create a new label and add it to the plot
        else:
            label = ax.annotate(str(jersey_no), xy=(x, y), fontsize=11, color='white', ha='center', va='center')
            player_labels[i] = label
    
    for i, row in df_tracking_away[df_tracking_away['frame']==frame].iterrows():
        # get the player's current position and velocity
        x, y = row['x'], row['y']
        vx, vy = row['vx'], row['vy']
        jersey_no = int(row['jersey_no'])
        if i in player_labels:
          label = player_labels[i]
          label.set_text(str(jersey_no))
          label.xy = (x, y)
        else:
          label = ax.annotate(str(jersey_no), xy=(x, y), fontsize=11, color='white', ha='center', va='center')
          player_labels[i] = label
   
    # remove labels for players who are not in the current frame
    for i in list(player_labels):
       if i not in df_tracking_home[df_tracking_home['frame']==frame].index and i not in df_tracking_away[df_tracking_away['frame']==frame].index:
           label = player_labels.pop(i)
           label.remove()
    
    return ball, away, home
    
# frames=len(df_tracking_ball) needed only when exporting the animation; interval=50 is an optional argument and sets the interval between frames in miliseconds
anim = animation.FuncAnimation(fig, animate, frames=len(df_tracking_ball), interval=50, blit=True)
path = r"C://Users/Vlado/Documents/GitHub/AZ/AZ/animations"
print(os.path.exists(path))
os.makedirs(path, exist_ok=True)
f = r"C://Users/Vlado/Documents/GitHub/AZ/AZ/animations/proba11.gif" # try with .mov, .avi
writervideo = animation.PillowWriter(fps = 25)
anim.save(f, writer = writervideo)