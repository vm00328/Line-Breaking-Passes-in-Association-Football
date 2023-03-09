# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:59:48 2023

@author: Vlado
"""
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib import animation
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import ffmpeg
import matplotlib
import os
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, MovieWriter
from matplotlib.text import Text

df_AZ_VIT_cleaned = pd.read_csv("C:/Users/Vlado/Desktop/AZ/AZ/data/AZ_VIT_tracking_cleaned.csv")
df_tracking_home = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==0]
df_tracking_home = df_tracking_home[(df_tracking_home['frame'] >= 1690842) & (df_tracking_home['frame'] < 1690992)].copy()
df_tracking_home_10 = df_tracking_home.drop(df_tracking_home[df_tracking_home['jersey_no'] == 16].index)
df_tracking_away = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==1] #MISSING frame stamp but works?
df_tracking_ball = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team'].isna() & (df_AZ_VIT_cleaned['frame'] >=1690842) & (df_AZ_VIT_cleaned['frame'] < 1690992)].copy()

pitch = Pitch(pitch_type='tracab', goal_type='line', pitch_length=105, pitch_width=68,
              axis=True, label=True, corner_arcs=True, line_color='white', pitch_color = "grass", stripe =True)
fig, ax = pitch.draw(figsize=(16, 10.4))

marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}                 
ball, = ax.plot([], [], ms=8, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms=11, markerfacecolor='#0000FF', **marker_kwargs) #blue
home, = ax.plot([], [], ms=11, markerfacecolor='#ff0000', **marker_kwargs) #red
X = df_tracking_home_10[['x']].values
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Add cluster labels to the dataframe
df_tracking_home = df_tracking_home.assign(cluster=0)
df_tracking_home_10['cluster'] = kmeans.labels_
# Perform K-means clustering on x coordinates of the players for each team
#kmeans_home = KMeans(n_clusters=3, random_state=0).fit(df_tracking_home.loc[df_tracking_home.frame == frame, ['x']])
player_labels={}
prev_line_update = 0
player_lines={}
def animate(i):
    ball.set_data(df_tracking_ball.iloc[i, 5], df_tracking_ball.iloc[i, 6])
    frame = df_tracking_ball.iloc[i, 2]
    home_frame = df_tracking_home_10[df_tracking_home_10['frame'] == frame]
    #home.set_data(home_frame['x'], home_frame['y'])
    home_frame_2 = df_tracking_home[df_tracking_home['frame'] == frame] #reason: keeper number is shown
    home.set_data(df_tracking_home.loc[df_tracking_home.frame == frame, 'x'],
                  df_tracking_home.loc[df_tracking_home.frame == frame, 'y'])
    away.set_data(df_tracking_away.loc[df_tracking_away.frame == frame, 'x'],
                  df_tracking_away.loc[df_tracking_away.frame == frame, 'y'])
    
    global prev_line_update
    if i % 5 == 0:
        for c in range(3):
            cluster_frame = home_frame[home_frame['cluster'] == c]
            if len(cluster_frame) > 1:
                x = cluster_frame['x'].values
                y = cluster_frame['y'].values
                # If a line for this cluster already exists, update its coordinates
                if c in player_lines:
                    player_lines[c].set_data(x, y)
                # Otherwise, create a new line for this cluster
                else:
                    line, = ax.plot(x, y, linewidth=2, color='white')
                    player_lines[c] = line
        # Store the current frame as the previous line update
        prev_line_update = i
    
    # update previous line update frame number
    if (frame - prev_line_update) % 25 == 0:
        prev_line_update = frame
    
    # Create Text objects for each player's jersey number
    for i, row in df_tracking_home[df_tracking_home['frame']==frame].iterrows(): 
        x, y = row['x'], row['y']
        jersey_no = int(row['jersey_no'])
        # if the player already has a label, move it to their current position
        if i in player_labels:
            label = player_labels[i]
            label.set_text(str(jersey_no))
            label.xy = (x, y)
        # otherwise, create a new label and add it to the plot
        else:
            label = ax.annotate(str(jersey_no), xy=(x, y),
                                fontsize=11, color='white', ha='center', va='center')
            player_labels[i] = label
            
    for i, row in df_tracking_away[df_tracking_away['frame']==frame].iterrows():
        x, y = row['x'], row['y']
        jersey_no = int(row['jersey_no'])
        if i in player_labels:
          label = player_labels[i]
          label.set_text(str(jersey_no))
          label.xy = (x, y)
        else:
          label = ax.annotate(str(jersey_no), xy=(x, y), fontsize=11, color='white', ha='center', va='center')
          player_labels[i] = label
        
    # Remove labels for players who are not in the current frame
    for i in list(player_labels):
        if i not in home_frame_2.index and i not in df_tracking_away[df_tracking_away['frame']==frame].index:
            label = player_labels.pop(i)
            label.remove()    
    return ball, away, home

anim = animation.FuncAnimation(fig, animate, frames=len(df_tracking_ball), interval=50, blit=True)
f = r"C://Users/Vlado/Documents/GitHub/AZ/AZ/animations/proba34.gif"
writervideo = animation.PillowWriter(fps = 25)
anim.save(f, writer = writervideo)