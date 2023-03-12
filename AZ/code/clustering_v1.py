# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:47:49 2023

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
df_tracking_away = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==1]

df_tracking_away = df_tracking_away[(df_tracking_away['frame'] >= 1690842) & (df_tracking_away['frame'] < 1690942)].copy()
df_tracking_home = df_tracking_home[(df_tracking_home['frame'] >= 1690842) & (df_tracking_home['frame'] < 1690942)].copy()
df_tracking_ball = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team'].isna() & (df_AZ_VIT_cleaned['frame'] >=1690842) & (df_AZ_VIT_cleaned['frame'] < 1690942)].copy()

df_home_positions = df_tracking_home[['frame', 'jersey_no', 'x']]
df_away_positions = df_tracking_away[['frame', 'jersey_no', 'x']]
df_positions = pd.concat([df_home_positions, df_away_positions])

df_positions = df_positions.drop_duplicates(subset=['frame', 'jersey_no'])

# Pivot the dataframe to get one row per frame and one column per player
df_positions = df_positions.pivot(index='frame', columns='jersey_no', values='x')

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_positions)
labels = kmeans.labels_

pitch = Pitch(pitch_type='tracab', goal_type='line', pitch_length=105, pitch_width=68,
              axis=True, label=True, corner_arcs=True, line_color='white', pitch_color = "grass", stripe =True)
fig, ax = pitch.draw(figsize=(16, 10.4))

marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}                 
ball, = ax.plot([], [], ms=8, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms=11, markerfacecolor='#0000FF', **marker_kwargs) #blue
home, = ax.plot([], [], ms=11, markerfacecolor='#ff0000', **marker_kwargs) #red

home_positions = df_tracking_home[df_tracking_home['frame'] == df_tracking_home['frame'].unique()[0]]['x'].values.reshape(-1, 1)

# Apply KMeans clustering to the home team positions
kmeans_home = KMeans(n_clusters=3, random_state=42).fit(home_positions)

# Get the cluster assignments for each player in the home team
home_labels = kmeans_home.predict(home_positions)

# Repeat the same steps for the away team
away_positions = df_tracking_away[df_tracking_away['frame'] == df_tracking_away['frame'].unique()[0]]['x'].values.reshape(-1, 1)
kmeans_away = KMeans(n_clusters=3, random_state=42).fit(away_positions)
away_labels = kmeans_away.predict(away_positions)
player_labels = {}
def animate(i):
    ball.set_data(df_tracking_ball.iloc[i, 5], df_tracking_ball.iloc[i, 6])
    frame = df_tracking_ball.iloc[i, 2] # get the frame id for the ith frame
    away.set_data(df_tracking_away.loc[df_tracking_away.frame == frame, 'x'],
                  df_tracking_away.loc[df_tracking_away.frame == frame, 'y'])
    home.set_data(df_tracking_home.loc[df_tracking_home.frame == frame, 'x'],
                  df_tracking_home.loc[df_tracking_home.frame == frame, 'y'])
    
    # Clustering players for each team
    clusters_home = KMeans(n_clusters=3).fit(df_tracking_home[df_tracking_home['frame']==frame][['x']])
    clusters_away = KMeans(n_clusters=3).fit(df_tracking_away[df_tracking_away['frame']==frame][['x']])
    
    # Plotting connections between players of the same team who belong to the same cluster
    for cluster in range(3):
        # Home team
        cluster_home = df_tracking_home[(df_tracking_home['frame']==frame) & (clusters_home.labels_==cluster)]
        if len(cluster_home) > 1:
            for i, row1 in cluster_home.iterrows():
                for j, row2 in cluster_home.iterrows():
                    if i < j:
                        x1, y1 = row1['x'], row1['y']
                        x2, y2 = row2['x'], row2['y']
                        ax.plot([x1, x2], [y1, y2], color='red', linewidth=1.5, linestyle='-')
    # Away team
        cluster_away = df_tracking_away[(df_tracking_away['frame']==frame) & (clusters_away.labels_==cluster)]
        if len(cluster_away) > 1:
            for i, row1 in cluster_away.iterrows():
                for j, row2 in cluster_away.iterrows():
                    if i < j:
                        x1, y1 = row1['x'], row1['y']
                        x2, y2 = row2['x'], row2['y']
                        ax.plot([x1, x2], [y1, y2], color='blue', linewidth=1.5, linestyle='-')
    
    for i, row in df_tracking_home[df_tracking_home['frame']==frame].iterrows():
        x, y = row['x'], row['y']
        vx, vy = row['vx'], row['vy']
        p = pitch.arrows(x, y, vx, vy, alpha=0.8, cmap='coolwarm_r',
                          headaxislength=3, headlength=3, headwidth=4, width=2, ax=ax, zorder=10)
    
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
        x, y = row['x'], row['y']
        vx, vy = row['vx'], row['vy']
        p = pitch.arrows(x, y, vx, vy, alpha=0.8, cmap='coolwarm_r',
                          headaxislength=3, headlength=3, headwidth=4, width=2, ax=ax, zorder=10)
    
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
anim = animation.FuncAnimation(fig, animate, frames=len(df_tracking_ball), interval=50, blit=True)
f = r"C://Users/Vlado/Documents/GitHub/AZ/AZ/animations/proba13.gif" # try with .mov, .avi
writervideo = animation.PillowWriter(fps = 25)
anim.save(f, writer = writervideo)