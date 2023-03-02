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
import os
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, MovieWriter
from matplotlib.text import Text

df_AZ_VIT_cleaned = pd.read_csv("C:/Users/Vlado/Desktop/AZ/AZ/data/AZ_VIT_tracking_cleaned.csv")
df_tracking_home = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==0]
df_tracking_away = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team']==1]

# Getting subset of the data (frame 1690841 is when the ball is first detected as alive)
df_tracking_away = df_tracking_away[(df_tracking_away['frame'] >= 1690842) & (df_tracking_away['frame'] < 1690942)].copy()
df_tracking_home = df_tracking_home[(df_tracking_home['frame'] >= 1690842) & (df_tracking_home['frame'] < 1690942)].copy()

# Creating a dataframe to hold the ball data
df_tracking_ball = df_AZ_VIT_cleaned[df_AZ_VIT_cleaned['team'].isna() & (df_AZ_VIT_cleaned['frame'] >=1690842) & (df_AZ_VIT_cleaned['frame'] < 1690942)].copy()
#df_tracking_ball.drop(['team', 'jersey_no'], axis = 1, inplace = True) #if inplace is False, the ball doesn't appear

# Creating the pitch interface
pitch = Pitch(pitch_type='tracab', goal_type='line', pitch_length=105, pitch_width=68,
              axis=True, label=True, corner_arcs=True, line_color='white', pitch_color = "grass", stripe =True)
fig, ax = pitch.draw(figsize=(16, 10.4))

# Setting up the pitch plot markers we want to animate
marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}                 
ball, = ax.plot([], [], ms=8, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms=11, markerfacecolor='#0000FF', **marker_kwargs) #blue
home, = ax.plot([], [], ms=11, markerfacecolor='#ff0000', **marker_kwargs) #red

player_labels = {}
player_arrows = {}
def animate(i):
    """ Function to animate the data. Each frame it sets the data for the players and the ball."""
    # set the ball data with the x and y positions for the ith frame
    ball.set_data(df_tracking_ball.iloc[i, 5], df_tracking_ball.iloc[i, 6]) # IMPORTANT, DEPENDS ON WHETHER COLUMNS WERE DROPPED
    # get the frame id for the ith frame
    frame = df_tracking_ball.iloc[i, 2]
    # set the player data using the frame id .set_data takes takes two one-dimensional arrays represneting x and y values to plot
    away.set_data(df_tracking_away.loc[df_tracking_away.frame == frame, 'x'],
                  df_tracking_away.loc[df_tracking_away.frame == frame, 'y'])
    home.set_data(df_tracking_home.loc[df_tracking_home.frame == frame, 'x'],
                  df_tracking_home.loc[df_tracking_home.frame == frame, 'y'])
    
    # create Text objects for each player's jersey number
    for i, row in df_tracking_home[df_tracking_home['frame']==frame].iterrows(): #df_tracking_home['frame'].min()
        # get the player's current position
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

        # create Arrow objects for each player's direction HOME TEAM
        vx, vy = row['vx'], row['vy']
        if i in player_arrows:
            player_arrows[i].remove()
        arrow = ax.arrow(x, y, vx, vy, head_width=1, head_length=1, fc='red', ec='red')
        player_arrows[i] = arrow  

    for i, row in df_tracking_away[df_tracking_away['frame']==frame].iterrows(): #df_tracking_away['frame'].min()
        # get the player's current position
        x, y = row['x'], row['y']
        jersey_no = int(row['jersey_no'])
        if i in player_labels:
           label = player_labels[i]
           label.set_text(str(jersey_no))
           label.xy = (x, y)
        else:
           label = ax.annotate(str(jersey_no), xy=(x, y),
                                   fontsize=11, color='white', ha='center', va='center')
           player_labels[i] = label
           
        # create Arrow objects for each player's direction AWAY TEAM
        vx, vy = row['vx'], row['vy']
        if i in player_arrows:
            player_arrows[i].remove()
        arrow = ax.arrow(x, y, vx, vy, head_width=1, head_length=1, fc='blue', ec='blue')
        player_arrows[i] = arrow
        
    # remove labels for players who are not in the current frame
    for i in list(player_labels):
        if i not in df_tracking_home[df_tracking_home['frame']==frame].index and i not in df_tracking_away[df_tracking_away['frame']==frame].index:
            label = player_labels.pop(i)   #old approach: player_labels[i].remove()
            label.remove()                 #old approach: del player_labels[i]
    
    return ball, away, home

# frames=len(df_tracking_ball) needed only when exporting the animation; interval=50 is an optional argument and sets the interval between frames in miliseconds
anim = animation.FuncAnimation(fig, animate, frames=len(df_tracking_ball), interval=50, blit=True)

f = r"C://Users/Vlado/Documents/GitHub/AZ/AZ/animations/velocity_v2.gif" # try with .mov, .avi
writervideo = animation.PillowWriter(fps = 25)
anim.save(f, writer = writervideo)










################################################Clustering#############################################

# Extract positional features from the tracking data
features = df_tracking_home[["x", "y"]]

# Computing centroids of each player at each frame
centroids = df_AZ_VIT_cleaned.groupby(['frame', 'jersey_no'])[['x', 'y']].mean().reset_index()

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in tqdm(range(1, 11)):
    #“init” argument is the method for initializing the centroid.
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(centroids) #scaled_features replaced by centroids
    wcss.append(kmeans.inertia_)
    
# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means clustering with the optimal number of clusters based on the Elbow method
clusters_number = 4
kmeans = KMeans(n_clusters = clusters_number, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(features)
print(cluster_labels)

"""
# Create a list of scatter plots, one for each frame
scatter_plots = []
for frame in range(1000):
    # Get the centroids for the current frame
    centroids_frame = centroids[centroids['frame'] == frame]
    
    # Create a scatter plot for the current frame
    fig, ax = plt.subplots()
    scatter = ax.scatter(centroids_frame['x'], centroids_frame['y'], cmap='Set1')
    ax.set_xlim(0, 120)  # adjust limits as needed
    ax.set_ylim(0, 80)
    ax.set_title(f'Frame {frame}')
    scatter_plots.append([scatter])
"""
    
df_event_data = pd.read_csv('C:/Users/Vlado/Desktop/AZ/AZ/data/event_data_azvit.csv', sep = ';', header = None)
df_head = df_AZ_VIT_cleaned.head(100)

def sort_key(name):
    # Extract the name and number from the string
    name_parts = name.split('_')
    name = name_parts[0]
    number = int(name_parts[1].replace('.png', ''))
    return name, number

def create_animation(folder_path, output_path):
    files = os.listdir(folder_path)
    images = []
    files = [name for name in files if not name.endswith('.npy')]
    sorted_files = sorted(files, key = sort_key)
    
    for file_name in sorted_files:
        if file_name.endswith('.png'):
            file_path = os.path.join(folder_path, file_name)
            images.append(imageio.imread(file_path))
    fps = 5
    imageio.mimsave(output_path, images, fps = fps)
create_animation('folder_path', 'output_path_animation.mp4')
