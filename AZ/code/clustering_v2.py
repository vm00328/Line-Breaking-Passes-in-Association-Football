# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:59:48 2023
@author: Vladislav Manolov
"""
from mplsoccer import Pitch
from matplotlib import animation
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("C:/Users/Vlado/Desktop/AZ/AZ/data/AZ_VIT_tracking_cleaned.csv")
df_home = data[data['team']==0]
df_home = df_home[(df_home['frame'] >= 1700842) & (df_home['frame'] < 1700942)].copy() #1690842
df_home_10 = df_home.drop(df_home[df_home['jersey_no'] == 16].index)
df_away = data[data['team']==1] #MISSING frame stamp but works?
df_ball = data[data['team'].isna() & (data['frame'] >=1700842) & (data['frame'] < 1700942)].copy()

pitch = Pitch(pitch_type='tracab', goal_type='line', pitch_length=105, pitch_width=68,
              axis=True, label=True, corner_arcs=True, line_color='white', pitch_color = "grass", stripe =True)
fig, ax = pitch.draw(figsize=(16, 10.4))

marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}                 
ball, = ax.plot([], [], ms = 11, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms = 15, markerfacecolor='#0000FF', **marker_kwargs) #blue
home, = ax.plot([], [], ms = 15, markerfacecolor='#ff0000', **marker_kwargs) #red

X = df_home_10[['x']].values
kmeans = KMeans(n_clusters=3, random_state=0).fit(X) # Perform K-means clustering on x coordinates of the players for each team

df_home = df_home.assign(cluster=0)                  # Add cluster labels to the dataframe
df_home_10['cluster'] = kmeans.labels_
player_labels={}
prev_line_update = 0
player_lines={}

def animate(i):
    ball.set_data(df_ball.iloc[i, 5], df_ball.iloc[i, 6])
    frame = df_ball.iloc[i, 2]
    home_frame = df_home_10[df_home_10['frame'] == frame]
    home_frame_2 = df_home[df_home['frame'] == frame] #reason: keeper number is shown
    home.set_data(df_home.loc[df_home.frame == frame, 'x'],
                  df_home.loc[df_home.frame == frame, 'y'])
    away.set_data(df_away.loc[df_away.frame == frame, 'x'],
                  df_away.loc[df_away.frame == frame, 'y'])
    
    global prev_line_update
    if i % 5 == 0:
        for c in range(3):
            cluster_frame = home_frame[home_frame['cluster'] == c]
            if len(cluster_frame) > 1:
                x = cluster_frame['x'].values
                y = cluster_frame['y'].values
                
                if c in player_lines:       # If a line for this cluster already exists, update its coordinates
                    prev_line = player_lines[c]
                    prev_line.set_data(x, y)
                
                else:                       # Otherwise, create a new line for this cluster
                    line, = ax.plot(x, y, linewidth=2, color='white')
                    player_lines[c] = line
        prev_line_update = i                # Store the current frame as the previous line update
    
    # Create Text objects for each player's jersey number
    for i, row in df_home[df_home['frame']==frame].iterrows(): 
        x, y = row['x'], row['y']
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
            
    for i, row in df_away[df_away['frame']==frame].iterrows():
        x, y = row['x'], row['y']
        jersey_no = int(row['jersey_no'])
        if i in player_labels:
          label = player_labels[i]
          label.set_text(str(jersey_no))
          label.xy = (x, y)
        else:
          label = ax.annotate(str(jersey_no), xy=(x, y), fontsize=11, color='white', ha='center', va='center')
          player_labels[i] = label
        
    for i in list(player_labels):           # Remove labels for players who are not in the current frame
        if i not in home_frame_2.index and i not in df_away[df_away['frame']==frame].index:
            label = player_labels.pop(i)
            label.remove()    
    return ball, away, home

anim = animation.FuncAnimation(fig, animate, frames=len(df_ball), interval=50, blit=True)
f = r"C://Users/Vlado/Documents/GitHub/AZ/AZ/animations/proba60.gif"
writervideo = animation.PillowWriter(fps = 25)
anim.save(f, writer = writervideo)