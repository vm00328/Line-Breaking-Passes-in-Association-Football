# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:59:48 2023
@author: Vladislav Manolov

"""
from mplsoccer import Pitch
from matplotlib import animation
import pandas as pd
from sklearn.cluster import KMeans

"""
    data: a dataframe containing the entire tracking data for both teams
    df_home: a dataframe containing all tracking data for all 11 players of the home team
    df_home_10: a dataframe containing all tracking data for all players of the home team apart from the GK
    df_away: a dataframe containing tracking data for all 11 players of the away team
    df_ball: a dataframe containing tracking data for the ball
    event_data: a dataframe containing event data for AZ vs Vitesse
"""
raw_tracking_data = pd.read_feather("C:/Users/Vlado/Documents/GitHub/AZ/AZ/data/raw_match1.feather")
data = pd.read_csv("C:/Users/Vlado/Desktop/AZ/AZ/data/AZ_VIT_tracking_cleaned.csv", index_col = 0)
df_home = data[data['team'] == 0].copy()
df_home = df_home.drop(['index', 'team', 'minutes', 'seconds', 'cum_secs'], axis = 1)
df_home_10 = df_home.drop(df_home[df_home['jersey_no'] == 16].index)

df_home_sampled = df_home[(df_home['frame'] >= 1700742) & (df_home['frame'] < 1701142)].copy()


df_away = data[data['team'] == 1].copy()
df_away = df_away.drop(['index', 'team', 'minutes', 'seconds', 'cum_secs'], axis = 1)
df_ball = data[data['team'].isna()].copy()
df_ball = df_ball.drop(['index', 'team', 'jersey_no', 'minutes', 'seconds', 'cum_secs'], axis = 1)
df_ball_sampled = df_ball[(df_ball['frame'] >= 1700742) & (df_ball['frame'] < 1701142)].copy()
event_data = pd.read_csv("C://Users/Vlado/Desktop/AZ/AZ/data/event_data_azvit_cleaned.csv")

"""
    Checking whether the data in the columns are of consistent format.
    If non-numeric values are found, they are printed out and dealt with accordingly
"""

numeric_cols = ["frame", "team", "x", "y", "period_id", "minutes", "seconds", "vx", "vy"]
for col in numeric_cols:
    non_numeric_values = data[col].apply(lambda x: not str(x).isnumeric() if isinstance(x, str) else False)
    if any(non_numeric_values):
        print(f"Column '{col}' contains non-numeric values: {data[col][non_numeric_values]}")

"""
    Drawing the Pitch interface with the mplsoccer package:
        pitch_type: the type of the pitch ('tracab')
        goal_type: the visual display of both goals
        pitch_length: the length of the pitch
        pitch_width: the width of the pitch
"""

pitch = Pitch(pitch_type='tracab', goal_type='circle', pitch_length=105, pitch_width=68,
              axis=True, label=True, corner_arcs=True, line_color='white', pitch_color = "grass", stripe =True)
fig, ax = pitch.draw(figsize=(16, 10.4))

marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}                 
ball, = ax.plot([], [], ms = 11, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms = 15, markerfacecolor='#0000FF', **marker_kwargs) #blue
home, = ax.plot([], [], ms = 15, markerfacecolor='#ff0000', **marker_kwargs) #red

X = df_home_10[['x']].values
clusters_number = 3
kmeans = KMeans(n_clusters = clusters_number, random_state=0).fit(X)

df_home = df_home.assign(cluster=0)
df_home_10['cluster'] = kmeans.labels_
player_labels={}
prev_line_update = 0

player_lines={}
cluster_lines={}

def animate(i):
    
    """
    This function is used to create a visual representation (an animation) over a chosen set of frames.
    Within the function, we display the respective jersey numbers of both teams'players.
    K-Means clustering is applied on the player objects, after which lines connecting
    the players within the same formation line are drawn and updated every 5 frames.
    ----------
    
    Parameters
    ----------
    i : TYPE
        DESCRIPTION.

    Returns
    -------
    ball : TYPE
        DESCRIPTION.
    away : TYPE
        DESCRIPTION.
    home : TYPE
        DESCRIPTION.

    """
    ball.set_data(df_ball.iloc[i, 5], df_ball.iloc[i, 6])
    frame = df_ball.iloc[i, 2]
    home_frame = df_home_10[df_home_10['frame'] == frame]
    home_frame_2 = df_home[df_home['frame'] == frame]                  #reason: keeper number is shown
    home.set_data(df_home.loc[df_home.frame == frame, 'x'],
                  df_home.loc[df_home.frame == frame, 'y'])
    away.set_data(df_away.loc[df_away.frame == frame, 'x'],
                  df_away.loc[df_away.frame == frame, 'y'])
    
    global prev_line_update

    if i % 5 == 0:                                                      # Performing the computations every 5 frames
        for c in range(clusters_number):
            cluster_frame = home_frame[home_frame['cluster'] == c]
            if len(cluster_frame) > 1:
                x = cluster_frame['x'].values
                y = cluster_frame['y'].values
                
                y_sorted = sorted(y, reverse=True)                       # Sorting the y-values in descending order
                
                y_indices = [list(y).index(y_val) for y_val in y_sorted] # Obtaining the sorted indices of the y-values
                
                x_sorted = [x[i] for i in y_indices]                     # Sorting the x-values based on the sorted y-indices
                
                if c in player_lines:                                    # If a line for this cluster exists, update its coordinates
                    prev_line = player_lines[c]
                    prev_line.set_data(x_sorted, y_sorted)
                
                else:                                                    # Otherwise, we create a new line for this cluster
                    line, = ax.plot(x_sorted, y_sorted, linewidth=2, color='white')
                    player_lines[c] = line
                    
        prev_line_update = i                                             # Storing the current frame as the previous line update
    
    """
        Creating text objects (labels) for each player's jersey number.
        A label for each player on the field is created and added to the plot
        The labels are updated as the position of the respective player is updated (based on the 'x' and 'y' coordinates)
        The procedure is first applied to all players of the home team and subsequently to all players of the away team.
    """
    for i, row in df_home[df_home['frame'] == frame].iterrows(): 
        x, y = row['x'], row['y']
        jersey_no = int(row['jersey_no'])
        
        if i in player_labels:                                          # if the player already has a label, move it to the current position
            label = player_labels[i]
            label.set_text(str(jersey_no))
            label.xy = (x, y)
        
        else:                                                           # otherwise, create a new label and add it to the plot
            label = ax.annotate(str(jersey_no),
                                xy = (x, y),
                                fontsize = 11,
                                color = 'white',
                                ha = 'center',
                                va = 'center')
            player_labels[i] = label
            
    for i, row in df_away[df_away['frame'] == frame].iterrows():
        x, y = row['x'], row['y']
        jersey_no = int(row['jersey_no'])
        
        if i in player_labels:
          label = player_labels[i]
          label.set_text(str(jersey_no))
          label.xy = (x, y)
          
        else:
          label = ax.annotate(str(jersey_no),
                              xy=(x, y),
                              fontsize=11,
                              color='white',
                              ha='center',
                              va='center')
          player_labels[i] = label
        
    for i in list(player_labels):
        if i not in home_frame_2.index and i not in df_away[df_away['frame'] == frame].index:
            label = player_labels.pop(i)
            label.remove()   
            
    return ball, away, home

"""
    Computing the animation in the form of a .gif file with a total length equal to the length of the ball dataframe.
    The animation is then saved locally.
"""

anim = animation.FuncAnimation(fig, animate, frames=len(df_ball), interval=50, blit=True)
f = r"C://Users/Vlado/Documents/GitHub/AZ/AZ/Visualizations/proba68.gif"
writervideo = animation.PillowWriter(fps = 25)
anim.save(f, writer = writervideo)