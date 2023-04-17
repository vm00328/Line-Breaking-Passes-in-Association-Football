 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:08:32 2023

@author: miles
"""
import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch, Sbopen, VerticalPitch
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.simpledialog
import csv
from tkinter import colorchooser
import PIL.ImageGrab as ImageGrab
import time

az_file_path = 'C:/Users/Vlado/Documents/GitHub/AZ/AZ/Anderlecht_Prep/'
#az_save_path = 'C:/Users/Miles/Documents/GitHub/set_pieces_app/src/tactics/az_tactics/'





class TacticsApp:
    def __init__(self):
        self.root = tk.Tk()
        #self.root.iconbitmap('C:/Users/Miles/Documents/GitHub/set_pieces_app/data/logos/football.ico')
        self.root.title("Tactics App")
        self.canvas = tk.Canvas(self.root, width=1050, height=680, bg="#0e1117")
        self.canvas.pack()
        self.players, self.player_circles, self.player_team = [], [], []
        self.setup_pitch()

        # Add text box and save/load buttons
        self.filename_entry = tk.Entry(self.root, width=40)
        self.filename_entry.pack(side="right", anchor="e")
        tk.Button(self.root, text="Save", command=self.save_positions).pack(side="right", anchor="e")
        tk.Button(self.root, text="Load", command=self.load_positions).pack(side="right", anchor="e")
        tk.Button(self.root, text="Clear Players", command=self.clear_positions).pack(side="right", anchor="e")
        tk.Button(self.root, text="Flip Y", command=self.flip_y).pack(side="right", anchor="e")
        tk.Button(self.root, text="Toggle Drawing", command=self.toggle_draw).pack(side="left", anchor="w")
        tk.Button(self.root, text="Clear Drawing", command=self.clear_draw).pack(side="left", anchor="w")
        tk.Button(self.root, text="Drawing Options", command=self.drawing_options).pack(side="left", anchor="w") 
        tk.Button(self.root, text="Screenshot", command=self.take_screenshot).pack(side="left", anchor="w")

        # Set up drawing state
        self.drawing_enabled = False
        self.current_line = None
        self.drawing_type = 'straight'
        self.drawing_color = 'darkgrey' 
        self.line_width = 2


    def take_screenshot(self):
        # Get coordinates of app window
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        # Take screenshot of app window
        image = ImageGrab.grab((x, y, x+width, y+height-25))
        image.save(f"{az_file_path}screenshots/{time.strftime('%Y%m%d-%H%M%S')}.png")

    def toggle_draw(self):
        self.drawing_enabled = not self.drawing_enabled
        if self.drawing_enabled:
            self.canvas.bind("<Button-1>", self.start_draw)
            self.canvas.bind("<B1-Motion>", self.draw)
        else:
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.current_line = None

    def clear_draw(self):
        self.canvas.delete("draw")
   
    def start_draw(self, event):
        self.current_line = self.canvas.create_line(event.x, event.y, event.x, event.y, fill=self.drawing_color, width=self.line_width, tags="draw")  # Modified to use the drawing_color

    # Modified drawing options function
    def drawing_options(self):
        options_window = tk.Toplevel(self.root)
        options_window.transient(self.root)  # Make the options window a transient window of the main app window
        options_window.title("Drawing Options")
    
        # Drawing type selection
        tk.Label(options_window, text="Drawing type:").grid(row=0, column=0, sticky='w')
        drawing_type_var = tk.StringVar(value=self.drawing_type)
        tk.Radiobutton(options_window, text="Straight lines", variable=drawing_type_var, value="straight").grid(row=1, column=0, sticky='w')
        tk.Radiobutton(options_window, text="Freehand", variable=drawing_type_var, value="freehand").grid(row=2, column=0, sticky='w')
        tk.Radiobutton(options_window, text="Arrows", variable=drawing_type_var, value="arrow").grid(row=3, column=0, sticky='w')
        tk.Radiobutton(options_window, text="Curved lines", variable=drawing_type_var, value="curved").grid(row=4, column=0, sticky='w')  # Added curved line option
    
        # Color selection
        tk.Label(options_window, text="Drawing color:").grid(row=5, column=0, sticky='w')
        color_button = tk.Button(options_window, text="Choose Color", command=lambda: self.choose_color(color_button))
        color_button.grid(row=6, column=0, sticky='w')
    
        tk.Label(options_window, text="Line thickness:").grid(row=7, column=0, sticky='w')
        self.line_width_scale = tk.Scale(options_window, from_=1, to=10, orient=tk.HORIZONTAL)
        self.line_width_scale.set(self.line_width)  # Set the default line thickness
        self.line_width_scale.grid(row=8, column=0, sticky='w')
    
        # Apply and Cancel buttons
        tk.Button(options_window, text="Apply", command=lambda: self.apply_drawing_options(options_window, drawing_type_var.get(), color_button)).grid(row=9, column=0, sticky='w')

        x = self.root.winfo_x() + 10
        y = self.root.winfo_y() + 40
        options_window.geometry(f"+{x}+{y}")
    
    # Modified function to apply drawing options
    def apply_drawing_options(self, dialog, drawing_type, color_button):
        self.drawing_type = drawing_type
        self.line_width = self.line_width_scale.get()  # Add this line
        if self.drawing_enabled:
            self.toggle_draw()
            self.toggle_draw()
        dialog.destroy()

    # Added function to choose color
    def choose_color(self, color_button):
        color = colorchooser.askcolor()[1]
        if color:
            self.drawing_color = color
            color_button.config(bg=color)

    def draw(self, event):
        if not self.current_line:
            self.current_line = self.canvas.create_line(event.x, event.y, event.x, event.y, fill=self.drawing_color, width=self.line_width, tags="draw")
        else:
            if self.drawing_type == 'straight':
                self.canvas.coords(self.current_line, self.canvas.coords(self.current_line)[0], self.canvas.coords(self.current_line)[1], event.x, event.y)
            elif self.drawing_type == 'freehand':
                self.canvas.create_line(self.canvas.coords(self.current_line)[-2:], event.x, event.y, fill=self.drawing_color, width=self.line_width, tags="draw")
                self.current_line = self.canvas.create_line(event.x, event.y, event.x, event.y, fill=self.drawing_color, width=self.line_width)
            elif self.drawing_type == 'arrow':
                self.canvas.coords(self.current_line, self.canvas.coords(self.current_line)[0], self.canvas.coords(self.current_line)[1], event.x, event.y)
                self.canvas.itemconfigure(self.current_line, arrow=tk.LAST, arrowshape=(8, 10, 5), tags="draw")
            elif self.drawing_type == 'curved':
                start_x, start_y = self.canvas.coords(self.current_line)[0], self.canvas.coords(self.current_line)[1]
                mid_x, mid_y = (start_x + event.x) / 2, (start_y + event.y) / 2
                shift_amount = 50  # Change this value to adjust the curvature
                dx, dy = start_x - event.x, start_y - event.y
                length = (dx ** 2 + dy ** 2) ** 0.5
                control_x, control_y = mid_x + (shift_amount * dy) / length, mid_y - (shift_amount * dx) / length
                self.canvas.delete(self.current_line)  # Delete the old curved line
                self.current_line = self.canvas.create_line(
                    start_x, start_y, control_x, control_y, event.x, event.y,
                    fill=self.drawing_color, width=self.line_width, tags="draw", smooth=True
                )
                

    def start_drag(self, event):
        self.dragged_player = event.widget.find_withtag("current")[0]
        self.dragged_circle = self.player_circles[self.players.index(self.dragged_player)]
        self.start_x = event.x
        self.start_y = event.y

    def drag(self, event):
        x = event.x - self.start_x
        y = event.y - self.start_y
        self.canvas.move(self.dragged_player, x, y)
        self.canvas.move(self.dragged_circle, x, y)
        self.start_x = event.x
        self.start_y = event.y

    def clear_positions(self):
        for player in self.players:
            self.canvas.delete(player)
        for circle in self.player_circles:
            self.canvas.delete(circle)
        self.players = []
        self.player_circles = []
        self.player_team = []
            
    def flip_y(self):
        PITCH_HEIGHT = 680
        
        for player in self.players:
            x, y = self.canvas.coords(player)
            self.canvas.coords(player, x, PITCH_HEIGHT-y)
        for circle  in self.player_circles:
            x1, y1, x2, y2 = self.canvas.coords(circle)
            self.canvas.coords(circle, x1, PITCH_HEIGHT-y2, x2, PITCH_HEIGHT-y1)

    def load_positions(self):
        # clear current players and ball
        #for player in self.players:
        #    self.canvas.delete(player)
        #for circle in self.player_circles:
        #    self.canvas.delete(circle)
    
        # load new players and ball from CSV
        filename = az_file_path + self.filename_entry.get() + '.csv'
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                number = row['jersey_no']
                team = str(row['team_flag'])
                x = float(row['sb_x'])
                y = float(row['sb_y']) 
                if team == 'AZ Alkmaar':
                    fill = 'red'
                elif team == 'Opponent':
                    fill = '#4B0082'
                else:
                    fill = 'black'
    
                if number == '999':#0
                    circle = self.canvas.create_oval(x-5, y-5, x+5, y+5, outline='black', fill=fill, width=1)
                    player = self.canvas.create_text(x, y, text=number, fill='black', font=('Arial', 10))
                    self.players.append(player)
                    self.player_circles.append(circle)
                    self.canvas.tag_bind(player, "<Button-1>", self.start_drag)
                    self.canvas.tag_bind(player, "<B1-Motion>", self.drag)

                else:
                    circle = self.canvas.create_oval(x-10, y-10, x+10, y+10, outline='white', fill=fill, width=1.2)
                    player = self.canvas.create_text(x, y, text=number, fill='white', font=('Arial', 10, 'bold'), tags=('player',))
                    self.players.append(player)
                    self.player_circles.append(circle)
                    self.player_team.append(team)
                    self.canvas.tag_bind(player, "<Button-1>", self.start_drag)
                    self.canvas.tag_bind(player, "<B1-Motion>", self.drag)
                    self.canvas.tag_bind(player, "<Button-3>", self.edit_player_number)
    
    def save_positions(self):
        filename = az_file_path + self.filename_entry.get()
        filename+='.csv'
        positions = []
        for i, player in enumerate(self.players):
            x, y, *_ = self.canvas.coords(player)
            circle_x, circle_y, *_ = self.canvas.coords(self.player_circles[i])
            #team = self.player_team(player)
            '''
            get correct player_team value
            '''
            team = self.player_team[i]
            positions.append((self.canvas.itemcget(player, 'text'), team, x, y, circle_x, circle_y))
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["jersey_no", "team_flag", "sb_x", "sb_y", "circle_x", "circle_y"])
            writer.writerows(positions)      
            
    def edit_player_number(self, event):
        player = event.widget.find_withtag('current')[0]
        new_number = tk.simpledialog.askinteger("Edit Player Number", "Enter a new player number:")
        self.canvas.itemconfig(player, text=new_number)

    def setup_pitch(self):
        # penalty boxes
        self.canvas.create_rectangle(0, 153, 157.5, 527, fill="", outline="white", width=2)
        self.canvas.create_rectangle(1050, 153, 892.5, 527, fill="", outline="white", width=2)

        # 6-yard boxes
        self.canvas.create_rectangle(0, 255, 52.5, 425, fill="", outline="white", width=2)
        self.canvas.create_rectangle(997.5, 255, 1050, 425, fill="", outline="white", width=2)
        
        # goals
        self.canvas.create_rectangle(0, 306, 8, 374, fill="", outline="white", width=2)
        self.canvas.create_rectangle(1050, 306, 1046, 374, fill="", outline="white", width=2)

        # penalty spots
        self.canvas.create_oval(100-3, 340-3, 100+3, 340+3, fill="white", outline="white")
        self.canvas.create_oval(950-3, 340-3, 950+3, 340+3, fill="white", outline="white")

        # Add halfway line
        self.canvas.create_line(525, 0, 525, 680, fill="white", width=2)
        self.canvas.create_rectangle(4, 4, 1050, 680, fill="", outline="white", width=2)

        # centre spot
        self.canvas.create_oval(525-3, 340-3, 525+3, 340+3, fill="white", outline="white")
        
        var = 70
        # penalty boxes "D"
        self.canvas.create_arc(157.5-(var/2), 340-(var), 157.5+(var/2), 340+(var), start = 270,extent=180, outline="white", width =2)
        self.canvas.create_arc(892.5-(var/2), 340-(var), 892.5+(var/2), 340+(var), start = 90,extent=180, outline="white", width =2)
        
        # centre circle
        self.canvas.create_oval(525-var, 340-var, 525+var, 340+var, outline="white", width = 2)


    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TacticsApp()
    app.run()











   






