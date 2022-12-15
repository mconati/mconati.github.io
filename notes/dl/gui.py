import tkinter as tk
import numpy as np
from value_iteration import gridworld
import argparse
import matplotlib as mpl


trap_cost = -1
goal_value = 1
horizon = 1000
gamma = 0.9
action_certainty = 0.8
noise = 1-action_certainty
policy_map = {0:"up", 1:"right", 2:"down", 3:"left"}


def GUILoop(gridworld):
    # Create the main window
    root = tk.Tk()
    root.geometry('1000x1000')
    root.title("Room Simulation")

    #Create a frame to hold button grid
    frame = tk.Frame(root, bg='black')
    frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Create a 2D array of buttons to represent the grid
    buttons = np.empty_like((gridworld), dtype=object)
    grid = np.zeros_like((buttons), dtype=np.int8)
    print("here")
    
    n_cols = buttons.shape[1]
    n_rows = buttons.shape[0]
    for i in range(n_rows):
        for j in range(n_cols):
            # Create a button for each element in the grid
            if gridworld[i, j] == 0:
                button = tk.Button(frame, text="   ", bg='white', fg='black')
                grid[i,j] = 0
            if gridworld[i, j] == 'o':
                button = tk.Button(frame, text="o", bg='black', fg='white')
                grid[i,j] = 1
            if gridworld[i, j] == 'g':
                button = tk.Button(frame, text="g", bg='green', fg='black')
                grid[i,j] = 2
            if gridworld[i, j] == 't':
                button = tk.Button(frame, text="t", bg='red', fg='black')
                grid[i,j] = 3
            button.config(height=int(3*10/n_rows), width=int(6*10/n_cols))
            button.grid(row=i, column=j)
            buttons[i, j] = button
            

            # Bind functions to the button's click events
            button.bind('<Button-1>', lambda e, i=i, j=j: left_click(tempGrid, grid, e, i, j))
            button.bind('<Button-3>', lambda e, i=i, j=j: right_click(buttons, tempGrid, grid, e, i, j))
    tempGrid = np.copy(grid)
    # Create a frame to hold the buttons on the side
    side_frame = tk.Frame(root, bg='white')
    side_frame.place(relx=0, rely=0.1, anchor=tk.W)

    # Create the "Policy Iteration" button
    policy_iteration_button = tk.Button(side_frame, text="Policy Iteration")
    policy_iteration_button.pack(side=tk.TOP)

    # Create the "Value Iteration" button
    value_iteration_button = tk.Button(side_frame, text="Value Iteration")
    value_iteration_button.pack(side=tk.TOP)
    value_iteration_button.bind('<Button-1>', lambda e: value_iteration(grid, buttons))

    # Clear button
    Clear_button = tk.Button(side_frame, text="Clear", bg = 'white')
    Clear_button.pack(side=tk.TOP)
    Clear_button.bind('<Button-1>', lambda e: clear(grid, tempGrid, buttons))


    # Create a function that updates the global variable when the user enters a new value
    def update_certainty(event):
        global action_certainty
        # Get the value entered by the user from the event object
        new_value = event.widget.get()
        # Convert the value to a float and update the global variable
        action_certainty = float(new_value)
        event.widget.delete(0, tk.END)
        event.widget.insert(0, "Certainty=" + str(action_certainty))
    def on_typing(event):
        # Clear the default text when the user starts typing
        event.widget.delete(0, tk.END)


    # Create a tkinter entry box
    entry = tk.Entry(side_frame)
    entry.insert(0, "Certainty=" + str(action_certainty))

    # Bind the entry box to the update function
    entry.bind("<Return>", update_certainty)
    entry.bind("<Button-1>", on_typing)
    # Pack the entry box to the root window
    entry.pack(side=tk.TOP)

    
    root.mainloop()

def clear(grid, tempGrid, buttons):
    grid = np.copy(tempGrid)
    for i in range(0,grid.shape[0]):
        for j in range(0,grid.shape[1]):
            button = buttons[i, j]
            if grid[i, j] == 1:
                button.config(bg='black', text = 'o', fg='white')
                grid[i, j] = 1
            elif grid[i, j] == 2:
                button.config(bg='green', text = 'g', fg='black')
                grid[i, j] = 2
            elif grid[i, j] == 3:
                button.config(bg='red', text = 't', fg='black')
                grid[i, j] = 3
            elif grid[i, j] == 0:
                button.config(bg='white', text = '', fg='black')
                grid[i, j] = 0
    print("Reset from Navigation mode")

    

def value_iteration(grid, buttons):
    print("Entering Navigation mode")
    gridworld = np.copy(grid)

    try:
        startingExists = True
        point = list([np.where(gridworld==4)[0][0], np.where(gridworld==4)[1][0]])
    except:
        startingExists = False
    values = np.zeros_like(gridworld, dtype=float)
    policies = np.zeros_like(gridworld)
    for h in range(1, horizon):
        temp = np.copy(values)
        for i in range(0,values.shape[0]):
            for j in range(0,values.shape[1]):
                if (gridworld[i, j]==3 or gridworld[i,j]==1 or gridworld[i,j]==2):
                    continue
                valid = np.zeros(4)
                if ((i-1)>-1):
                    reward = 0
                    if ((gridworld[i-1][j]!=1)):
                        if ((gridworld[i-1][j]==2)):
                            reward = goal_value
                        if ((gridworld[i-1][j]==3)):
                            reward = trap_cost
                        valid[0]=1
                        up = reward + values[i-1][j]
                if ((j-1)>-1):
                    reward = 0
                    if ((gridworld[i][j-1]!=1)):
                        if ((gridworld[i][j-1]==2)):
                            reward = goal_value
                        if ((gridworld[i][j-1]==3)):
                            reward = trap_cost
                        valid[3]=1
                        left = reward + values[i][j-1]
                if ((i+1)<gridworld.shape[0]):
                    reward = 0
                    if ((gridworld[i+1][j]!=1)):
                        if ((gridworld[i+1][j]==2)):
                            reward = goal_value
                        if ((gridworld[i+1][j]==3)):
                            reward = trap_cost
                        valid[2]=1
                        down = reward + values[i+1][j]
                if ((j+1)<gridworld.shape[1]):
                    reward = 0
                    if ((gridworld[i][j+1]!=1)):
                        if ((gridworld[i][j+1]==2)):
                            reward = goal_value
                        if ((gridworld[i][j+1]==3)):
                            reward = trap_cost
                        valid[1]=1
                        right = reward + values[i][j+1]
                best = 0
                reward = -np.inf
                rewards = np.zeros_like(valid)
                for action in np.where(valid==1)[0]:
                    match action:
                        case 0:
                            if  up>reward:
                                best = 0
                                reward = up
                            rewards[0] = up
                        case 1:
                            if  right>reward:
                                best = 1
                                reward = right
                            rewards[1] = right
                        case 2:
                            if  down>reward:
                                best = 2
                                reward = down
                            rewards[2] = down
                        case 3:
                            if  left>reward:
                                best = 3
                                reward = left
                            rewards[3] = left
                newValue = 0
                for x, val in enumerate(rewards):
                    if (x==best and valid[x]==1):
                        newValue+=action_certainty*gamma*val
                    elif (x==best and valid[x]==0):
                        newValue+=gamma*values[i][j]
                    elif(valid[x]==0):
                        newValue+=noise/3*gamma*values[i][j]
                    elif(valid[x]==1):
                        newValue+=noise/3*gamma*val
                temp[i, j] = newValue
                policies[i, j] = best
        if h>3 and np.allclose(temp, values, rtol=1e-04):
            print("Converged in "+str(h)+" iterations")
            break
        else:
            values = temp



    for i in range(0,values.shape[0]):
        for j in range(0,values.shape[1]):  
            if grid[i,j]==0:    
                mix = (values[i,j]-trap_cost)/(goal_value-trap_cost)
                buttons[i,j].config(bg=colorFader(mix))
                if(buttons[i,j]['text']=='P'): buttons[i,j].config(text='')

    if startingExists:
        goals_numpy = np.where(grid==2)
        goals = []
        for i in range(goals_numpy[0].shape[0]):
            goals.append([goals_numpy[0][i], goals_numpy[1][i]])
        goals = list(goals)
        origVal = values[point[0], point[1]]
        origPoint = point
        buttons[origPoint[0], origPoint[1]].config(text='s')
        broken = False
        while (point not in goals):
            move = policies[point[0], point[1]]
            point = makeMove(point, move)
            newVal = values[point[0], point[1]]
            if newVal<0 or (grid[origPoint[0], origPoint[1]]!=0 and grid[origPoint[0], origPoint[1]]!=4 and [origPoint[0], origPoint[1]]!=2):
                buttons[origPoint[0], origPoint[1]].config(text='S!', fg = 'black')
                broken = True
                break
            origVal = newVal
            origPoint = point
            buttons[point[0], point[1]].config(text='P', fg = 'black')
        if not broken:
            buttons[point[0], point[1]].config(text = 'g', fg = 'black')


def colorFader(mix, c1='red',c2='green'): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def makeMove(point, move):
    if move ==0:
        newpoint = [point[0]-1, point[1]]
    if move ==1:
        newpoint = [point[0], point[1]+1]
    if move ==2:
        newpoint = [point[0]+1, point[1]]
    if move ==3:
        newpoint = [point[0], point[1]-1]
    return newpoint

# Function to handle left clicks on the buttons
def left_click(tempGrid, grid, event, i, j):
    # Get the button that was clicked
    button = event.widget
    # Update the button's text and the numpy array based on its current state
    if grid[i, j] == 0:
        button.config(bg='black', text = 'o',fg='white')
        grid[i, j] = 1
        tempGrid[i,j] = 1
    elif grid[i, j] == 1:
        button.config(bg='green', text='g', fg='black')
        grid[i, j] = 2
        tempGrid[i,j] = 2
    elif grid[i, j] == 2:
        button.config(bg='red', text='t', fg='black')
        grid[i, j] = 3
        tempGrid[i,j] = 3
    elif grid[i, j] == 3:
        button.config(bg='white', text='', fg='black')
        grid[i, j] = 0
        tempGrid[i,j] = 0


# Function to handle right clicks on the buttons
def right_click(buttons, tempGrid, grid, event, i, j):
    # Get the button that was clicked
    button = event.widget
    prevStart = False
    if grid[i,j]==4:
        prevStart = True
    # Reset any previously-selected starting square to an empty square
    start_i, start_j = np.where(grid == 4)
    if len(start_i) > 0:
        buttons[start_i[0], start_j[0]].config(bg='white', text='')
        grid[start_i[0], start_j[0]] = 0

    # Update the button's text and the numpy array to make it the starting square
    if prevStart!=True:
        button.config(bg='yellow', text='s')
        grid[i, j] = 4
        tempGrid[i,j] = 4
    else:
        button.config(bg='white', text='')
        grid[i,j] = 0
        tempGrid[i,j] = 0
    

# Start the GUI event loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', dest = 'size', default = 20, help = 'size of gridworld')
    parser.add_argument('-o', '--obstacles', dest = 'obstacles', default = 30, help = 'number of obstacles')
    parser.add_argument('-g', '--goals', dest = 'goals', default = 2, help = 'number of goals')
    parser.add_argument('-t', '--traps', dest = 'traps', default = 30, help = 'number of traps')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = main()
    gridworld = gridworld(args)
    GUILoop(gridworld)
    

