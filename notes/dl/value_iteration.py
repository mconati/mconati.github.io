import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


trap_cost = -1
goal_value = 1
horizon = 1000
gamma = 0.9
action_certainty = 0.8
noise = 1-action_certainty
policy_map = {0:"up", 1:"right", 2:"down", 3:"left"}

def gridworld(args):
    s = args.size
    o = args.obstacles
    g = args.goals
    t = args.traps
    total_special_squares = o+g+t
    if((o+g+t)>(s*s)):
        print("Can't make this gridworld")
        return -1
    gridworld = np.zeros((s, s), dtype=object)
    choices = np.random.choice(s*s, total_special_squares, replace=False)
    print(gridworld.shape)
    counter = 0
    for obstacle in range(0, o):
        row = int(np.floor(choices[counter]/s))
        col = choices[counter]%s
        gridworld[row,col] = 'o'
        counter+=1
    for obstacle in range(0, g):
        row = int(np.floor(choices[counter]/s))
        col = choices[counter]%s
        gridworld[row,col] = 'g'
        counter+=1
    for obstacle in range(0, t):
        row = int(np.floor(choices[counter]/s))
        col = choices[counter]%s
        gridworld[row,col] = 't'
        counter+=1
    return gridworld



def value_iteration(gridworld):
    values = np.zeros_like(gridworld, dtype=float)
    policies = np.zeros_like(gridworld)
    for h in range(1, horizon):
        temp = np.copy(values)
        for i in range(0,values.shape[0]):
            for j in range(0,values.shape[1]):
                if (gridworld[i, j]=='o' or gridworld[i,j]=='t' or gridworld[i,j]=='g'):
                    continue
                valid = np.zeros(4)
                if ((i-1)>-1):
                    reward = 0
                    if ((gridworld[i-1][j]!='o')):
                        if ((gridworld[i-1][j]=='g')):
                            reward = goal_value
                        if ((gridworld[i-1][j]=='t')):
                            reward = trap_cost
                        valid[0]=1
                        up = reward + values[i-1][j]
                if ((j-1)>-1):
                    reward = 0
                    if ((gridworld[i][j-1]!='o')):
                        if ((gridworld[i][j-1]=='g')):
                            reward = goal_value
                        if ((gridworld[i][j-1]=='t')):
                            reward = trap_cost
                        valid[3]=1
                        left = reward + values[i][j-1]
                if ((i+1)<gridworld.shape[0]):
                    reward = 0
                    if ((gridworld[i+1][j]!='o')):
                        if ((gridworld[i+1][j]=='g')):
                            reward = goal_value
                        if ((gridworld[i+1][j]=='t')):
                            reward = trap_cost
                        valid[2]=1
                        down = reward + values[i+1][j]
                if ((j+1)<gridworld.shape[1]):
                    reward = 0
                    if ((gridworld[i][j+1]!='o')):
                        if ((gridworld[i][j+1]=='g')):
                            reward = goal_value
                        if ((gridworld[i][j+1]=='t')):
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
    return values, policies
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', dest = 'size', default = 20, help = 'size of gridworld')
    parser.add_argument('-o', '--obstacles', dest = 'obstacles', default = 30, help = 'number of obstacles')
    parser.add_argument('-g', '--goals', dest = 'goals', default = 2, help = 'number of goals')
    parser.add_argument('-t', '--traps', dest = 'traps', default = 30, help = 'number of traps')

    args = parser.parse_args()
    return args

def prettyPrint(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = ' '.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

if __name__ == "__main__":
    args = main()
    start = time.time()
    gridworld = gridworld(args)

    if (gridworld !=-1).any():
        gridVals, policies = value_iteration(gridworld)
        gridVals = np.around(gridVals.astype(float), decimals=3)
        end = time.time()
        print("Building the gridworld took:" + str(end-start)+" seconds")
        # create discrete colormap
        cmap = colors.ListedColormap(['red','black', 'brown','peru', 'orange', 'yellow','green'])
        bounds = [2*trap_cost,trap_cost,0.0001,0.25*goal_value,0.5*goal_value, 0.75*goal_value, 0.99*goal_value, 1*goal_value]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(0.5, args.size, 1));
        ax.set_yticks(np.arange(0.5, args.size, 1));
        for i in range(policies.shape[0]):
            for j in range(policies.shape[1]):
                policies[i,j] = policy_map[policies[i][j]]
                if gridworld[i,j]=='o':
                    policies[i,j] = 'obstacle'
                    circle = plt.Rectangle((j-0.5, i-0.5), 1, 1,  color='k')
                    ax.add_patch(circle)
                elif gridworld[i,j]=='t':
                    policies[i,j] = 'trap!'
                    circle = plt.Rectangle((j-0.5, i-0.5), 1, 1,  color='r')
                    ax.add_patch(circle)
                elif gridworld[i,j]=='g':
                    policies[i,j] = 'GOAAAAAL'
                    gridVals[i,j] = goal_value
                    circle = plt.Circle((j, i), 0.3, color='g')
                    ax.add_patch(circle)
                elif gridVals[i][j]<=0:
                    policies[i][j] = 'Stuck'
                if policies[i,j]=='up':
                    ax.arrow(j, i+.3, 0, -0.22, width=.1, edgecolor='black', facecolor='white')
                if policies[i,j]=='down':
                    ax.arrow(j, i-.3, 0, 0.22, width=.1, edgecolor='black', facecolor='white')
                if policies[i,j]=='left':
                    ax.arrow(j+0.3, i, -0.22, 0, width=.1, edgecolor='black', facecolor='white')
                if policies[i,j]=='right':
                    ax.arrow(j-0.3, i, 0.22, 0, width=.1, edgecolor='black', facecolor='white')
                    
        im1 = ax.imshow(gridVals, interpolation='none')
        prettyPrint(policies)
        prettyPrint(gridVals)
        fig.colorbar(im1, orientation='vertical')
        plt.show()
        
