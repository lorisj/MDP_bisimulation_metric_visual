import random
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as sp
# TODO: Use https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html


class Node:
    def __init__(self):
        self.transitions = {}
        self.rewards = {}

class Grid:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.grid = [[Node() for _ in range(columns)] for _ in range(rows)]
        for row in range(rows):
            for column in range(columns):
                self.set_random_probabilities_and_rewards(row, column)


    def set_probabilities(self, row, column,probabilities):
        node = self.grid[row][column]
        node.transitions = probabilities
    

    def set_rewards(self, row, column, rewards):
        node = self.grid[row][column]
        node.rewards = rewards
    
    def set_random_probabilities_and_rewards(self, row, column,reward_max=1):
        node = self.grid[row][column]
        num_connections = 0

        # Check for neighboring nodes
        if row > 0:  # Up
            num_connections += 1
        if row < self.rows - 1:  # Down
            num_connections += 1
        if column > 0:  # Left
            num_connections += 1
        if column < self.columns - 1:  # Right
            num_connections += 1

        if num_connections == 0:
            return

        probabilities = generate_random_pmf(num_connections)
        rewards_dist = generate_random_pmf(num_connections)
        transitions = {}
        rewards = {}

        index = 0

        if row > 0:  # Up
            transitions['up'] = probabilities[index]
            rewards['up'] = rewards_dist[index] * reward_max
            index += 1

        if row < self.rows - 1:  # Down
            transitions['down'] = probabilities[index]
            rewards['down'] = rewards_dist[index] * reward_max
            index += 1

        if column > 0:  # Left
            transitions['left'] = probabilities[index]
            rewards['left'] = rewards_dist[index] * reward_max
            index += 1

        if column < self.columns - 1:  # Right
            transitions['right'] = probabilities[index]
            rewards['right'] = rewards_dist[index] * reward_max
            index += 1

        node.transitions = transitions
        node.rewards = rewards

    def visualize(self, start_node=""):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.columns - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect('equal')

        for row in range(self.rows):
            for column in range(self.columns):
                node = self.grid[row][column]
                transitions = node.transitions
                rewards = node.rewards

                x = column
                y = self.rows - 1 - row

                # Draw arrows for transitions
                if 'up' in transitions:
                    #ax.arrow(x, y, 0, 0.4, fc='black', ec='black', head_width=0.1, head_length=0.1)
                    ax.text(x, y + 0.2, f'{transitions["up"]:.2f}', ha='center', va='bottom', color='red')
                if 'down' in transitions:
                    #ax.arrow(x, y, 0, -0.4, fc='black', ec='black', head_width=0.1, head_length=0.1)
                    ax.text(x, y - 0.2, f'{transitions["down"]:.2f}', ha='center', va='top', color='red')

                if 'left' in transitions:
                    #ax.arrow(x, y, -0.4, 0, fc='black', ec='black', head_width=0.1, head_length=0.1)
                    ax.text(x - 0.2, y, f'{transitions["left"]:.2f}', ha='right', va='center', color='red')

                if 'right' in transitions:
                    #ax.arrow(x, y, 0.4, 0, fc='black', ec='black', head_width=0.1, head_length=0.1)
                    ax.text(x + 0.2, y, f'{transitions["right"]:.2f}', ha='left', va='center', color='red')

                # Draw nodes as rectangles
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)

                # Print rewards
                for direction, reward in rewards.items():
                    if direction in transitions:
                        if direction == 'up':
                            ax.text(x, y + 0.15, f'{reward:.2f}', ha='center', va='center', color='blue')
                        elif direction == 'down':
                            ax.text(x, y - 0.15, f'{reward:.2f}', ha='center', va='center', color='blue')
                        elif direction == 'left':
                            ax.text(x - 0.15, y, f'{reward:.2f}', ha='center', va='center', color='blue')
                        elif direction == 'right':
                            ax.text(x + 0.15, y, f'{reward:.2f}', ha='center', va='center', color='blue')

        if start_node !="":
            self.draw_arrows_with_placeholder(ax, start_node)

        
        plt.xticks(np.arange(0, self.columns, 1))
        plt.yticks(np.arange(0, self.rows, 1))
        plt.grid(True)
        plt.show()

        
    def print(self):
        for row in range(self.rows):
            for column in range(self.columns):
                node = self.grid[row][column]
                transitions = node.transitions
                rewards = node.rewards

                print(f'Node ({row}, {column}):')

                if 'up' in transitions:
                    print(f'  Transition Probability (up): {transitions["up"]:.2f}')
                    print(f'  Reward (up): {rewards["up"]:.2f}')

                if 'down' in transitions:
                    print(f'  Transition Probability (down): {transitions["down"]:.2f}')
                    print(f'  Reward (down): {rewards["down"]:.2f}')

                if 'left' in transitions:
                    print(f'  Transition Probability (left): {transitions["left"]:.2f}')
                    print(f'  Reward (left): {rewards["left"]:.2f}')

                if 'right' in transitions:
                    print(f'  Transition Probability (right): {transitions["right"]:.2f}')
                    print(f'  Reward (right): {rewards["right"]:.2f}')



    def save(self, file_path):
        data = {
            'rows': self.rows,
            'columns': self.columns,
            'nodes': []
        }

        for row in range(self.rows):
            for column in range(self.columns):
                node = self.grid[row][column]
                transitions = node.transitions
                rewards = node.rewards

                data['nodes'].append({
                    'row': row,
                    'column': column,
                    'transitions': transitions,
                    'rewards': rewards
                })

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Grid saved to {file_path}.")
    def load(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        rows = data['rows']
        columns = data['columns']
        grid = Grid(rows, columns)

        for node_data in data['nodes']:
            row = node_data['row']
            column = node_data['column']
            transitions = node_data['transitions']
            rewards = node_data['rewards']

            grid.grid[row][column].transitions = transitions
            grid.grid[row][column].rewards = rewards

        print(f"Grid loaded from {file_path}.")
        return grid

    def placeholder_function(self, loc1, loc2):
        # loc1 and loc2 are tuples representing the (row, column) coordinates

        return sp.wasserstein_distance(self.get_padded_transitions(loc1), self.get_padded_transitions(loc2))

    def draw_arrows_with_placeholder(self, ax, start_node):
        already_printed = []

        
        row,col = start_node
        start_x = col
        start_y = self.rows - 1 - row

        for r in range(self.rows):
            for c in range(self.columns):
                end_node = (r, c)
                end_x = c
                end_y = self.rows - 1 - r
                #man_distance = abs(start_x - end_x) + abs(start_y-end_y) 
                if start_node != end_node and (end_node, start_node) not in already_printed:
                    # Draw an arrow from start_node to end_node
                    ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                            fc='black', ec='red', head_width=0, head_length=0.1)

                    # Display placeholder_function() between start_node and end_node
                    ax.text((start_x + end_x)/ 2, (start_y + end_y) / 2,
                            f'{self.placeholder_function(start_node, end_node):.2f}',color='orange',
                            #f"({row},{col}) -> ({r},{c})",color='orange',
                            ha='center', va='center')

                    already_printed.append((start_node,end_node))


    def get_padded_transitions(self,loc):
        row, col = loc
        node = self.grid[row][col]
        transitions = node.transitions

        # Check the number of connections
        num_connections = len(transitions)

        # Create an array of zeros of size 4
        probabilities = [0.0] * 4

        # Assign the probabilities to the corresponding positions in the array
        if 'up' in transitions:
            probabilities[0] = transitions['up']
        if 'down' in transitions:
            probabilities[1] = transitions['down']
        if 'left' in transitions:
            probabilities[2] = transitions['left']
        if 'right' in transitions:
            probabilities[3] = transitions['right']

        return probabilities



# Function to generate random PMF
def generate_random_pmf(n):
    pmf = [random.random() for _ in range(n)]
    total = sum(pmf)
    pmf = [p / total for p in pmf]
    return pmf



grid = Grid(3,3)
#grid = Grid.load("testgrid.json")





grid.visualize((0,1))




