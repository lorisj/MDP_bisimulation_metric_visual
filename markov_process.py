import json
import os
import numpy as np
import cvxpy as cp

def wasserstein_metric(P, Q, h) -> float:
        
    if len(P) > len(Q): # pad Q with zeros
        Q = np.concatenate((Q, np.zeros(len(P) - len(Q))))
        print("WARNING: probability input to wasserstein_metric() function has differing lengths.")
    elif len(Q) > len(P):
        P = np.concatenate((P, np.zeros(len(Q) - len(P))))
        print("WARNING: probability input to wasserstein_metric() function has differing lengths.")

    u = cp.Variable(len(P))

    # Objective
    objective = cp.Maximize(sum((P[i] - Q[i]) * u[i] for i in range(len(P))))

    # Constraints
    constraints = [u[i] - u[j] <= h[i][j] for i in range(len(P)) for j in range(len(P))]

    # Problem
    problem = cp.Problem(objective, constraints)

    # Solve
    problem.solve()
    old_return = cp.max(u).value
    # Maybe check if old_return exists? return "?" if not
    return old_return.item()



class MarkovDecisionProcess:
    def __init__(self, states, actions, transition_function, reward_function, discount_factor):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.discount_factor = discount_factor
        self.policy = {state: actions[0] for state in states}  # Set an initial policy
        self.values = {state: 0 for state in states}  # Initialize value table
        num_states = len(states)
        self.metric  = np.ones([num_states,num_states])- np.eye(num_states) # Initial bisimulation metric

    def set_policy(self, policy):
        """
        Set a policy to be used in the MDP.

        Args:
            policy: A dictionary mapping states to actions.
        """
        self.policy = policy

    def get_next_state(self, state, action):
        """
        Get the next state given the current state and action.

        Args:
            state: The current state.
            action: The action taken.

        Returns:
            The next state.
        """
        return self.transition_function[state][action]
    

    def get_reward(self, state, action):
        """
        Get the reward for a state transition.

        Args:
            state: The current state.
            action: The action taken.

        Returns:
            The reward for the state transition.
        """
        return self.reward_function[state][action]

    def get_expected_value(self, state):
        """
        Get the expected value of a state under the current policy.

        Args:
            state: The state to get the expected value of.

        Returns:
            The expected value of the state.
        """
        action = self.policy[state]
        expected_value = 0
        for next_state in self.states:
            transition_prob = self.transition_function[state][action][next_state]
            reward = self.reward_function[state][action]
            expected_value += transition_prob * (reward + self.discount_factor * self.values[next_state])
        return expected_value

    def value_iteration(self, tolerance=1e-6):
        """
        Perform value iteration to find the optimal policy.

        Args:
            tolerance: The tolerance for the stopping condition.

        Returns:
            The optimal policy.
        """
        while True:
            delta = 0
            for state in self.states:
                old_value = self.values[state]
                self.values[state] = max(sum(
                    self.transition_function[state][action][next_state] * (self.reward_function[state][action][next_state] + self.discount_factor * self.values[next_state])
                    for next_state in self.states) for action in self.actions)
                delta = max(delta, abs(old_value - self.values[state]))
            if delta < tolerance:
                break

        # After convergence, update the policy based on the optimal value function
        for state in self.states:
            self.policy[state] = max(self.actions, key=lambda action: sum(
                    self.transition_function[state][action][next_state] * (self.reward_function[state][action][next_state] + self.discount_factor * self.values[next_state])
                    for next_state in self.states))
        return self.policy
    
    
    def iterate_bisimulation_matrix(self):
        """
        Generate a symmetric matrix where each entry i,j represents the iterated bisimulation metric between the 
        transition probabilities of states i and j under any action.

        Parameters:
            h (2D numpy array): The previous metric used to compute the new metric.

        Returns:
            new_metric_matrix (2D numpy array): A symmetric matrix where each entry i,j is the 
            iterated bisimulation between the transition probabilities of states i and j under any action.
        """
        num_states = len(self.states)
        new_metric_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(i+1):
                current_distance = max(
                    #(1-self.discount_factor) * abs(self.get_reward(i,action)) + self.discount_factor *
                    wasserstein_metric(
                        self.get_probability_vector(i, action),
                        self.get_probability_vector(j, action),
                        self.metric
                    )
                    for action in self.actions
                )

                new_metric_matrix[i, j] = current_distance
                # Use symmetry to avoid duplicate calculations
                new_metric_matrix[j, i] = current_distance

        self.metric = new_metric_matrix
        return new_metric_matrix
    
    def calculate_metric(self, s1, s2):
        return self.metric[s1, s2]

    def generate_mdp_visualization_data(self, action_colors = []):
        nodes = []
        edges = []
        edge_types = {}

        # Generate nodes
        for state in self.states:
            nodes.append({
                "id": str(state),
                "label": str(state)
            })

        # Generate edges and edge types for actions, transition probabilities:
        for state in self.states:
            for action in self.actions:
                if action in self.transition_function[state]:
                    for next_state in self.transition_function[state][action]:
                        probability = self.transition_function[state][action].get(next_state, 0.0)
                        reward = self.reward_function[state][action]

                        if probability > 0.0:
                            edges.append({
                                "from": str(state),
                                "to": str(next_state),
                                "label": f"{probability:.3f}",
                                "type": action
                            })

                            if action not in edge_types:
                                if action_colors:
                                    edge_types[action] = {
                                        "color": action_colors[action],
                                        "arrows": True
                                    }
                                else:
                                    edge_types[action] = {
                                        "color": "blue",
                                        "arrows": True
                                    }
                        
                            
                else:
                    # Handle missing action for state
                    edges.append({
                        "from": str(state),
                        "to": str(state),
                        "label": "N/A",
                        "type": action
                    })

                    if action not in edge_types:
                        edge_types[action] = {
                            "color": "gray",
                            "arrows": False
                        }
        # Generate edges for metric:
        for state_1 in self.states:
            for state_2 in self.states:
                if state_1 != state_2:
                    dummy_value = self.calculate_metric(state_1, state_2)
                    edges.append({
                        'from': state_1,
                        'to': state_2,
                        'label': f"{dummy_value:.5f}",
                        'type': "Metric",
                        'arrows': ''
                    })         
        # Add metric
        edge_types["Metric"]= {
            "color": "black",
            "arrows": False
        } 

        return nodes, edges, edge_types

    def generate_mdp_html(self, template_file_path, output_file_path, action_colors = []):
        """
        Generate an HTML file for visualizing the MDP as a network.

        Args:
            template_file_path: The path to the template HTML file.
            output_file_path: The path where the output HTML file should be written.
        """
        # Generate the data for visualization
        nodes, edges, edge_types = self.generate_mdp_visualization_data(action_colors)

        # Convert the nodes, edges, and edge types to JSON
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        edge_types_json = json.dumps(edge_types)

        # Load the template file
        with open(template_file_path, "r") as template_file:
            template_html = template_file.read()

        # Replace the placeholders in the template with the JSON data
        mdp_html = template_html.replace("{{nodes_json}}", nodes_json) \
                                .replace("{{edges_json}}", edges_json) \
                                .replace("{{edge_types_json}}", edge_types_json)

        # Write the resulting HTML to the output file
        with open(output_file_path, "w") as output_file:
            output_file.write(mdp_html)

    def get_probability_vector(self, state, action):
        # Initialize a vector of zeros with length equal to the number of states
        probabilities = [0.0]*len(self.states)

        # Get the transition probabilities for the given state and action
        transitions = self.transition_function[state][action]

        # For each state that can be reached from the current state, update its probability
        for next_state, prob in transitions.items():
            probabilities[next_state] = prob

        return probabilities

def get_incorrect_states(num_rows, num_cols, state, action):
    row = state // num_cols
    col = state % num_cols
    incorrect_states = [state]  # add the current state as it is always a possible incorrect state
    
    # Define the moves corresponding to each action
    moves = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    
    # Diagonal moves for each action
    diagonal_moves = {
        "up": [(0, -1), (0, 1)],
        "down": [(0, -1), (0, 1)],
        "left": [(-1, 0), (1, 0)],
        "right": [(-1, 0), (1, 0)]
    }

    # get the move for the action
    move = moves[action]
    
    # get the new row and column
    new_row = row + move[0]
    new_col = col + move[1]
    
    if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
        # If the move is valid, add the diagonals to the incorrect states
        for d_move in diagonal_moves[action]:
            d_row = new_row + d_move[0]
            d_col = new_col + d_move[1]

            if 0 <= d_row < num_rows and 0 <= d_col < num_cols:
                incorrect_states.append(d_row * num_cols + d_col)
    
    return incorrect_states

def get_correct_state(num_rows, num_cols, current_state, action):
    row, col = divmod(current_state, num_cols)

    if action == "up" and row > 0:
        return (row - 1) * num_cols + col
    elif action == "down" and row < num_rows - 1:
        return (row + 1) * num_cols + col
    elif action == "left" and col > 0:
        return row * num_cols + (col - 1)
    elif action == "right" and col < num_cols - 1:
        return row * num_cols + (col + 1)
    else:
        return None  # No correct state for the given action
def generate_6_bisimular_mdp():
    num_states = 6
    states = list(range(num_states))
    actions = ["A"]
    transition_function = {
    0: {"A": {0: 0.5, 1: 0.5}},
    1: {"A": {0: 0.5, 2: 0.5}},
    2: {"A": {1: 0.5, 2: 0.5}},
    3: {"A": {3: 0.5, 4: 0.5}},
    4: {"A": {3: 0.5, 5: 0.5}},
    5: {"A": {4: 0.5, 5: 0.5}},
    }
    reward_function = {s: {a: {0}} for s in states for a in actions}

    

    discount_factor = 0.9
    mdp = MarkovDecisionProcess(states, actions, transition_function, reward_function, discount_factor)
    action_colors = {
        "A": "red"
    }
    return mdp, action_colors


def generate_grid_mdp(num_rows, num_cols, correct_prob, discount_factor, good_reward, bad_reward):
    num_states = num_rows * num_cols
    states = list(range(num_states))
    actions = ["up", "down", "left", "right"]

    action_colors = {
        "up": "yellow",
        "down": "brown",
        "left": "green",
        "right": "red",
    }

    transition_function = {state: {action: {} for action in actions} for state in states}
    reward_function = {state: {action: {} for action in actions} for state in states}

    for state in states:
        for action in actions:
            correct_state = get_correct_state(num_rows, num_cols, state, action)
            incorrect_states = get_incorrect_states(num_rows, num_cols, state, action)

            if correct_state is not None and correct_state in incorrect_states:
                incorrect_states.remove(correct_state)

            remaining_prob = 1 - correct_prob if correct_state is not None else 1
            for incorrect_state in incorrect_states:
                transition_function[state][action][incorrect_state] = remaining_prob / len(incorrect_states)
                reward_function[state][action][incorrect_state] = bad_reward

            if correct_state is not None:
                transition_function[state][action][correct_state] = correct_prob
                reward_function[state][action][correct_state] = good_reward

    mdp = MarkovDecisionProcess(states, actions, transition_function, reward_function, discount_factor)
    return mdp, action_colors

def main():
   
    num_states = 4
    states = list(range(num_states))
    actions = ["A"]
    transition_function = {
    0: {"A": {0: 1}},
    1: {"A": {1: 1}},
    2: {"A": {0: 0.1, 1: 0.9}},
    3: {"A": {0: 0.9, 1: 0.1}},
    }
    reward_function = {s: {a: {0}} for s in states for a in actions}
    
    discount_factor = 0.9
    mdp = MarkovDecisionProcess(states, actions, transition_function, reward_function, discount_factor)
    action_colors = {
        "A": "red"
    }

    html_file_name = "intersection.html"
    
    dir = "."
    # Generate HTML
    mdp.generate_mdp_html(f"{dir}/template_labeled_graph.html", html_file_name, action_colors)
    os.system('firefox ' + f"{dir}/{html_file_name}")
    
    exit()


    #mdp, action_colors = generate_grid_mdp(3,3,0.7,0.9,1,-1)
    mdp, action_colors = generate_6_bisimular_mdp()

    for i in range(2):
        mdp.iterate_bisimulation_matrix()

    print(mdp.metric)

    dir = "."
    
    html_file_name = "6bisim.html"
    #html_file_name = "grid.html"
    
    # Generate HTML
    mdp.generate_mdp_html(f"{dir}/template_labeled_graph.html", html_file_name, action_colors)
    
 


    # Open the HTML file in Firefox
    os.system('firefox ' + f"{dir}/{html_file_name}")
main()