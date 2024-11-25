import pygame
import sys
import numpy as np

import pymdp
from pymdp import utils
from pymdp.agent import Agent
from pymdp.maths import softmax

class WormState:
    def __init__(self):
        self.movement = np.array([0, 0], dtype=float)
        self.warn = False
        self.noci = False
        self.warn_reset_prob = 0.01
    
    def update(self):
        self.warn = self.warn if np.random.rand() > self.warn_reset_prob else False

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Actively Enhanced Worm Simulation")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
LIGHTGRAY = (200, 200, 200)
ORANGE = (255, 165, 0)

# Fonts
font = pygame.font.Font(None, 30)

# Simulation properties
SIM_WIDTH, SIM_HEIGHT = 400, 600
sim_surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))

# Worm properties
worm_pos = np.array([SIM_WIDTH // 2, 0], dtype=float) # bottom center
worm_radius = 6
worm_length = 40
worm_poss = [worm_pos.copy() for _ in range(worm_length)] #worm positions
worm_vels = np.zeros((worm_length, 2))

# Regions
regions = [
    {"rect": pygame.Rect(0, 485, SIM_WIDTH, 30), "color": ORANGE, "effect": "warning"},
    {"rect": pygame.Rect(0, 550, SIM_WIDTH, 50), "color": RED, "effect": "nociception"},
]

# Configuration panel
CONFIG_WIDTH = WIDTH - SIM_WIDTH
config_surface = pygame.Surface((CONFIG_WIDTH, HEIGHT))

# Simulation properties
clock = pygame.time.Clock()
FPS = 60

def update_worm_state(worm_state, worm_pos, action):
    speed = 2

    #initialize to difference between previous position and current position
    movement = worm_state.movement
    #movement = (worm_pos[-1] - worm_poss[-2])
    movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
    movement += np.random.uniform(-1, 1, 2)*.1

    if action == 1:  # 'retreat' action
        # Move toward the top of the screen
        movement += np.array([0, -2])  # up direction
    else:  # action == 0, 'stay'
        movement += np.array([0, .01]) # bias downward

    movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
    movement *= speed

    worm_state.noci = False
    if action != 1:
        # Apply region effects
        for region in regions:
            if region["rect"].collidepoint(worm_pos):
                if region["effect"] == "slow": #should remove these soon
                    movement *= 0.5
                elif region["effect"] == "fast":
                    movement *= 2
                elif region["effect"] == "random":
                    movement += np.random.uniform(-10, 10, 2)
                elif region["effect"] == "warning":
                    worm_state.warn = True # this will last for some time after setting it to true, with the time scale set by worm_state.warn_reset_prob
                elif region["effect"] == "nociception":
                    worm_state.noci = True

    new_pos = worm_pos + movement
    worm_state.movement = movement
    worm_state.update()

    desired_radius = worm_radius
    ideal_poss = [worm_poss[0]]
    for i in range(1, len(worm_poss) - 2):
        diff = worm_poss[i] - worm_poss[i-1] + np.array([0, 1e-6]) # tie-break downward
        diff += (worm_poss[i] - worm_poss[0]) * .01
        if i > 1:
            diff += (worm_poss[i] - worm_poss[i-2]) * 10
        ideal_poss.append(ideal_poss[-1] + diff/np.linalg.norm(diff) * desired_radius)
    for i in range(len(worm_poss) - 2):
        # move a little of the way to the ideal position
        worm_poss[i] += (ideal_poss[i] - worm_poss[i]) * 0.03

    # Keep the worm within the simulation boundaries
    new_pos[0] = np.clip(new_pos[0], worm_radius, SIM_WIDTH - worm_radius)
    new_pos[1] = np.clip(new_pos[1], worm_radius, SIM_HEIGHT - worm_radius)

    worm_poss.pop(0)
    worm_poss.append(new_pos)

    return new_pos

def draw_worm(surface):
    pygame.draw.circle(surface, RED, worm_pos.astype(int), worm_radius)
    for pos in worm_poss:
        pygame.draw.circle(surface, RED, pos.astype(int), worm_radius)

def draw_regions(surface):
    for region in regions:
        pygame.draw.rect(surface, region["color"], region["rect"], 2)
        text = font.render(region["effect"], True, region["color"])
        surface.blit(text, (region["rect"].centerx - text.get_width() // 2, 
                            region["rect"].centery - text.get_height() // 2))

def draw_config_panel(surface):
    surface.fill(LIGHTGRAY)
    title = font.render("Configuration", True, BLACK)
    surface.blit(title, (10, 10))
    
    # Add more configuration options here


def main():
    global worm_pos
    worm_state = WormState()

    # Initialize the agent
    num_obs_noci = 2       # Observation modality noci dimensions
    num_obs_warn = 2       # Observation modality warning dimensions
    num_states = 2    # Hidden state factor dimensions
    num_controls = 2  # Control state factor dimensions

    # Define the A matrix (likelihood) maps observations to hidden states
    A_array = utils.obj_array_zeros([(num_obs_noci, num_states), (num_obs_warn, num_states)])
    # A[observation][hidden_state]
    # Observation 0: 'nociception', Observation 1: 'no-nociception'
    # Observation 1: 'warning', Observation 2: 'no-warning'
    # Hidden State 0: 'safe', Hidden State 1: 'harmful'

    A_array[0][:, 0] = [0.0, 1.0]  # no noci -> 'safe' state
    A_array[0][:, 1] = [1.0, 0.0]  # noci -> 'harmful' state

    A_array[1][:, 0] = [0.5, 0.5]  # no warning -> 'safe' state
    A_array[1][:, 1] = [0.5, 0.5]  # warning -> 'harmful' state

    # Define the B matrix (transitions)
    B_array = utils.obj_array_zeros([(num_states, num_states, num_controls)])
    # B[next_state][current_state][action]
    # Action 0: 'stay', Action 1: 'retreat'
    # 'stay' action: state remains the same
    B_array[0][:, :, 0] = np.eye(2)
    # 'retreat' action: moves to 'safe' state
    B_array[0][:, :, 1] = [[1, 1], # NOTE: Dana randomly did this at the end to make the B array normalized because we had a normalization error when instantiating the agent using the original B matrix, which had these 1s and 0s transposed.
                           [0, 0]]

    # Define the C vector (preferences)
    C_vector = utils.obj_array_zeros([(num_obs_noci,), (num_obs_warn,)])
    # Preference for 'no-nociception' over 'nociception'
    C_vector[0] = np.array([0.0, 1.0])
    # Preference for 'no-warning' over 'warning' (none)
    C_vector[1] = np.array([0.5, 0.5])

    # E matrix: prior over policies. We prefer not to retreat.
    E_matrix = np.array([0.8, 0.2])

    # Initialize the agent
    my_agent = Agent(A=A_array, B=B_array, C=C_vector, E=E_matrix)

    # Initialize prior beliefs
    qs = utils.obj_array_uniform([(num_states,)])  # Uniform prior

    running = True



    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the surfaces
        sim_surface.fill(WHITE)
        config_surface.fill(WHITE)

        # Determine observations based on worm's position
        if worm_state.warn:  # whether we were in the warning zone in some previous time step not so long ago
            warn_observation = 1  # 'warning'
        else:
            warn_observation = 0  # 'no-warning'

        # Update worm_state.noci based on the worm's current position
        worm_state.noci = False  # Reset nociception flag
        for region in regions:
            if region["effect"] == "nociception" and region["rect"].collidepoint(worm_pos):
                worm_state.noci = True  # Set nociception flag if worm is in the nociceptive region

        # Now, determine noci_observation based on updated state
        if worm_state.noci:
            noci_observation = 0  # 'nociception'
        else:
            noci_observation = 1  # 'no-nociception'

        observation = (noci_observation, warn_observation)

        # Update agent's belief over hidden states
        qs = my_agent.infer_states(observation)

        # Infer policies and sample action
        q_pi, efe = my_agent.infer_policies()
        action = my_agent.sample_action()[0]  # action is a list of length 1 (there is only one control factor, retreat/stay)

        # Very slowly and explicitly update the A matrix based on the current belief over hidden states using small words and no scary maths
        safe_state_value = qs[0][0]
        harmful_state_value = qs[0][1]
        one_hot_noci_vector_for_this_iteration = np.array([1.0, 0.0]).T if noci_observation == 0 else np.array([0.0, 1.0]).T
        one_hot_warn_vector_for_this_iteration = np.array([1.0, 0.0]).T if warn_observation == 0 else np.array([0.0, 1.0]).T
        my_agent.A[0][0, :] += one_hot_noci_vector_for_this_iteration * safe_state_value
        my_agent.A[0][1, :] += one_hot_noci_vector_for_this_iteration * harmful_state_value
        my_agent.A[1][0, :] += one_hot_warn_vector_for_this_iteration * safe_state_value
        my_agent.A[1][1, :] += one_hot_warn_vector_for_this_iteration * harmful_state_value
        print(qs)
        print(my_agent.A)

        # Update worm state based on action
        worm_pos = update_worm_state(worm_state, worm_pos, action)

        # Draw warning indicator
        warning_indicator = font.render("Warning: ON" if worm_state.warn else "Warning: OFF", True, RED)
        sim_surface.blit(warning_indicator, (10, 10))

        # same for action taken
        action_indicator = font.render("Action: Retreat" if action == 1 else "Action: Stay", True, GREEN)
        sim_surface.blit(action_indicator, (10, 40))

        # Draw regions
        draw_regions(sim_surface)

        # Draw worm
        draw_worm(sim_surface)

        # Draw configuration panel
        draw_config_panel(config_surface)

        # Blit surfaces to the main screen
        screen.blit(sim_surface, (0, 0))
        screen.blit(config_surface, (SIM_WIDTH, 0))

        # Update display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()