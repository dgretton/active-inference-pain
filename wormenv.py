import pygame
import sys
import numpy as np

import pymdp
from pymdp import utils
from pymdp.agent import Agent

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

# Fonts
font = pygame.font.Font(None, 30)

# Simulation properties
SIM_WIDTH, SIM_HEIGHT = 400, 600
sim_surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))

# Worm properties
worm_pos = np.array([SIM_WIDTH // 2, SIM_HEIGHT], dtype=float) # bottom center
worm_radius = 6
worm_length = 40
worm_poss = [worm_pos.copy() for _ in range(worm_length)]
worm_vels = np.zeros((worm_length, 2))

# Regions
regions = [
    {"rect": pygame.Rect(50, 50, 100, 100), "color": BLUE, "effect": "slow"},
    {"rect": pygame.Rect(200, 200, 150, 150), "color": GREEN, "effect": "fast"},
    {"rect": pygame.Rect(200, 400, 120, 120), "color": YELLOW, "effect": "random"}
]

# Configuration panel
CONFIG_WIDTH = WIDTH - SIM_WIDTH
config_surface = pygame.Surface((CONFIG_WIDTH, HEIGHT))

# Simulation properties
clock = pygame.time.Clock()
FPS = 60

def update_worm_state(worm_pos, action):
    speed = 5

    #initialize to difference between previous position and current position
    movement = (worm_pos[-1] - worm_poss[-2])
    movement /= np.linalg.norm(movement)
    movement += np.random.uniform(-1, 1, 2)*.1

    if action == 1:  # 'retreat' action
        # Move toward the top of the screen
        movement += np.array([-1, -2])  # up direction
    else:  # action == 0, 'stay'
        movement += np.array([-1, .1]) # bias downward

    movement /= np.linalg.norm(movement)
    movement *= speed

    if action != 1:
        # Apply region effects
        for region in regions:
            if region["rect"].collidepoint(worm_pos):
                if region["effect"] == "slow":
                    movement *= 0.5
                elif region["effect"] == "fast":
                    movement *= 2
                elif region["effect"] == "random":
                    movement += np.random.uniform(-10, 10, 2)

    new_pos = worm_pos + movement

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

    # Initialize the agent
    num_obs = 2       # Observation modality dimensions
    num_states = 2    # Hidden state factor dimensions
    num_controls = 2  # Control state factor dimensions

    # Define the A matrix (likelihood)
    A_array = utils.obj_array_zeros([(num_obs, num_states)])
    # A[observation][hidden_state]
    # Observation 0: 'nociception', Observation 1: 'no-nociception'
    # Hidden State 0: 'safe', Hidden State 1: 'harmful'

    A_array[0][:, 0] = [0.0, 1.0]  # 'safe' state
    A_array[0][:, 1] = [1.0, 0.0]  # 'harmful' state

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
    C_vector = utils.obj_array_zeros([(num_obs,)])
    # Preference for 'no-nociception' over 'nociception'
    C_vector[0] = np.array([0.0, 1.0])

    # Initialize the agent
    my_agent = Agent(A=A_array, B=B_array, C=C_vector)

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

        # Determine observation based on worm's position
        if worm_pos[1] > SIM_HEIGHT - 100:
            observation = [0]  # 'nociception'
        else:
            observation = [1]  # 'no-nociception'

        # Update agent's belief over hidden states
        qs = my_agent.infer_states(observation)

        # Infer policies and sample action
        q_pi, efe = my_agent.infer_policies()
        action = my_agent.sample_action()[0]  # action is a list of length 1

        # Update worm state based on action
        worm_pos = update_worm_state(worm_pos, action)

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