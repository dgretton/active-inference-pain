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
        self.regions = None  # Will be set by simulation
    
    def update(self):
        self.warn = self.warn if np.random.rand() > self.warn_reset_prob else False

class WormSimulation:
    def __init__(self, sim_width, sim_height):
        # Initialize simulation parameters
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.worm_state = WormState()
        
        # Worm properties
        self.worm_pos = np.array([sim_width // 2, 0], dtype=float)
        self.worm_radius = 6
        self.worm_length = 40
        self.worm_poss = [self.worm_pos.copy() for _ in range(self.worm_length)]
        self.worm_vels = np.zeros((self.worm_length, 2))

        # Regions
        self.regions = [
            {"rect": pygame.Rect(0, 485, sim_width, 30), "color": (255, 165, 0), "effect": "warning"},
            {"rect": pygame.Rect(0, 550, sim_width, 50), "color": (255, 0, 0), "effect": "nociception"},
        ]
        self.worm_state.regions = self.regions  # Pass to worm state for visualization

        # Initialize the active inference agent
        self.setup_agent()

    def setup_agent(self):
        self.num_obs_noci = 2
        self.num_obs_warn = 2
        self.num_states = 2
        self.num_controls = 2

        # Define A matrix
        self.A_array = utils.obj_array_zeros([(self.num_obs_noci, self.num_states), 
                                            (self.num_obs_warn, self.num_states)])
        self.A_array[0][:, 0] = [0.0, 1.0]
        self.A_array[0][:, 1] = [1.0, 0.0]
        self.A_array[1][:, 0] = [0.5, 0.5]
        self.A_array[1][:, 1] = [0.5, 0.5]

        # Define B matrix
        self.B_array = utils.obj_array_zeros([(self.num_states, self.num_states, self.num_controls)])
        self.B_array[0][:, :, 0] = np.eye(2)
        self.B_array[0][:, :, 1] = [[1, 1], [0, 0]]

        # Define C vector
        self.C_vector = utils.obj_array_zeros([(self.num_obs_noci,), (self.num_obs_warn,)])
        self.C_vector[0] = np.array([0.0, 1.0])
        self.C_vector[1] = np.array([0.5, 0.5])

        # Define E matrix
        self.E_matrix = np.array([0.8, 0.2])

        self.agent = Agent(A=self.A_array, B=self.B_array, C=self.C_vector, E=self.E_matrix)
        self.qs = utils.obj_array_uniform([(self.num_states,)])

    def update_physics(self, action):
        """Update worm physics"""
        speed = 2
        movement = self.worm_state.movement
        movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
        movement += np.random.uniform(-1, 1, 2)*.1

        if action == 1:
            movement += np.array([0, -2])
        else:
            movement += np.array([0, .01])

        movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
        movement *= speed

        self.worm_state.noci = False
        if action != 1:
            for region in self.regions:
                if region["rect"].collidepoint(self.worm_pos):
                    if region["effect"] == "warning":
                        self.worm_state.warn = True
                    elif region["effect"] == "nociception":
                        self.worm_state.noci = True

        new_pos = self.worm_pos + movement
        self.worm_state.movement = movement
        self.worm_state.update()

        desired_radius = self.worm_radius
        ideal_poss = [self.worm_poss[0]]
        for i in range(1, len(self.worm_poss) - 2):
            diff = self.worm_poss[i] - self.worm_poss[i-1] + np.array([0, 1e-6])
            diff += (self.worm_poss[i] - self.worm_poss[0]) * .01
            if i > 1:
                diff += (self.worm_poss[i] - self.worm_poss[i-2]) * 10
            ideal_poss.append(ideal_poss[-1] + diff/np.linalg.norm(diff) * desired_radius)
        
        for i in range(len(self.worm_poss) - 2):
            self.worm_poss[i] += (ideal_poss[i] - self.worm_poss[i]) * 0.03

        new_pos[0] = np.clip(new_pos[0], self.worm_radius, self.sim_width - self.worm_radius)
        new_pos[1] = np.clip(new_pos[1], self.worm_radius, self.sim_height - self.worm_radius)

        self.worm_poss.pop(0)
        self.worm_poss.append(new_pos)
        self.worm_pos = new_pos

    def get_observations(self):
        """Get current observations from worm state"""
        warn_observation = 1 if self.worm_state.warn else 0
        noci_observation = 0 if self.worm_state.noci else 1
        return (noci_observation, warn_observation)

    def update_A_matrix(self, observation, learning_rate=0.01):
        """Update A matrix based on current beliefs"""
        noci_observation, warn_observation = observation
        safe_state_value = self.qs[0][0]
        harmful_state_value = self.qs[0][1]
        
        one_hot_warn_vector = np.array([1.0, 0.001]) if warn_observation == 0 else np.array([0.0001, 1.0]) #.001 is for debuugging
        
        self.agent.A[1][:, 0] += one_hot_warn_vector * safe_state_value * learning_rate
        self.agent.A[1][:, 1] += one_hot_warn_vector * harmful_state_value * learning_rate

    def step(self):
        """Perform one simulation step"""
        # Get observations
        observation = self.get_observations()

        # Update agent's beliefs and get action
        self.qs = self.agent.infer_states(observation)
        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()[0]

        # Update A matrix
        self.update_A_matrix(observation)

        # Update physics
        self.update_physics(action)

        return self.worm_state, self.worm_pos, self.worm_poss, action, self.qs, self.agent.A

class WormVisualizer:
    def __init__(self, width, height, sim_width, sim_height):
        pygame.init()
        
        # Display setup
        self.width = width
        self.height = height
        self.sim_width = sim_width
        self.sim_height = sim_height
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.BLACK = (0, 0, 0)
        self.LIGHTGRAY = (200, 200, 200)
        self.ORANGE = (255, 165, 0)

        # Pygame setup
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Actively Enhanced Worm Simulation")
        self.sim_surface = pygame.Surface((sim_width, sim_height))
        self.config_surface = pygame.Surface((width - sim_width, height))
        self.font = pygame.font.Font(None, 30)
        self.clock = pygame.time.Clock()
        self.fps = 60

    def draw_regions(self, surface, regions):
        for region in regions:
            pygame.draw.rect(surface, region["color"], region["rect"], 2)
            text = self.font.render(region["effect"], True, region["color"])
            surface.blit(text, (region["rect"].centerx - text.get_width() // 2, 
                              region["rect"].centery - text.get_height() // 2))

    def draw_config_panel(self, surface):
        surface.fill(self.LIGHTGRAY)
        title = self.font.render("Configuration", True, self.BLACK)
        surface.blit(title, (10, 10))

    def draw_frame(self, sim_state):
        """Draw a single frame based on simulation state. Returns False if window closed."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Unpack simulation state
        worm_state, worm_pos, worm_poss, action, qs, A = sim_state
        
        # Clear surfaces
        self.sim_surface.fill(self.WHITE)
        self.config_surface.fill(self.WHITE)

        # Print debug info
        print(qs)
        print(A)

        # Draw indicators
        warning_indicator = self.font.render(
            "Warning: ON" if worm_state.warn else "Warning: OFF", 
            True, self.RED)
        self.sim_surface.blit(warning_indicator, (10, 10))
        
        action_indicator = self.font.render(
            "Action: Retreat" if action == 1 else "Action: Stay", 
            True, self.GREEN)
        self.sim_surface.blit(action_indicator, (10, 40))

        # Draw regions
        self.draw_regions(self.sim_surface, worm_state.regions)

        # Draw worm
        pygame.draw.circle(self.sim_surface, self.RED, worm_pos.astype(int), 6)
        for pos in worm_poss:
            pygame.draw.circle(self.sim_surface, self.RED, pos.astype(int), 6)

        self.draw_config_panel(self.config_surface)

        # Update display
        self.screen.blit(self.sim_surface, (0, 0))
        self.screen.blit(self.config_surface, (self.sim_width, 0))
        pygame.display.flip()
        self.clock.tick(self.fps)
        
        return True

    def cleanup(self):
        pygame.quit()

def main():
    # Constants
    WIDTH, HEIGHT = 1000, 600
    SIM_WIDTH, SIM_HEIGHT = 400, 600

    # Initialize simulation and visualization
    sim = WormSimulation(SIM_WIDTH, SIM_HEIGHT)
    vis = WormVisualizer(WIDTH, HEIGHT, SIM_WIDTH, SIM_HEIGHT)

    running = True
    while running:
        # Update simulation
        sim_state = sim.step()
        
        # Update visualization (returns False if window closed)
        running = vis.draw_frame(sim_state)

    vis.cleanup()
    sys.exit()

if __name__ == "__main__":
    main()