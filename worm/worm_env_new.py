# model.py
import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class WormState:
    movement: np.ndarray 
    warn: bool
    noci: bool
    warn_reset_prob: float
    position: np.ndarray
    radius: float
    length: int
    positions: List[np.ndarray]
    velocities: np.ndarray

    @classmethod
    def create_initial(cls, sim_width: int, sim_height: int):
        position = np.array([sim_width // 2, 0], dtype=float)
        movement = np.array([0, 0], dtype=float)
        radius = 6
        length = 40
        positions = [position.copy() for _ in range(length)]
        velocities = np.zeros((length, 2))
        
        return cls(
            movement=movement,
            warn=False,
            noci=False,
            warn_reset_prob=0.01,
            position=position,
            radius=radius,
            length=length, 
            positions=positions,
            velocities=velocities
        )

    def update_warn(self):
        self.warn = self.warn if np.random.rand() > self.warn_reset_prob else False

class Region:
    def __init__(self, rect: Tuple[int, int, int, int], effect: str):
        self.rect = rect  # (x, y, width, height)
        self.effect = effect

    def contains_point(self, point: np.ndarray) -> bool:
        x, y = point
        rx, ry, rw, rh = self.rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

class ActiveInferenceAgent:
    def __init__(self):
        # Initialize the agent parameters
        self.num_obs_noci = 2
        self.num_obs_warn = 2
        self.num_states = 2
        self.num_controls = 2

        self.setup_matrices()
        self.agent = Agent(A=self.A_array, B=self.B_array, C=self.C_vector, E=self.E_matrix)
        self.qs = utils.obj_array_uniform([(self.num_states,)])

    def setup_matrices(self):
        # A matrix setup
        self.A_array = utils.obj_array_zeros([(self.num_obs_noci, self.num_states), 
                                            (self.num_obs_warn, self.num_states)])
        
        self.A_array[0][:, 0] = [0.0, 1.0]  
        self.A_array[0][:, 1] = [1.0, 0.0]  
        self.A_array[1][:, 0] = [0.0, 1.0]  
        self.A_array[1][:, 1] = [1.0, 0.0]  

        # B matrix setup
        self.B_array = utils.obj_array_zeros([(self.num_states, self.num_states, self.num_controls)])
        self.B_array[0][:, :, 0] = np.eye(2)
        self.B_array[0][:, :, 1] = [[1, 1],
                                   [0, 0]]

        # C vector setup  
        self.C_vector = utils.obj_array_zeros([(self.num_obs_noci,), (self.num_obs_warn,)])
        self.C_vector[0] = np.array([0.0, 1.0])
        self.C_vector[1] = np.array([0.5, 0.5])

        # E matrix setup
        self.E_matrix = np.array([0.8, 0.2])

    def update_matrices(self, A_array=None, B_array=None, C_vector=None, E_matrix=None):
        """Update any of the matrices with new values"""
        if A_array is not None:
            self.A_array = A_array
        if B_array is not None:
            self.B_array = B_array
        if C_vector is not None:
            self.C_vector = C_vector
        if E_matrix is not None:
            self.E_matrix = E_matrix
            
        # Recreate agent with updated matrices
        self.agent = Agent(A=self.A_array, B=self.B_array, C=self.C_vector, E=self.E_matrix)

    def get_action(self, observation: Tuple[int, int]) -> int:
        """Get next action based on current observation"""
        self.qs = self.agent.infer_states(observation)
        _, _ = self.agent.infer_policies()
        return self.agent.sample_action()[0]

class WormSimulation:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.worm_state = WormState.create_initial(width, height)
        self.agent = ActiveInferenceAgent()
        
        # Define regions
        self.regions = [
            Region((0, 275, width, 200), "warning"),
            Region((0, 550, width, 50), "nociception"),
        ]

    def update_physics(self, action: int) -> None:
        """Update worm physics based on action"""
        speed = 2
        movement = self.worm_state.movement
        movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
        movement += np.random.uniform(-1, 1, 2)*.1

        if action == 1:  # retreat
            movement += np.array([0, -2])
        else:  # stay
            movement += np.array([0, .01])

        movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
        movement *= speed

        self.worm_state.noci = False
        
        if action != 1:
            for region in self.regions:
                if region.contains_point(self.worm_state.position):
                    if region.effect == "warning":
                        self.worm_state.warn = True
                    elif region.effect == "nociception":
                        self.worm_state.noci = True

        new_pos = self.worm_state.position + movement
        self.worm_state.movement = movement
        self.worm_state.update_warn()

        # Update worm body positions
        desired_radius = self.worm_state.radius
        positions = self.worm_state.positions
        ideal_poss = [positions[0]]
        
        for i in range(1, len(positions) - 2):
            diff = positions[i] - positions[i-1] + np.array([0, 1e-6])
            diff += (positions[i] - positions[0]) * .01
            if i > 1:
                diff += (positions[i] - positions[i-2]) * 10
            ideal_poss.append(ideal_poss[-1] + diff/np.linalg.norm(diff) * desired_radius)
            
        for i in range(len(positions) - 2):
            positions[i] += (ideal_poss[i] - positions[i]) * 0.03

        # Keep within boundaries
        new_pos[0] = np.clip(new_pos[0], self.worm_state.radius, self.width - self.worm_state.radius)
        new_pos[1] = np.clip(new_pos[1], self.worm_state.radius, self.height - self.worm_state.radius)

        positions.pop(0)
        positions.append(new_pos)
        self.worm_state.position = new_pos

    def get_observation(self) -> Tuple[int, int]:
        """Get current observation tuple (noci, warn)"""
        noci_observation = 0 if self.worm_state.noci else 1
        warn_observation = 1 if self.worm_state.warn else 0
        return (noci_observation, warn_observation)

    def step(self) -> Tuple[WormState, int]:
        """Perform one simulation step"""
        observation = self.get_observation()
        action = self.agent.get_action(observation)
        self.update_physics(action)
        return self.worm_state, action

# visualization.py
import pygame
from typing import Tuple, List

class WormVisualizer:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width 
        self.height = height
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.BLACK = (0, 0, 0)
        
        # Setup display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Worm Simulation")
        self.sim_surface = pygame.Surface((width, height))
        self.font = pygame.font.Font(None, 30)
        self.clock = pygame.time.Clock()

    def draw_frame(self, worm_state: WormState, action: int, regions: List[Region]) -> bool:
        """Draw a single frame. Returns False if window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.sim_surface.fill(self.WHITE)

        # Draw regions
        for region in regions:
            pygame.draw.rect(self.sim_surface, 
                           self.ORANGE if region.effect == "warning" else self.RED,
                           region.rect, 2)

        # Draw worm
        for pos in worm_state.positions:
            pygame.draw.circle(self.sim_surface, self.RED, pos.astype(int), worm_state.radius)

        # Draw indicators
        warning_text = self.font.render(
            "Warning: ON" if worm_state.warn else "Warning: OFF", 
            True, self.RED)
        action_text = self.font.render(
            "Action: Retreat" if action == 1 else "Action: Stay",
            True, self.GREEN)
            
        self.sim_surface.blit(warning_text, (10, 10))
        self.sim_surface.blit(action_text, (10, 40))

        self.screen.blit(self.sim_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)
        
        return True

    def cleanup(self):
        pygame.quit()

# main.py
def main():
    # Initialize simulation
    SIM_WIDTH, SIM_HEIGHT = 400, 600
    sim = WormSimulation(SIM_WIDTH, SIM_HEIGHT)
    vis = WormVisualizer(SIM_WIDTH, SIM_HEIGHT)

    running = True
    while running:
        # Update simulation
        worm_state, action = sim.step()
        
        # Visualize (returns False if window closed)
        running = vis.draw_frame(worm_state, action, sim.regions)

    vis.cleanup()

if __name__ == "__main__":
    main()