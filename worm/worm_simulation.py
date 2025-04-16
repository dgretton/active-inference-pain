# model.py
import numpy as np
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Tuple
from pymdp import utils
# from worm_simulation_dynamic_perception_and_policy_selection_chatgpt import ActiveInferenceAgent
from agent_worm import ActiveInferenceWormAgent, SimpleHomeostaticAgent

@dataclass
class SimulationConfig:
    """Configuration parameters for simulation"""
    sim_width: float = 400.0
    sim_height: float = 600.0
    worm_radius: float = 6.0
    worm_length: int = 40
    warn_reset_prob: float = 1.0
    speed: float = 2.0
    learning_rate: float = .05
    
    # Region definitions
    warning_height: float = 525.0
    warning_thickness: float = 75.0
    noci_height: float = 550.0
    noci_thickness: float = 50.0
    # warning_height: float = 550.0  # Change from 485.0
    # warning_thickness: float = 50.0  # Change from 145.0
    # noci_height: float = 550.0  # Already correct
    # noci_thickness: float = 50.0  # Already correct

@dataclass
class WormPhysState:
    """Physical state of the worm"""
    position: np.ndarray  # Current head position
    positions: List[np.ndarray]  # List of segment positions
    movement: np.ndarray  # Current movement vector
    warn: bool = False  # Warning signal state
    noci: bool = False  # Nociception state

    @classmethod
    def initialize(cls, config: SimulationConfig):
        """Create initial worm state"""
        pos = np.array([config.sim_width / 2, 0.0])
        positions = [pos.copy() for _ in range(config.worm_length)]
        return cls(
            position=pos,
            positions=positions,
            movement=np.zeros(2),
            warn=False,
            noci=False
        )



class WormSimulation:
    """Main simulation class"""
    
    def __init__(self, config: SimulationConfig, agent_type:ActiveInferenceWormAgent, A_matrix=None):
        self.config = config
        self.phys_state = WormPhysState.initialize(config)
        self.A_matrix = A_matrix
        self.agent = agent_type(self.A_matrix)

    def update_physics(self, action: int) -> None:
        """Update worm physics"""
        # Update movement vector
        movement = self.phys_state.movement
        movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
        movement += np.random.uniform(-1, 1, 2) * 0.1

        if action == 1:  # 'retreat' action
            # Move toward the top of the screen
            movement += np.array([0, -.05])  # up direction
        else:  # action == 0, 'stay'
            movement += np.array([0, .05]) # bias downward

        movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
        movement *= self.config.speed

        # Update position
        new_pos = self.phys_state.position + movement
        self.phys_state.movement = movement
        
        # Reset warning with probability
        if np.random.rand() < self.config.warn_reset_prob:
            self.phys_state.warn = False

        # Update segment positions
        positions = self.phys_state.positions
        ideal_positions = [positions[0]]
        
        for i in range(1, len(positions) - 2):
            diff = positions[i] - positions[i-1] + np.array([0, 1e-6])
            diff += (positions[i] - positions[0]) * 0.01
            if i > 1:
                diff += (positions[i] - positions[i-2]) * 10
            ideal_positions.append(
                ideal_positions[-1] + diff/np.linalg.norm(diff) * self.config.worm_radius
            )
        
        for i in range(len(positions) - 2):
            positions[i] += (ideal_positions[i] - positions[i]) * 0.03

        # Enforce boundaries
        new_pos[0] = np.clip(new_pos[0], self.config.worm_radius, 
                            self.config.sim_width - self.config.worm_radius)
        new_pos[1] = np.clip(new_pos[1], self.config.worm_radius, 
                            self.config.sim_height - self.config.worm_radius)

        positions.pop(0)
        positions.append(new_pos)
        self.phys_state.position = new_pos

        # Check region collisions
        self.phys_state.noci = False
        y = self.phys_state.position[1]
        if self.config.warning_height <= y <= (self.config.warning_height + self.config.warning_thickness):
            self.phys_state.warn = True
        if self.config.noci_height <= y <= (self.config.noci_height + self.config.noci_thickness):
            self.phys_state.noci = True

    def get_observations(self) -> Tuple[int, int]:
        """Get current observations from worm state"""
        warn_observation = 0 if self.phys_state.warn else 1
        noci_observation = 0 if self.phys_state.noci else 1
        return (noci_observation, warn_observation)

    def step(self) -> Tuple[WormPhysState, np.ndarray, int]:
        """Perform one simulation step"""
        observation = self.get_observations()
        action, qs = self.agent.infer(observation)
        self.update_physics(action)
        return self.phys_state, qs, action

def run_simulation(config: SimulationConfig, num_steps: int) -> List[Tuple[WormPhysState, np.ndarray, int]]:
    """Run a single simulation for specified number of steps"""
    sim = WormSimulation(config, agent_type=SimpleHomeostaticAgent)
    history = []
    
    for _ in range(num_steps):
        phys_state, qs, action = sim.step()
        history.append((phys_state, qs, action))
        
    return history

def run_parallel_simulations(configs: List[SimulationConfig], 
                           num_steps: int, 
                           num_processes: int = None) -> List[List[Tuple[WormPhysState, np.ndarray, int]]]:
    """Run multiple simulations in parallel"""
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(run_simulation, 
                             [(config, num_steps) for config in configs])
    return results

class WormVisualizer:
    """Optional visualization component using Pygame"""
    def __init__(self, config: SimulationConfig, width: int = 1000, height: int = 600):
        import pygame
        self.pygame = pygame
        self.pygame.init()
        
        # Display setup
        self.width = width
        self.height = height
        self.sim_width = int(config.sim_width)
        self.sim_height = int(config.sim_height)
        
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
        self.sim_surface = pygame.Surface((self.sim_width, self.sim_height))
        self.config_surface = pygame.Surface((width - self.sim_width, height))
        self.font = pygame.font.Font(None, 30)
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Store config for drawing regions
        self.config = config

    def draw_regions(self):
        """Draw warning and nociception regions"""
        # Warning region
        warning_rect = self.pygame.Rect(
            0, 
            self.config.warning_height,
            self.sim_width, 
            self.config.warning_thickness
        )
        self.pygame.draw.rect(self.sim_surface, self.ORANGE, warning_rect, 2)
        warning_text = self.font.render("warning", True, self.ORANGE)
        self.sim_surface.blit(warning_text, (
            warning_rect.centerx - warning_text.get_width() // 2,
            warning_rect.centery - warning_text.get_height() // 2
        ))
        
        # Nociception region
        noci_rect = self.pygame.Rect(
            0,
            self.config.noci_height,
            self.sim_width,
            self.config.noci_thickness
        )
        self.pygame.draw.rect(self.sim_surface, self.RED, noci_rect, 2)
        noci_text = self.font.render("nociception", True, self.RED)
        self.sim_surface.blit(noci_text, (
            noci_rect.centerx - noci_text.get_width() // 2,
            noci_rect.centery - noci_text.get_height() // 2
        ))

    def draw_frame(self, state: WormPhysState, qs: np.ndarray, action: int) -> bool:
        """Draw a single frame. Returns False if window closed."""
        # Handle pygame events
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False

        # Clear surfaces
        self.sim_surface.fill(self.WHITE)
        self.config_surface.fill(self.WHITE)

        # Draw indicators
        warning_indicator = self.font.render(
            f"Warning: {'ON' if state.warn else 'OFF'}",
            True, self.RED
        )
        self.sim_surface.blit(warning_indicator, (10, 10))

        # Draw noci indicator
        noci_indicator = self.font.render(
            f"Nociception: {'ON' if state.noci else 'OFF'}", 
            True, self.RED
        )
        self.sim_surface.blit(noci_indicator, (10, 40))

        # Draw action indicator
        action_indicator = self.font.render(
            f"Action: {'Retreat' if action == 1 else 'Stay'}", 
            True, self.GREEN
        )
        self.sim_surface.blit(action_indicator, (10, 70))
        
        # Draw belief state
        belief_indicator = self.font.render(
            f"Safe belief: {qs[0][0]:.2f}", 
            True, self.BLUE
        )
        self.sim_surface.blit(belief_indicator, (10, 100))

        # Draw regions
        self.draw_regions()

        # Draw worm
        self.pygame.draw.circle(
            self.sim_surface, 
            self.RED, 
            state.position.astype(int), 
            int(self.config.worm_radius)
        )
        for pos in state.positions:
            self.pygame.draw.circle(
                self.sim_surface,
                self.RED,
                pos.astype(int),
                int(self.config.worm_radius)
            )

        # Update display
        self.screen.blit(self.sim_surface, (0, 0))
        self.screen.blit(self.config_surface, (self.sim_width, 0))
        self.pygame.display.flip()
        self.clock.tick(self.fps)
        
        return True

    def cleanup(self):
        """Clean up pygame resources"""
        self.pygame.quit()

def run_visual_simulation(config: SimulationConfig, A_matrix, num_steps: int = None):
    """Run a single simulation with visualization"""
    sim = WormSimulation(config, agent_type=SimpleHomeostaticAgent, A_matrix=A_matrix)
    vis = WormVisualizer(config)
    
    running = True
    step = 0
    history = []
    
    try:
        while running and (num_steps is None or step < num_steps):
            phys_state, qs, action = sim.step()
            history.append((phys_state, qs, action))
            running = vis.draw_frame(phys_state, qs, action)
            step += 1
    finally:
        vis.cleanup()
    
    return history

def update_A_from_history(history, A, learning_rate=1):
    for phys_state, qs, action in history:
            
            # Update A matrix - now only need to update one matrix
            safe_state_value = qs[0][0]
            harmful_state_value = qs[0][1]
            
            # Create one-hot vector for the observation
            update_vector = np.zeros(4) #num joint observations
            warn, noci = phys_state.warn, phys_state.noci
            joint_observation = 0 if (warn, noci) == (False, False) else 1 if (warn, noci) == (False, True) else 2 if (warn, noci) == (True, False) else 3
            update_vector[joint_observation] = 1.0
            # update_vector += np.array([.5, .5, 0, 0]) if noci_observation == 0 else np.array([0, 0, .5, .5])
            # update_vector += np.array([.5, 0, .5, 0]) if warn_observation == 0 else np.array([0, .5, 0, .5])
            
            # Update A matrix for both states
            A[0][:, 0] += update_vector * safe_state_value * learning_rate
            A[0][:, 1] += update_vector * harmful_state_value * learning_rate
    # Normalize A matrix columns to maintain proper probabilities
    A[0][:, 0] = A[0][:, 0] / np.sum(A[0][:, 0])
    A[0][:, 1] = A[0][:, 1] / np.sum(A[0][:, 1])
    return A

if __name__ == "__main__":
    # Example usage showing both visual and parallel capabilities
    import sys
    


    if len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        # Run parallel simulation experiment
        configs = [SimulationConfig() for _ in range(4)]
        results = run_parallel_simulations(configs, num_steps=1000)

        # Analysis
        for i, history in enumerate(results):
            actions = [step[2] for step in history]
            print(f"Simulation {i} retreat percentage: {sum(a == 1 for a in actions) / len(actions):.2%}")
    else:
        # Run visual simulation
        config = SimulationConfig()

        A_array = utils.obj_array_zeros([
            (4, 2) # num joint observations, num states
        ])
        
        A_array[0][:, 0] = [0.0,  # low prob of warning & noci
                                0.0,   # low prob of no warning & noci
                                0.5,   # high prob of warning & no noci
                                0.5]   # high prob of no warning & no noci

        # In harmful state (1):
        A_array[0][:, 1] = [0.25,  # high prob of warning & noci
                                0.25,   # high prob of no warning & noci
                                0.25,   # zero prob of warning & no noci
                                0.25]   # zero prob of no warning & no noci


        while True:
            history = run_visual_simulation(config, A_array, num_steps=1000)
            # Learning
            A_array = update_A_from_history(history, A_array)
            print(f"Updated A matrix:\n{A_array[0]}")
