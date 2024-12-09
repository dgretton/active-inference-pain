# model.py
import numpy as np
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pymdp
from pymdp import utils
from pymdp.agent import Agent

@dataclass
class SimulationConfig:
    """Configuration parameters for simulation"""
    sim_width: float = 400.0
    sim_height: float = 600.0
    worm_radius: float = 6.0
    worm_length: int = 40
    warn_reset_prob: float = 0.01
    speed: float = 2.0
    learning_rate: float = 0.01
    
    # Region definitions
    warning_height: float = 485.0
    warning_thickness: float = 145.0
    noci_height: float = 550.0
    noci_thickness: float = 50.0

@dataclass
class WormState:
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

class ActiveInferenceAgent:
    """Agent implementing active inference for worm control"""
    
    def __init__(self):
        # Define model dimensions
        self.num_obs_noci = 2
        self.num_obs_warn = 2
        self.num_states = 2
        self.num_controls = 2

        # Initialize matrices
        self.A_array = utils.obj_array_zeros([
            (self.num_obs_noci, self.num_states),
            (self.num_obs_warn, self.num_states)
        ])
        self.A_array[0][:, 0] = [0.0, 1.0]
        self.A_array[0][:, 1] = [1.0, 0.0]
        self.A_array[1][:, 0] = [0.5, 0.5]
        self.A_array[1][:, 1] = [0.5, 0.5]

        self.B_array = utils.obj_array_zeros([
            (self.num_states, self.num_states, self.num_controls)
        ])
        self.B_array[0][:, :, 0] = np.eye(2)
        self.B_array[0][:, :, 1] = [[1, 1], [0, 0]]

        self.C_vector = utils.obj_array_zeros([
            (self.num_obs_noci,),
            (self.num_obs_warn,)
        ])
        self.C_vector[0] = np.array([0.0, 1.0])
        self.C_vector[1] = np.array([0.5, 0.5])

        self.E_matrix = np.array([0.8, 0.2])

        # Initialize agent and beliefs
        self.agent = Agent(A=self.A_array, B=self.B_array, C=self.C_vector, E=self.E_matrix)
        self.qs = utils.obj_array_uniform([(self.num_states,)])

    def update(self, observation: Tuple[int, int], learning_rate: float) -> Tuple[int, np.ndarray]:
        """Update agent beliefs and get action"""
        # Update beliefs
        self.qs = self.agent.infer_states(observation)
        
        # Get action
        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()[0]

        # Update A matrix
        noci_observation, warn_observation = observation
        safe_state_value = self.qs[0][0]
        harmful_state_value = self.qs[0][1]
        
        one_hot_warn_vector = np.array([1.0, 0.001]) if warn_observation == 0 else np.array([0.001, 1.0])
        
        self.agent.A[1][:, 0] += one_hot_warn_vector * safe_state_value * learning_rate
        self.agent.A[1][:, 1] += one_hot_warn_vector * harmful_state_value * learning_rate
        print("A matrix:", self.agent.A)

        return action, self.qs

class WormSimulation:
    """Main simulation class"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = WormState.initialize(config)
        self.agent = ActiveInferenceAgent()

    def update_physics(self, action: int) -> None:
        """Update worm physics"""
        # Update movement vector
        movement = self.state.movement
        movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
        movement += np.random.uniform(-1, 1, 2) * 0.1

        if action == 1:
            movement += np.array([0, -2])
        else:
            movement += np.array([0, 0.01])

        movement /= np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else 1
        movement *= self.config.speed

        # Check region collisions
        self.state.noci = False
        if action != 1:
            y = self.state.position[1]
            if self.config.warning_height <= y <= (self.config.warning_height + self.config.warning_thickness):
                self.state.warn = True
            elif self.config.noci_height <= y <= (self.config.noci_height + self.config.noci_thickness):
                self.state.noci = True

        # Update position
        new_pos = self.state.position + movement
        self.state.movement = movement
        
        # Reset warning with probability
        if np.random.rand() < self.config.warn_reset_prob:
            self.state.warn = False

        # Update segment positions
        positions = self.state.positions
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
        self.state.position = new_pos

    def get_observations(self) -> Tuple[int, int]:
        """Get current observations from worm state"""
        warn_observation = 1 if self.state.warn else 0
        noci_observation = 0 if self.state.noci else 1
        return (noci_observation, warn_observation)

    def step(self) -> Tuple[WormState, np.ndarray, int]:
        """Perform one simulation step"""
        observation = self.get_observations()
        action, qs = self.agent.update(observation, self.config.learning_rate)
        self.update_physics(action)
        return self.state, qs, action

def run_simulation(config: SimulationConfig, num_steps: int) -> List[Tuple[WormState, np.ndarray, int]]:
    """Run a single simulation for specified number of steps"""
    sim = WormSimulation(config)
    history = []
    
    for _ in range(num_steps):
        state, qs, action = sim.step()
        history.append((state, qs, action))
        
    return history

def run_parallel_simulations(configs: List[SimulationConfig], 
                           num_steps: int, 
                           num_processes: int = None) -> List[List[Tuple[WormState, np.ndarray, int]]]:
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

    def draw_frame(self, state: WormState, qs: np.ndarray, action: int) -> bool:
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
        
        action_indicator = self.font.render(
            f"Action: {'Retreat' if action == 1 else 'Stay'}", 
            True, self.GREEN
        )
        self.sim_surface.blit(action_indicator, (10, 40))
        
        # Draw belief state
        belief_indicator = self.font.render(
            f"Safe belief: {qs[0][0]:.2f}", 
            True, self.BLUE
        )
        self.sim_surface.blit(belief_indicator, (10, 70))

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

def run_visual_simulation(config: SimulationConfig, num_steps: int = None):
    """Run a single simulation with visualization"""
    sim = WormSimulation(config)
    vis = WormVisualizer(config)
    
    running = True
    step = 0
    
    try:
        while running and (num_steps is None or step < num_steps):
            state, qs, action = sim.step()
            running = vis.draw_frame(state, qs, action)
            step += 1
    finally:
        vis.cleanup()
    
    return step

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
        run_visual_simulation(config)