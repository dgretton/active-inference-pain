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
    weird_smell_reset_prob: float = 1.0
    speed: float = 2.0
    learning_rate: float = .05
    
    # Region definitions
    weird_smell_height: float = 525.0
    weird_smell_thickness: float = 75.0
    noci_height: float = 550.0
    noci_thickness: float = 50.0
    # weird_smell_height: float = 550.0  # Change from 485.0
    # weird_smell_thickness: float = 50.0  # Change from 145.0
    # noci_height: float = 550.0  # Already correct
    # noci_thickness: float = 50.0  # Already correct

@dataclass
class WormPhysState:
    """Physical state of the worm"""
    position: np.ndarray  # Current head position
    positions: List[np.ndarray]  # List of segment positions
    movement: np.ndarray  # Current movement vector
    weird_smell: bool = False  # Weird smell signal state
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
            weird_smell=False,
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
        
        # Reset weird smell with probability
        if np.random.rand() < self.config.weird_smell_reset_prob:
            self.phys_state.weird_smell = False

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
        if self.config.weird_smell_height <= y <= (self.config.weird_smell_height + self.config.weird_smell_thickness):
            self.phys_state.weird_smell = True
        if self.config.noci_height <= y <= (self.config.noci_height + self.config.noci_thickness):
            self.phys_state.noci = True

    def get_observations(self) -> Tuple[int, int]:
        """Get current observations from worm state"""
        weird_smell_observation = 0 if self.phys_state.weird_smell else 1
        noci_observation = 0 if self.phys_state.noci else 1
        return (noci_observation, weird_smell_observation)

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
        self.PURPLE = (128, 0, 128)
        self.DARKGRAY = (64, 64, 64)

        # Pygame setup
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Actively Enhanced Worm Simulation")
        self.sim_surface = pygame.Surface((self.sim_width, self.sim_height))
        self.config_surface = pygame.Surface((width - self.sim_width, height))
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Store config for drawing regions
        self.config = config
        
        # Tracking for visualization
        self.observation_counts = np.zeros((4,))  # Count of each joint observation
        self.total_observations = 0

    def draw_regions(self):
        """Draw weird smell and nociception regions"""
        # Weird smell region
        weird_smell_rect = self.pygame.Rect(
            0, 
            self.config.weird_smell_height,
            self.sim_width, 
            self.config.weird_smell_thickness
        )
        self.pygame.draw.rect(self.sim_surface, self.ORANGE, weird_smell_rect, 2)
        weird_smell_text = self.font.render("weird smell", True, self.ORANGE)
        self.sim_surface.blit(weird_smell_text, (
            weird_smell_rect.centerx - weird_smell_text.get_width() // 2,
            weird_smell_rect.centery - weird_smell_text.get_height() // 2
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

    def draw_a_matrix_panel(self, A_matrix: np.ndarray):
        """Draw the A matrix visualization panel"""
        panel_width = self.config_surface.get_width()
        y_offset = 20
        
        # Title
        title = self.font.render("A Matrix (Observation | State)", True, self.BLACK)
        self.config_surface.blit(title, (10, y_offset))
        y_offset += 40
        
        # Observation labels (matching joint observation encoding)
        obs_labels = [
            "Smell, Noci",          # joint_observation = 0 (both stimuli present)
            "No smell, Noci",       # joint_observation = 1 (only noci present)
            "Smell, No noci",       # joint_observation = 2 (only smell present)
            "No smell, No noci"     # joint_observation = 3 (neither present)
        ]
        
        # State labels
        state_labels = ["Safe", "Harmful"]
        
        # Draw headers
        state_header_y = y_offset
        for i, state_label in enumerate(state_labels):
            x_pos = 120 + i * 120
            header = self.small_font.render(state_label, True, self.BLACK)
            self.config_surface.blit(header, (x_pos, state_header_y))
        y_offset += 25
        
        # Draw A matrix values
        for obs_idx, obs_label in enumerate(obs_labels):
            # Observation label
            obs_text = self.small_font.render(obs_label, True, self.BLACK)
            self.config_surface.blit(obs_text, (10, y_offset))
            
            # Values for each state
            for state_idx in range(2):
                x_pos = 120 + state_idx * 120
                value = A_matrix[obs_idx, state_idx]
                
                # Color code based on value
                if value > 0.6:
                    color = self.GREEN
                elif value > 0.3:
                    color = self.ORANGE
                else:
                    color = self.RED
                    
                value_text = self.small_font.render(f"{value:.3f}", True, color)
                self.config_surface.blit(value_text, (x_pos, y_offset))
            
            y_offset += 20
        
        return y_offset + 20
    
    def draw_observation_counts_panel(self, y_start: int):
        """Draw observation counts panel"""
        y_offset = y_start
        
        # Title
        title = self.font.render("Observation Counts", True, self.BLACK)
        self.config_surface.blit(title, (10, y_offset))
        y_offset += 30
        
        obs_labels = [
            "Smell, Noci",          # joint_observation = 0 (both stimuli present)
            "No smell, Noci",       # joint_observation = 1 (only noci present)
            "Smell, No noci",       # joint_observation = 2 (only smell present)
            "No smell, No noci"     # joint_observation = 3 (neither present)
        ]
        
        # Draw counts and percentages
        for i, label in enumerate(obs_labels):
            count = int(self.observation_counts[i])
            percentage = (self.observation_counts[i] / max(self.total_observations, 1)) * 100
            
            # Draw label
            label_text = self.small_font.render(label, True, self.BLACK)
            self.config_surface.blit(label_text, (10, y_offset))
            
            # Draw count and percentage
            count_text = self.small_font.render(f"{count} ({percentage:.1f}%)", True, self.BLUE)
            self.config_surface.blit(count_text, (200, y_offset))
            
            # Draw simple bar chart
            bar_width = int((percentage / 100) * 150)
            if bar_width > 0:
                bar_rect = self.pygame.Rect(360, y_offset + 2, bar_width, 12)
                self.pygame.draw.rect(self.config_surface, self.LIGHTGRAY, bar_rect)
            
            y_offset += 18
        
        # Total observations
        y_offset += 10
        total_text = self.font.render(f"Total: {self.total_observations}", True, self.BLACK)
        self.config_surface.blit(total_text, (10, y_offset))
        
        return y_offset + 30

    def update_observation_counts(self, state: WormPhysState):
        """Update observation counts for visualization"""
        weird_smell, noci = state.weird_smell, state.noci
        # Match the encoding used in agent: 0 = stimulus present, 1 = stimulus absent
        noci_observation = 0 if noci else 1
        weird_smell_observation = 0 if weird_smell else 1
        
        if (noci_observation, weird_smell_observation) == (0, 0):  # both present
            joint_observation = 0
        elif (noci_observation, weird_smell_observation) == (0, 1):  # noci present, no smell
            joint_observation = 1
        elif (noci_observation, weird_smell_observation) == (1, 0):  # no noci, smell present
            joint_observation = 2
        elif (noci_observation, weird_smell_observation) == (1, 1):  # neither present
            joint_observation = 3
        
        self.observation_counts[joint_observation] += 1
        self.total_observations += 1

    def draw_frame(self, state: WormPhysState, qs: np.ndarray, action: int, A_matrix: np.ndarray = None) -> bool:
        """Draw a single frame. Returns False if window closed."""
        # Handle pygame events
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False

        # Update observation counts
        self.update_observation_counts(state)

        # Clear surfaces
        self.sim_surface.fill(self.WHITE)
        self.config_surface.fill(self.WHITE)

        # Draw indicators
        weird_smell_indicator = self.font.render(
            f"Weird smell: {'ON' if state.weird_smell else 'OFF'}",
            True, self.ORANGE if state.weird_smell else self.DARKGRAY
        )
        self.sim_surface.blit(weird_smell_indicator, (10, 10))

        # Draw noci indicator
        noci_indicator = self.font.render(
            f"Nociception: {'ON' if state.noci else 'OFF'}", 
            True, self.RED if state.noci else self.DARKGRAY
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

        # Draw analysis panels on the right
        if A_matrix is not None:
            y_offset = self.draw_a_matrix_panel(A_matrix)
            self.draw_observation_counts_panel(y_offset)

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
            # Pass the current A matrix to the visualizer
            current_A_matrix = sim.agent.agent.A[0] if hasattr(sim.agent, 'agent') else A_matrix[0]
            running = vis.draw_frame(phys_state, qs, action, current_A_matrix)
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
            weird_smell, noci = phys_state.weird_smell, phys_state.noci
            joint_observation = 0 if (weird_smell, noci) == (False, False) else 1 if (weird_smell, noci) == (False, True) else 2 if (weird_smell, noci) == (True, False) else 3
            update_vector[joint_observation] = 1.0
            # update_vector += np.array([.5, .5, 0, 0]) if noci_observation == 0 else np.array([0, 0, .5, .5])
            # update_vector += np.array([.5, 0, .5, 0]) if weird_smell_observation == 0 else np.array([0, .5, 0, .5])
            
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
            (4, 2), # num joint observations over smell/noci, num states
            #(2, 2) # pain/no pain, num states
        ])

        # The initial A matrix sets the "polarity" of safe and harmful states
        # The A matrix will change significantly during simulation as learning occurs,
        # but initial values correlate noci with harmful state to break symmetry
        # This gives the baseline for where subsequent observations will be "sorted"
        
        # Safe state (0) - noci should be impossible
        A_array[0][:, 0] = [0.0,  # joint_obs 0: smell + noci (impossible in safe)
                           0.0,   # joint_obs 1: no smell + noci (impossible in safe)
                           0.5,   # joint_obs 2: smell + no noci (possible in safe)
                           0.5]   # joint_obs 3: no smell + no noci (possible in safe)

        # Harmful state (1) - noci is likely
        A_array[0][:, 1] = [.5,   # joint_obs 0: smell + noci (likely in harmful)
                           .5,    # joint_obs 1: no smell + noci (likely in harmful)
                           0,     # joint_obs 2: smell + no noci (unlikely in harmful)
                           0]     # joint_obs 3: no smell + no noci (unlikely in harmful)

        # Second matrix is identity
        # A_array[1][:, 0] = [1.0, 0.0]
        # A_array[1][:, 1] = [0.0, 1.0]


        while True:
            history = run_visual_simulation(config, A_array, num_steps=1000)
            # Learning
            A_array = update_A_from_history(history, A_array)
            print(f"Updated A matrix:\n{A_array[0]}")
