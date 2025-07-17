# model.py
import numpy as np
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Tuple
from pymdp import utils
# from worm_simulation_dynamic_perception_and_policy_selection_chatgpt import ActiveInferenceAgent
from agent_worm import ActiveInferenceWormAgent, SimpleHomeostaticAgent, AssociativeLearningWormAgent, SimpleLearningAgent

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
    weird_smell_height: float = 450.0
    weird_smell_thickness: float = 150.0
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
    
    def __init__(self, config: SimulationConfig, agent_type=AssociativeLearningWormAgent, A_matrix=None):
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

    def get_observations(self) -> Tuple[bool, bool]:
        """Get current observations from worm state as boolean values"""
        return (self.phys_state.noci, self.phys_state.weird_smell)

    def step(self) -> Tuple[WormPhysState, np.ndarray, int]:
        """Perform one simulation step"""
        observation = self.get_observations()
        action, qs = self.agent.infer(observation)
        self.update_physics(action)
        return self.phys_state, qs, action

def run_simulation(config: SimulationConfig, num_steps: int, agent_type=AssociativeLearningWormAgent) -> List[Tuple[WormPhysState, np.ndarray, int]]:
    """Run a single simulation for specified number of steps"""
    sim = WormSimulation(config, agent_type=agent_type)
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
        
        # Track preference history for visualization
        self.preference_history = []
        self.max_history_length = 100  # Keep last 100 updates

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

    def draw_a_matrix_panel(self, A_matrix):
        """Draw the A matrix visualization panel for separate modalities"""
        panel_width = self.config_surface.get_width()
        y_offset = 20
        
        # Check if A_matrix is the new format (list/array of modalities) or old format
        if isinstance(A_matrix, (list, tuple)) or (hasattr(A_matrix, '__len__') and len(A_matrix) == 2):
            # New format: separate modalities
            
            # Title
            title = self.font.render("A Matrices (Separate Modalities)", True, self.BLACK)
            self.config_surface.blit(title, (10, y_offset))
            y_offset += 30
            
            # State labels for new 3-state system
            state_labels = ["Safe", "Warning", "Harmful"]
            
            # Draw Noci Modality
            noci_title = self.small_font.render("Noci Modality:", True, self.BLACK)
            self.config_surface.blit(noci_title, (10, y_offset))
            y_offset += 20
            
            # Headers for noci modality
            for i, state_label in enumerate(state_labels):
                x_pos = 120 + i * 80
                header = self.small_font.render(state_label, True, self.BLACK)
                self.config_surface.blit(header, (x_pos, y_offset))
            y_offset += 20
            
            # Noci observations
            noci_obs_labels = ["Noci Present", "Noci Absent"]
            for obs_idx, obs_label in enumerate(noci_obs_labels):
                obs_text = self.small_font.render(obs_label, True, self.BLACK)
                self.config_surface.blit(obs_text, (10, y_offset))
                
                for state_idx in range(3):
                    x_pos = 120 + state_idx * 80
                    value = A_matrix[0][obs_idx, state_idx]
                    
                    # Color code based on value
                    if value > 0.6:
                        color = self.GREEN
                    elif value > 0.3:
                        color = self.ORANGE
                    else:
                        color = self.RED
                        
                    value_text = self.small_font.render(f"{value:.2f}", True, color)
                    self.config_surface.blit(value_text, (x_pos, y_offset))
                
                y_offset += 18
            
            y_offset += 10
            
            # Draw Smell Modality
            smell_title = self.small_font.render("Smell Modality:", True, self.BLACK)
            self.config_surface.blit(smell_title, (10, y_offset))
            y_offset += 20
            
            # Headers for smell modality
            for i, state_label in enumerate(state_labels):
                x_pos = 120 + i * 80
                header = self.small_font.render(state_label, True, self.BLACK)
                self.config_surface.blit(header, (x_pos, y_offset))
            y_offset += 20
            
            # Smell observations
            smell_obs_labels = ["Smell Present", "Smell Absent"]
            for obs_idx, obs_label in enumerate(smell_obs_labels):
                obs_text = self.small_font.render(obs_label, True, self.BLACK)
                self.config_surface.blit(obs_text, (10, y_offset))
                
                for state_idx in range(3):
                    x_pos = 120 + state_idx * 80
                    value = A_matrix[1][obs_idx, state_idx]
                    
                    # Color code based on value
                    if value > 0.6:
                        color = self.GREEN
                    elif value > 0.3:
                        color = self.ORANGE
                    else:
                        color = self.RED
                        
                    value_text = self.small_font.render(f"{value:.2f}", True, color)
                    self.config_surface.blit(value_text, (x_pos, y_offset))
                
                y_offset += 18
                
        else:
            # Old format: joint observations (fallback)
            title = self.font.render("A Matrix (Joint Observations)", True, self.BLACK)
            self.config_surface.blit(title, (10, y_offset))
            y_offset += 40
            
            obs_labels = ["Smell+Noci", "NoSmell+Noci", "Smell+NoNoci", "NoSmell+NoNoci"]
            state_labels = ["Safe", "Harmful"]
            
            # Draw headers
            for i, state_label in enumerate(state_labels):
                x_pos = 120 + i * 120
                header = self.small_font.render(state_label, True, self.BLACK)
                self.config_surface.blit(header, (x_pos, y_offset))
            y_offset += 25
            
            # Draw values
            for obs_idx, obs_label in enumerate(obs_labels):
                obs_text = self.small_font.render(obs_label, True, self.BLACK)
                self.config_surface.blit(obs_text, (10, y_offset))
                
                for state_idx in range(min(2, A_matrix.shape[1])):
                    x_pos = 120 + state_idx * 120
                    value = A_matrix[obs_idx, state_idx]
                    
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
    
    def draw_preferences_panel(self, agent, y_start: int):
        """Draw the preferences (C vector) visualization panel"""
        y_offset = y_start
        
        # Title
        title = self.font.render("Preferences (C Vector)", True, self.BLACK)
        self.config_surface.blit(title, (10, y_offset))
        y_offset += 30
        
        # Check if agent has C_vector attribute
        if hasattr(agent, 'C_vector'):
            # Current preferences
            current_title = self.small_font.render("Current Preferences:", True, self.BLACK)
            self.config_surface.blit(current_title, (10, y_offset))
            y_offset += 20
            
            # Noci preferences
            noci_pref_text = self.small_font.render("Noci:", True, self.BLACK)
            self.config_surface.blit(noci_pref_text, (10, y_offset))
            
            noci_present_pref = agent.C_vector[0][0]
            noci_absent_pref = agent.C_vector[0][1]
            
            # Color code preferences
            noci_present_color = self.RED if noci_present_pref < 0 else self.GREEN if noci_present_pref > 0 else self.BLACK
            noci_absent_color = self.GREEN if noci_absent_pref > 0 else self.RED if noci_absent_pref < 0 else self.BLACK
            
            noci_values_text = self.small_font.render(f"Present: {noci_present_pref:.3f}, Absent: {noci_absent_pref:.3f}", True, self.BLACK)
            self.config_surface.blit(noci_values_text, (60, y_offset))
            y_offset += 18
            
            # Smell preferences
            smell_pref_text = self.small_font.render("Smell:", True, self.BLACK)
            self.config_surface.blit(smell_pref_text, (10, y_offset))
            
            smell_present_pref = agent.C_vector[1][0]
            smell_absent_pref = agent.C_vector[1][1]
            
            # Color code smell preferences
            smell_present_color = self.GREEN if smell_present_pref > 0 else self.RED if smell_present_pref < 0 else self.BLACK
            smell_absent_color = self.GREEN if smell_absent_pref > 0 else self.RED if smell_absent_pref < 0 else self.BLACK
            
            # Show smell preference with color coding
            smell_present_text = self.small_font.render(f"Present: {smell_present_pref:.3f}", True, smell_present_color)
            smell_absent_text = self.small_font.render(f"Absent: {smell_absent_pref:.3f}", True, smell_absent_color)
            
            self.config_surface.blit(smell_present_text, (60, y_offset))
            self.config_surface.blit(smell_absent_text, (200, y_offset))
            y_offset += 25
            
            # Store current preferences for history
            current_prefs = {
                'smell_present': smell_present_pref,
                'smell_absent': smell_absent_pref,
                'noci_present': noci_present_pref,
                'noci_absent': noci_absent_pref
            }
            self.preference_history.append(current_prefs)
            
            # Keep history manageable
            if len(self.preference_history) > self.max_history_length:
                self.preference_history.pop(0)
            
            # Draw preference trend graph for smell
            if len(self.preference_history) > 1:
                trend_title = self.small_font.render("Smell Preference Trend:", True, self.BLACK)
                self.config_surface.blit(trend_title, (10, y_offset))
                y_offset += 20
                
                # Graph parameters
                graph_width = 300
                graph_height = 60
                graph_x = 10
                graph_y = y_offset
                
                # Draw graph background
                graph_rect = self.pygame.Rect(graph_x, graph_y, graph_width, graph_height)
                self.pygame.draw.rect(self.config_surface, self.LIGHTGRAY, graph_rect, 1)
                
                # Draw zero line
                zero_y = graph_y + graph_height // 2
                self.pygame.draw.line(self.config_surface, self.DARKGRAY, 
                                    (graph_x, zero_y), (graph_x + graph_width, zero_y), 1)
                
                # Plot smell preference history
                if len(self.preference_history) > 1:
                    smell_values = [p['smell_present'] for p in self.preference_history]
                    
                    # Use fixed range for better visualization
                    display_min = -1.0  # Expected minimum smell preference
                    display_max = 1.0   # Expected maximum smell preference
                    display_range = display_max - display_min
                    
                    points = []
                    for i, val in enumerate(smell_values):
                        x = graph_x + (i / max(len(smell_values) - 1, 1)) * graph_width
                        # Normalize to fixed range and invert y so positive values go up
                        normalized_val = (val - display_min) / display_range
                        # Clamp to [0, 1] range
                        normalized_val = max(0.0, min(1.0, normalized_val))
                        y = graph_y + graph_height - (normalized_val * graph_height)
                        points.append((int(x), int(y)))
                    
                    # Draw the trend line
                    if len(points) > 1:
                        self.pygame.draw.lines(self.config_surface, self.BLUE, False, points, 2)
                    
                    # Draw current value indicator
                    if points:
                        current_point = points[-1]
                        self.pygame.draw.circle(self.config_surface, self.RED, current_point, 3)
                
                # Add fixed labels for better reference
                max_label = self.small_font.render("1.0", True, self.BLACK)
                min_label = self.small_font.render("-1.0", True, self.BLACK)
                zero_label = self.small_font.render("0.0", True, self.DARKGRAY)
                
                self.config_surface.blit(max_label, (graph_x + graph_width + 5, graph_y))
                self.config_surface.blit(min_label, (graph_x + graph_width + 5, graph_y + graph_height - 15))
                self.config_surface.blit(zero_label, (graph_x + graph_width + 5, graph_y + graph_height // 2 - 8))
                
                y_offset += graph_height + 25
        else:
            # Fallback if no C_vector
            no_prefs_text = self.small_font.render("No preference data available", True, self.DARKGRAY)
            self.config_surface.blit(no_prefs_text, (10, y_offset))
            y_offset += 20
        
        return y_offset
    
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

    def draw_frame(self, state: WormPhysState, qs: np.ndarray, action: int, A_matrix: np.ndarray = None, agent=None) -> bool:
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
            y_offset = self.draw_preferences_panel(agent, y_offset)
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

def run_visual_simulation(config: SimulationConfig, A_matrix=None, num_steps: int = None, agent_type=AssociativeLearningWormAgent):
    """Run a single simulation with visualization"""
    sim = WormSimulation(config, agent_type=agent_type, A_matrix=A_matrix)
    vis = WormVisualizer(config)
    
    running = True
    step = 0
    history = []
    
    try:
        while running and (num_steps is None or step < num_steps):
            phys_state, qs, action = sim.step()
            history.append((phys_state, qs, action))
            # Pass the current A matrix to the visualizer
            if hasattr(sim.agent, 'agent'):
                current_A_matrix = sim.agent.agent.A
            elif hasattr(sim.agent, 'A_array'):
                current_A_matrix = sim.agent.A_array
            else:
                current_A_matrix = A_matrix
            running = vis.draw_frame(phys_state, qs, action, current_A_matrix, sim.agent)
            step += 1
    finally:
        vis.cleanup()
    
    return history

def update_A_from_history(history, A, learning_rate=0.02, pseudocount=0.01):
    """
    Updated learning function that works with the new associative agent structure.
    This is kept for backward compatibility but the agent's internal learning is preferred.
    """
    # For the new agent structure, we should use the agent's own learning mechanism
    # This function is kept for compatibility with existing code
    
    # Convert history to experience format
    experience_history = []
    for phys_state, qs, action in history:
        observation = (phys_state.noci, phys_state.weird_smell)
        reward = -1.0 if phys_state.noci else 0.0
        experience_history.append((observation, qs, action, reward))
    
    # Create a temporary agent to do the learning
    temp_agent = AssociativeLearningWormAgent(A)
    temp_agent.learn_associations(experience_history, learning_rate)
    
    return temp_agent.A_array

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

        # Initialize A matrices for separate modalities (noci and smell)
        A_array = utils.obj_array_zeros([
            (2, 3),  # noci modality: [noci_present, noci_absent] x [safe, warning, harmful]
            (2, 3)   # smell modality: [smell_present, smell_absent] x [safe, warning, harmful]
        ])

        # A matrix for noci modality
        # States: 0=safe, 1=warning, 2=harmful
        A_array[0][:, 0] = [0.05, 0.95]  # safe: very unlikely noci
        A_array[0][:, 1] = [0.1, 0.9]    # warning: still mostly no noci
        A_array[0][:, 2] = [0.9, 0.1]    # harmful: very likely noci
        
        # A matrix for smell modality
        A_array[1][:, 0] = [0.1, 0.9]    # safe: unlikely smell
        A_array[1][:, 1] = [0.8, 0.2]    # warning: likely smell (predictive cue)
        A_array[1][:, 2] = [0.3, 0.7]    # harmful: some smell but not as strong predictor

        print("Starting associative learning simulation...")
        print("The agent should learn to associate smell with upcoming nociception.")
        print("Watch for:")
        print("1. Smell aversion development (C_smell[0] becomes negative)")
        print("2. Predictive behavior (retreat when smell detected)")
        print("3. A matrix updates showing smell-noci associations")
        print()

        episode = 0
        while True:
            print(f"\n=== Episode {episode + 1} ===")
            
            # Run simulation with learning agent
            history = run_visual_simulation(config, A_array, num_steps=800, agent_type=SimpleLearningAgent)
            
            # Get learning metrics from the last simulation
            if history:
                # Create agent to analyze final state
                temp_agent = SimpleLearningAgent(A_array)
                temp_agent.learn(history)
                metrics = temp_agent.get_learning_metrics()
                
                if metrics:
                    print(f"Learning Progress:")
                    print(f"  Smell aversion: {metrics['smell_aversion']:.3f}")
                    print(f"  Smell preference difference: {metrics['smell_preference_diff']:.3f}")
                    print(f"  A(smell|warning): {metrics['A_smell_warning_predictive']:.3f}")
                    print(f"  A(noci|harmful): {metrics['A_noci_harmful_predictive']:.3f}")
                
                # Update A_array for next episode
                A_array = temp_agent.A_array
            
            episode += 1
            
            # Optional: break after certain number of episodes
            if episode >= 10:
                print(f"\nCompleted {episode} learning episodes.")
                break
