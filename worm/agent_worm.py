from typing import Tuple
import numpy as np
from pymdp import utils
from pymdp.agent import Agent

class ActiveInferenceWormAgent:
    pass

class AssociativeLearningWormAgent(ActiveInferenceWormAgent):
    """
    Worm agent that can learn associations between smell and nociception,
    based on successful classical conditioning structure with separate modalities.
    """
    
    def __init__(self, A_matrix=None):
        # Separate observation modalities (key insight from classical conditioning)
        self.num_obs_noci = 2    # [noci_present, noci_absent]
        self.num_obs_smell = 2   # [smell_present, smell_absent]
        self.num_states = 3      # [safe, warning, harmful] - allows temporal prediction
        self.num_controls = 2    # [stay, retreat]
        
        # Initialize A matrices for separate modalities
        if A_matrix is not None:
            self.A_array = A_matrix
        else:
            self.A_array = utils.obj_array_zeros([
                (self.num_obs_noci, self.num_states),   # noci observations
                (self.num_obs_smell, self.num_states)   # smell observations
            ])
            
            # A matrix for noci modality
            # States: 0=safe, 1=warning, 2=harmful
            self.A_array[0][:, 0] = [0.05, 0.95]  # safe: very unlikely noci
            self.A_array[0][:, 1] = [0.1, 0.9]    # warning: still mostly no noci
            self.A_array[0][:, 2] = [0.9, 0.1]    # harmful: very likely noci
            
            # A matrix for smell modality - START NEUTRAL
            self.A_array[1][:, 0] = [0.33, 0.67]  # safe: neutral about smell
            self.A_array[1][:, 1] = [0.33, 0.67]  # warning: neutral about smell initially  
            self.A_array[1][:, 2] = [0.33, 0.67]  # harmful: neutral about smell initially

        # B matrix: state transitions with temporal structure
        self.B_array = utils.obj_array_zeros([
            (self.num_states, self.num_states, self.num_controls)
        ])
        
        # Action 0: stay - more conservative transitions to encourage exploration
        # Each column must sum to 1.0 (transition probabilities from each state)
        self.B_array[0][:, :, 0] = [
            [0.9, 0.1, 0.0],  # to safe: from safe=0.9, from warning=0.1, from harmful=0.0
            [0.1, 0.6, 0.2],  # to warning: from safe=0.1, from warning=0.6, from harmful=0.2
            [0.0, 0.3, 0.8]   # to harmful: from safe=0.0, from warning=0.3, from harmful=0.8
        ]
        
        # Action 1: retreat - move toward safety
        self.B_array[0][:, :, 1] = [
            [0.9, 0.7, 0.4],  # to safe: from safe=0.9, from warning=0.7, from harmful=0.4
            [0.1, 0.2, 0.4],  # to warning: from safe=0.1, from warning=0.2, from harmful=0.4
            [0.0, 0.1, 0.2]   # to harmful: from safe=0.0, from warning=0.1, from harmful=0.2
        ]
        
        # Preferences over observations (C vector) - key for association learning
        self.C_vector = utils.obj_array_zeros([
            (self.num_obs_noci,),
            (self.num_obs_smell,)
        ])
        
        # Strong preference against noci
        self.C_vector[0] = np.array([-2.0, 1.0])  # hate noci, prefer no noci
        
        # Initially neutral about smell (will learn through association)
        self.C_vector[1] = np.array([0.0, 0.0])   # completely neutral about smell initially
        
        # Action preferences - balanced to encourage exploration
        self.E_matrix = np.array([0.6, 0.4])
        
        # Initialize agent
        self.agent = Agent(A=self.A_array, B=self.B_array, C=self.C_vector, E=self.E_matrix)
        self.qs = utils.obj_array_uniform([(self.num_states,)])
        
        # Track learning history for analysis
        self.learning_history = []
        
        # Store last action probabilities for debugging
        self.last_action_probs = np.array([0.5, 0.5])

    def infer(self, observation: Tuple[bool, bool]) -> Tuple[int, np.ndarray]:
        """
        Update agent beliefs and get action.
        observation: (noci_present, smell_present)
        """
        noci_present, smell_present = observation
        
        # Convert to observation indices (0 = present, 1 = absent)
        noci_obs = 0 if noci_present else 1
        smell_obs = 0 if smell_present else 1
        
        # Update beliefs using both observation modalities
        self.qs = self.agent.infer_states([noci_obs, smell_obs])
        
        # Get action through policy inference
        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()[0]
        
        # Store last action probabilities for debugging
        self.last_action_probs = q_pi
        
        # Debug decision making occasionally
        if not hasattr(self, 'debug_counter'):
            self.debug_counter = 0
        self.debug_counter += 1
        if self.debug_counter % 100 == 0:  # Print every 100 steps
            self.debug_decision(observation)
        
        return action, self.qs

    def debug_decision(self, observation):
        """Debug why agent is choosing actions"""
        noci_present, smell_present = observation
        noci_obs = 0 if noci_present else 1
        smell_obs = 0 if smell_present else 1
        
        # Get current beliefs and policy
        qs = self.agent.infer_states([noci_obs, smell_obs])
        q_pi, efe = self.agent.infer_policies()
        
        print(f"Observation: noci={noci_present}, smell={smell_present}")
        print(f"State beliefs: {qs[0]}")
        print(f"Action probabilities: stay={q_pi[0]:.3f}, retreat={q_pi[1]:.3f}")
        print(f"C_smell: {self.C_vector[1]}")
        print(f"E_matrix: {self.E_matrix}")
        print("---")

    def learn_associations(self, experience_history, learning_rate=0.003):
        """
        Learn associations based on experience history.
        Experience history: list of (observation, state_beliefs, action, reward) tuples
        """
        for obs, qs, action, reward in experience_history:
            noci_present, smell_present = obs
            noci_obs = 0 if noci_present else 1
            smell_obs = 0 if smell_present else 1
            
            # Update A matrices based on experience (similar to classical conditioning)
            self._update_observation_model(noci_obs, smell_obs, qs, learning_rate)
            
            # Learn preferences (key insight from classical conditioning)
            self._update_preferences(noci_obs, smell_obs, reward, learning_rate)
        
        # Update agent matrices
        self.agent.A = self.A_array
        self.agent.C = self.C_vector
        
        # Store learning state for analysis
        self.learning_history.append({
            'A_noci': self.A_array[0].copy(),
            'A_smell': self.A_array[1].copy(),
            'C_noci': self.C_vector[0].copy(),
            'C_smell': self.C_vector[1].copy()
        })

    def _update_observation_model(self, noci_obs, smell_obs, qs, learning_rate):
        """Update A matrices based on observed co-occurrences"""
        # Create one-hot observation vectors
        noci_vec = np.zeros(2)
        noci_vec[noci_obs] = 1.0
        
        smell_vec = np.zeros(2)
        smell_vec[smell_obs] = 1.0
        
        # Conservative A matrix updates with Dirichlet-style learning
        for state in range(self.num_states):
            belief_weight = qs[0][state]
            
            # Only update if belief weight is significant
            if belief_weight > 0.2:  # Lower threshold for updates
                # Update noci A matrix with small learning rate
                self.A_array[0][:, state] += noci_vec * belief_weight * learning_rate * 0.1
                
                # Update smell A matrix with small learning rate
                self.A_array[1][:, state] += smell_vec * belief_weight * learning_rate * 0.1
        
        # Normalize to maintain probability constraints
        for state in range(self.num_states):
            # Normalize noci modality
            col_sum = self.A_array[0][:, state].sum()
            if col_sum > 0:
                self.A_array[0][:, state] /= col_sum
                
            # Normalize smell modality
            col_sum = self.A_array[1][:, state].sum()
            if col_sum > 0:
                self.A_array[1][:, state] /= col_sum

    def _update_preferences(self, noci_obs, smell_obs, reward, learning_rate):
        """
        Update preferences based on reward - key for association learning.
        This is the mechanism that makes smell acquire motivational significance.
        """
        # Persistent preference learning that accumulates over time
        if noci_obs == 0:  # noci present
            # If smell was also present, create strong negative association
            if smell_obs == 0:  # smell present too
                self.C_vector[1][0] -= learning_rate * 0.1  # stronger aversion to smell
                self.C_vector[1][1] += learning_rate * 0.05  # stronger preference for no smell
            # Even if smell wasn't present during noci, slightly reduce smell preference
            # This creates a more general aversion that persists
            else:
                self.C_vector[1][0] -= learning_rate * 0.02  # general smell aversion
                self.C_vector[1][1] += learning_rate * 0.01  # general preference for no smell
            
        # If we avoided noci in presence of smell, only very slightly reduce aversion
        elif noci_obs == 1 and smell_obs == 0:  # no noci, but smell present
            # Much smaller positive update to prevent oscillation
            self.C_vector[1][0] += learning_rate * 0.005  # very slight positive update

    def get_learning_metrics(self):
        """Get metrics to analyze learning progress"""
        if not self.learning_history:
            return None
            
        latest = self.learning_history[-1]
        
        return {
            'smell_aversion': -latest['C_smell'][0],  # negative preference for smell
            'smell_preference_diff': latest['C_smell'][1] - latest['C_smell'][0],
            'A_smell_warning_predictive': latest['A_smell'][0, 1],  # P(smell|warning)
            'A_noci_harmful_predictive': latest['A_noci'][0, 2],   # P(noci|harmful)
            'learning_episodes': len(self.learning_history)
        }


class SimpleHomeostaticAgent(AssociativeLearningWormAgent):
    """Backward compatibility wrapper for the original agent interface"""
    
    def __init__(self, A_matrix=None):
        super().__init__(A_matrix)
        
    def infer(self, observation: Tuple[int, int]) -> Tuple[int, np.ndarray]:
        """Convert old observation format to new format"""
        noci_observation, weird_smell_observation = observation
        
        # Convert from old encoding (0=present, 1=absent) to boolean
        noci_present = (noci_observation == 0)
        smell_present = (weird_smell_observation == 0)
        
        return super().infer((noci_present, smell_present))
    
    # def learn(self, history, learning_rate):
    #     pass
    

class SimpleLearningAgent(AssociativeLearningWormAgent):
    """Agent with enhanced learning capabilities using the new associative structure"""
    
    def __init__(self, A_matrix=None):
        super().__init__(A_matrix)

    def learn(self, history, learning_rate=0.003, pseudocount=0.01):
        """
        Learn from history using the new associative learning framework.
        history: list of tuples [(phys_state, qs, action), ...]
        """
        # Convert history to experience format expected by learn_associations
        experience_history = []
        
        for phys_state, qs, action in history:
            # Convert physical state to observation format
            observation = (phys_state.noci, phys_state.weird_smell)
            
            # Calculate reward (negative for noci)
            reward = -1.0 if phys_state.noci else 0.0
            
            experience_history.append((observation, qs, action, reward))
        
        # Use the associative learning mechanism
        self.learn_associations(experience_history, learning_rate)
        
        # Print learning progress with more detail
        metrics = self.get_learning_metrics()
        if metrics:
            print(f"Learning metrics: {metrics}")
            print(f"Current C_smell: {self.C_vector[1]}")
            print(f"A_noci shape: {self.A_array[0].shape}, A_smell shape: {self.A_array[1].shape}")
