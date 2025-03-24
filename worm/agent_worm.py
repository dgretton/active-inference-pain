from typing import Tuple
import numpy as np
from pymdp import utils
from pymdp.agent import Agent

class ActiveInferenceWormAgent:
    pass

class SimpleHomeostaticAgent(ActiveInferenceWormAgent):
    """Agent implementing active inference for worm control using joint observations"""
    
    def __init__(self):
        # Define model dimensions
        self.num_obs_joint = 4  # (warn,noci)/(no-warn,noci)/(warn,no-noci)/(no-warn,no-noci)
        self.num_states = 2     # safe/harmful
        self.num_controls = 2   # stay/retreat

        # Initialize matrices
        self.A_array = utils.obj_array_zeros([
            (self.num_obs_joint, self.num_states)
        ])
        
        # A[observation, state]
        # States: 0=safe, 1=harmful
        # Observations: (warning,nociception)
        # 0 = (0,0) = no-warn,no-noci
        # 1 = (0,1) = no-warn,noci
        # 2 = (1,0) = warn,no-noci
        # 3 = (1,1) = warn,noci

        # Initialize with reasonable priors:
        # In safe state (0):
        self.A_array[0][:, 0] = [0.0,  # low prob of warning & noci
                                0.0,   # low prob of no warning & noci
                                0.5,   # high prob of warning & no noci
                                0.5]   # high prob of no warning & no noci

        # In harmful state (1):
        self.A_array[0][:, 1] = [0.25,  # high prob of warning & noci
                                0.25,   # high prob of no warning & noci
                                0.25,   # zero prob of warning & no noci
                                0.25]   # zero prob of no warning & no noci

        # Keep same B matrix structure
        self.B_array = utils.obj_array_zeros([
            (self.num_states, self.num_states, self.num_controls)
        ])
        self.B_array[0][:, :, 0] = np.eye(2)  # stay action keeps state
        self.B_array[0][:, :, 1] = [[1, .8],  # retreat action tends toward safe
                                   [0, .2]]

        # Preferences over joint observations
        self.C_vector = utils.obj_array_zeros([
            (self.num_obs_joint,)
        ])
        # Prefer no nociception and warning does not matter
        self.C_vector[0] = np.array([0.0, 0.0, 0.5, 0.5])

        # Keep same action preferences
        self.E_matrix = np.array([0.8, 0.2])  # prefer staying to retreating

        # Initialize agent and beliefs
        self.agent = Agent(A=self.A_array, B=self.B_array, C=self.C_vector, E=self.E_matrix)
        self.qs = utils.obj_array_uniform([(self.num_states,)])

    def infer(self, observation: Tuple[int, int]) -> Tuple[int, np.ndarray]:
        """Update agent beliefs and get action"""

        noci_observation, warn_observation = observation # REMEMBER: 0 = warning/noci, 1 = no warning/noci
        if (noci_observation, warn_observation) == (0, 0):  #  noci, warning
            joint_observation = 0
        elif (noci_observation, warn_observation) == (0, 1): # noci, no warning
            joint_observation = 1
        elif (noci_observation, warn_observation) == (1, 0): # no noci, warning
            joint_observation = 2
        elif (noci_observation, warn_observation) == (1, 1): # no noci, no warning
            joint_observation = 3
        else:
            raise ValueError("Invalid observation")

        # Update beliefs
        self.qs = self.agent.infer_states([joint_observation])

        # Get action
        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()[0]

        return action, self.qs
    
    # def learn(self, history, learning_rate):
    #     pass
    

class SimpleLearningAgent(SimpleHomeostaticAgent):
    """Agent implementing active inference for worm control using joint observations"""
    
    def __init__(self):
        # Define model dimensions
        self.num_obs_joint = 4  # (warn,noci)/(no-warn,noci)/(warn,no-noci)/(no-warn,no-noci)
        self.num_states = 2     # safe/harmful
        self.num_controls = 2   # stay/retreat

        # Initialize matrices
        self.A_array = utils.obj_array_zeros([
            (self.num_obs_joint, self.num_states)
        ])
        
        # A[observation, state]
        # States: 0=safe, 1=harmful
        # Observations: (warning,nociception)
        # 0 = (0,0) = no-warn,no-noci
        # 1 = (0,1) = no-warn,noci
        # 2 = (1,0) = warn,no-noci
        # 3 = (1,1) = warn,noci

        # Initialize with reasonable priors:
        # In safe state (0):
        self.A_array[0][:, 0] = [0.0,  # low prob of warning & noci
                                0.0,   # low prob of no warning & noci
                                0.5,   # high prob of warning & no noci
                                0.5]   # high prob of no warning & no noci

        # In harmful state (1):
        self.A_array[0][:, 1] = [0.25,  # high prob of warning & noci
                                0.25,   # high prob of no warning & noci
                                0.25,   # zero prob of warning & no noci
                                0.25]   # zero prob of no warning & no noci

        # Keep same B matrix structure
        self.B_array = utils.obj_array_zeros([
            (self.num_states, self.num_states, self.num_controls)
        ])
        self.B_array[0][:, :, 0] = np.eye(2)  # stay action keeps state
        self.B_array[0][:, :, 1] = [[1, .8],  # retreat action tends toward safe
                                   [0, .2]]

        # Preferences over joint observations
        self.C_vector = utils.obj_array_zeros([
            (self.num_obs_joint,)
        ])
        # Prefer no nociception and warning does not matter
        self.C_vector[0] = np.array([0.0, 0.0, 0.5, 0.5])

        # Keep same action preferences
        self.E_matrix = np.array([0.8, 0.2])  # prefer staying to retreating

        # Initialize agent and beliefs
        self.agent = Agent(A=self.A_array, B=self.B_array, C=self.C_vector, E=self.E_matrix)
        self.qs = utils.obj_array_uniform([(self.num_states,)])

    # def learn(self, history, learning_rate): TODO: accumulate information from history and figure out how to update A matrix based on sum of episodic experience
    #     # Update A matrix - now only need to update one matrix
    #     safe_state_value = self.qs[0][0]
    #     harmful_state_value = self.qs[0][1]
        
    #     # Create one-hot vector for the observation
    #     update_vector = np.zeros(self.num_obs_joint)
    #     update_vector[joint_observation] = 1.0
    #     # update_vector += np.array([.5, .5, 0, 0]) if noci_observation == 0 else np.array([0, 0, .5, .5])
    #     # update_vector += np.array([.5, 0, .5, 0]) if warn_observation == 0 else np.array([0, .5, 0, .5])
        
    #     # Update A matrix for both states
    #     self.agent.A[0][:, 0] += update_vector * safe_state_value * learning_rate
    #     self.agent.A[0][:, 1] += update_vector * harmful_state_value * learning_rate

    #     # No matter what, the probability of noci in safe state should be zero
    #     # self.agent.A[0][0, 0] = 0.0
    #     # self.agent.A[0][1, 0] = 0.0
        
    #     # # Normalize A matrix columns to maintain proper probabilities
    #     # self.agent.A[0][:, 0] = self.agent.A[0][:, 0] / np.sum(self.agent.A[0][:, 0])
    #     # self.agent.A[0][:, 1] = self.agent.A[0][:, 1] / np.sum(self.agent.A[0][:, 1])
        
    #     print(f"A matrix:\n{self.agent.A[0]}")
    #     print(f"qs: {self.qs}, joint observation: {joint_observation}, action: {action}")
