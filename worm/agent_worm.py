from typing import Tuple
import numpy as np
from pymdp import utils
from pymdp.agent import Agent

class ActiveInferenceWormAgent:
    pass

class SimpleHomeostaticAgent(ActiveInferenceWormAgent):
    """Agent implementing active inference for worm control using joint observations"""
    
    def __init__(self, A_matrix=None):
        # Define model dimensions
        self.num_obs_joint = 4  # (weird_smell,noci)/(no-weird-smell,noci)/(weird_smell,no-noci)/(no-weird-smell,no-noci)
        self.num_states = 2     # safe/harmful
        self.num_controls = 2   # stay/retreat
        self.A_matrix = A_matrix

        # Initialize matrices
        if self.A_matrix is not None:
            self.A_array = self.A_matrix
        else:
            self.A_array = utils.obj_array_zeros([
                (self.num_obs_joint, self.num_states)
            ])
        
            # A[observation, state]
            # States: 0=safe, 1=harmful
            # Joint observations (encoding: 0=stimulus present, 1=stimulus absent):
            # 0 = smell present, noci present (most dangerous)
            # 1 = smell absent, noci present  
            # 2 = smell present, noci absent
            # 3 = smell absent, noci absent (safest)

            # Initialize with reasonable priors:
            # In safe state (0):
            self.A_array[0][:, 0] = [0.0,  # joint_obs 0: smell + noci (impossible in safe state)
                                    0.0,   # joint_obs 1: no smell + noci (impossible in safe state)
                                    0.5,   # joint_obs 2: smell + no noci (possible in safe state)
                                    0.5]   # joint_obs 3: no smell + no noci (possible in safe state)

            # In harmful state (1):
            self.A_array[0][:, 1] = [0.25,  # joint_obs 0: smell + noci (likely in harmful state)
                                    0.25,   # joint_obs 1: no smell + noci (likely in harmful state)
                                    0.25,   # joint_obs 2: smell + no noci (possible in harmful state)
                                    0.25]   # joint_obs 3: no smell + no noci (possible in harmful state)

        # Keep same B matrix structure
        self.B_array = utils.obj_array_zeros([
            (self.num_states, self.num_states, self.num_controls)
        ])
        self.B_array[0][:, :, 0] = np.eye(2)  # stay action keeps state
        self.B_array[0][:, :, 1] = [[1, .8],  # retreat action tends toward safe
                                   [0, .2]]

        # Preferences over joint observations
        self.C_vector = utils.obj_array_zeros([
            (self.num_obs_joint,),
            #(self.num_states,)
        ])
        # Prefer no pain and neither nociception nor weird_smell technically matter, 
        # though in practice there is a strong preference induced for no nociception
        self.C_vector[0] = np.array([0.5, 0.5, 0.5, 0.5])  # No preferences
        #self.C_vector[1] = np.array([1.0, 0])  # Prefer safe state

        # Keep same action preferences
        self.E_matrix = np.array([0.8, 0.2])  # prefer staying to retreating

        # Initialize agent and beliefs
        self.agent = Agent(A=self.A_array, B=self.B_array, C=self.C_vector, E=self.E_matrix)
        self.qs = utils.obj_array_uniform([(self.num_states,)])

    def infer(self, observation: Tuple[int, int]) -> Tuple[int, np.ndarray]:
        """Update agent beliefs and get action"""

        noci_observation, weird_smell_observation = observation # REMEMBER: 0 = weird_smell/noci, 1 = no weird_smell/noci
        if (noci_observation, weird_smell_observation) == (0, 0):  #  noci, weird_smell
            joint_observation = 0
        elif (noci_observation, weird_smell_observation) == (0, 1): # noci, no weird_smell
            joint_observation = 1
        elif (noci_observation, weird_smell_observation) == (1, 0): # no noci, weird_smell
            joint_observation = 2
        elif (noci_observation, weird_smell_observation) == (1, 1): # no noci, no weird_smell
            joint_observation = 3
        else:
            raise ValueError("Invalid observation")

        # sample pain observation directly from the agent's state
        # Sample pain observation based on the probability of being in safe state
        # pain_observation = np.random.choice([0, 1], p=[self.qs[0][0], self.qs[0][1]])  # sample joint observation based on the probability of being in safe state

        # Update beliefs
        self.qs = self.agent.infer_states([joint_observation])#, pain_observation])

        # Get action
        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()[0]

        return action, self.qs
    
    # def learn(self, history, learning_rate):
    #     pass
    

class SimpleLearningAgent(SimpleHomeostaticAgent):
    """Agent implementing active inference for worm control using joint observations"""
    

    def learn(self, history, learning_rate, pseudocount=0.01): #TODO: accumulate information from history and figure out how to update A matrix based on sum of episodic experience
        # history is a list of tuples [(phys_state, qs, action), ...]
        for phys_state, qs, action in history:
            
            # Update A matrix - now only need to update one matrix
            safe_state_value = qs[0][0]
            harmful_state_value = qs[0][1]
            
            # Create one-hot vector for the observation
            update_vector = np.zeros(self.num_obs_joint)
            weird_smell, noci = phys_state.weird_smell, phys_state.noci
            joint_observation = 0 if (weird_smell, noci) == (False, False) else 1 if (weird_smell, noci) == (False, True) else 2 if (weird_smell, noci) == (True, False) else 3
            update_vector[joint_observation] = 1.0
            # update_vector += np.array([.5, .5, 0, 0]) if noci_observation == 0 else np.array([0, 0, .5, .5])
            # update_vector += np.array([.5, 0, .5, 0]) if weird_smell_observation == 0 else np.array([0, .5, 0, .5])
            
            # Update A matrix for both states with pseudocounts to prevent collapse
            self.agent.A[0][:, 0] += update_vector * safe_state_value * learning_rate + pseudocount
            self.agent.A[0][:, 1] += update_vector * harmful_state_value * learning_rate + pseudocount

            # No matter what, the probability of noci in safe state should be zero
            # self.agent.A[0][0, 0] = 0.0
            # self.agent.A[0][1, 0] = 0.0
            
            # # Normalize A matrix columns to maintain proper probabilities
            # self.agent.A[0][:, 0] = self.agent.A[0][:, 0] / np.sum(self.agent.A[0][:, 0])
            # self.agent.A[0][:, 1] = self.agent.A[0][:, 1] / np.sum(self.agent.A[0][:, 1])
            
        print(f"A matrix:\n{self.agent.A[0]}")
