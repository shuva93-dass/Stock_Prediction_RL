# Agent Script
from actor import Actor
from critic import Critic
import numpy as np
from numpy.random import choice
import random
from collections import namedtuple, deque

class ReplayBuffer:
    #Fixed sized buffer to store experience tuples
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen = buffer_size)  #memory size of replay buffer
        self.batch_size = batch_size               #Training batch size for Neural nets
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])                                           #Tuple containing experienced replay

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done) #create an exp tuple
        self.memory.append(e) #add it to memory
        
    def sample(self, batch_size = 32):
        return random.sample(self.memory, k=self.batch_size) #sampling 32 experiences at a time

    def __len__(self):
        return len(self.memory)  # counts the number of experience tuples stored
