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

    
class Agent:
    def __init__(self, state_size, batch_size, is_eval = False):
        self.state_size = state_size 
        self.action_size = 3  #buy,sell,hold
        
        #defining replay memory size
        self.buffer_size = 1000000
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.inventory = []
        
        #define wether or not training is going on
        self.is_eval = is_eval   
        #Discount factor
        self.gamma = 0.99 
        # soft update for AC model
        self.tau = 0.001
        
        #instantiate the local and target actor models for soft updates
        self.actor_local = Actor(self.state_size, self.action_size) 
        self.actor_target = Actor(self.state_size, self.action_size)
        
        #critic model mapping state-action pairs with Q-values 
        self.critic_local = Critic(self.state_size, self.action_size)
        
        #instantiate the local and target critic models for soft updates
        self.critic_target = Critic(self.state_size, self.action_size)    
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        
        #set target model parameter to local model parameters 
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
    #Returns an action given a state using policy(actor)network
    def act(self, state):
        options = self.actor_local.model.predict(state) #returns probabilities of each action
        self.last_state = state
        if not self.is_eval:
            return choice(range(3), p = options[0])     
        return np.argmax(options[0])
    
    #method to return set of actions carried out by agent at every  step of episode
    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state,done) 
        if len(self.memory) > self.batch_size:   
            experiences = self.memory.sample(self.batch_size) #sampling random batch from memory to train
            self.learn(experiences) 
            self.last_state = next_state   
    
