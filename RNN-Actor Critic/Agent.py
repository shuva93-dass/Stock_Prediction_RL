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
    
    def learn(self, experiences):   
        #Extracting the states,actions,etc from all the experience tuples
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)    
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size) 

        
        #Reshaping all the vectors into 3-dimensional vector to be fed into LSTM architecture
        states = np.reshape(states, (states.shape[0],states.shape[1],1))
        next_states = np.reshape(next_states, (next_states.shape[0],next_states.shape[1],1))
        rewards = np.reshape(rewards, (rewards.shape[0],rewards.shape[1],1))
        dones = np.reshape(dones, (dones.shape[0],dones.shape[1],1))
        actions = np.reshape(actions, (actions.shape[0],actions.shape[1],1))
        
        #return a separate array for each exp and predict actions based on next states
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        
        #Reshaping the vector
        actions_next = np.reshape(actions_next,(actions_next.shape[0],actions_next.shape[1],1))
       
        #predict qvalues for actor o/p for the next state
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next]) 
        
        #target the q-value to serve as label for critic model based on temporal diff
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        
        #fit the critic model to the time difference of the target
        Q_targets = np.reshape(Q_targets, (Q_targets.shape[0],Q_targets.shape[1],1))        
        self.critic_local.model.train_on_batch(x = [states, actions], y = Q_targets) 
        
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),(-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1]) 
        self.soft_update(self.actor_local.model, self.actor_target.model)
        
    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights)
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

