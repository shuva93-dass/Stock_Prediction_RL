# Stock_Prediction_RL
# Recurrent (Dropped) 
This folder has 2 components
1. Recurrent_Graphs.py - converts a time series stock data to recurrent graphs for input to CNN 
2. recc.png - recurrent graph image corresponding to the train data for over a time of 9 years.

The finalized architectire is RNN

# RNN-Actor Critic
This folder has 5 scripts:
1. Actor.py: It returns the best action or a policy that refers to a probability distribution over actions.
2. Critic.py: The critic evaluates the actions returned by the actor-network.
3. Agent.py: Train an agent to perform reinforcement learning based on the actor and critic networks
4. Helper.py: Create functions like formating the stock price and get a state vector that will be helpful for training,
5. Training.py: Train the data, based on our agent and helper methods.
