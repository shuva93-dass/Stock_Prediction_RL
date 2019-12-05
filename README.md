# Stock_Prediction_RL
# Recurrent (Dropped) 
This folder has 2 components
1. Recurrent_Graphs.py - converts a time series stock data to recurrent graphs for input to CNN 
2. recc.png - recurrent graph image corresponding to the train data for over a time of 4 years.


# RNN-Actor Critic

The finalized architectire is RNN

This folder has 6 scripts:
1. Actor.py: It returns the best action or a policy that refers to a probability distribution over actions.
2. Critic.py: The critic evaluates the actions returned by the actor-network.
3. Agent.py: Train an agent to perform reinforcement learning based on the actor and critic networks
4. Helper.py: Create functions like formating the stock price and get a state vector that will be helpful for training,
5. Train.py: Train the agent, based on our training data and helper methods.
6. Test.py : Test the trained agent on testing data.


# DataSet
This folder contains two CSV files 
1. training_data.csv - contains stock data from January 2014 to August 2018 with columns Date, Open, High, Low, Close, Adj Close and Volume.
2. testing_data.csv - contains stock data from September 2018 to September 2019 with columns Date, Open, High, Low, Close, Adj Close and Volume.
3. The CSV files include following fields:
   1. Date: Indicates the date under consideration.
   2. Open: Indicates the price at which the market opens on that particular date.
   3. High: Indicates the highest market price on that date.
   4. Low: Indicates the lowest market price on that date.
   5. Close: Indicates the price at which the market closes on the date, adjusted for the split.
   6. Adj Close: Indicates the adjusted closing price for both the split and dividends.
   7. Volume: Indicates the total volume of shares available.

# Order of execution
1. Actor.py --> Critic.py --> Agent.py --> Helper.py --> Train.py --> Test.py 
2. Make sure all the above scripts and both the training and testing csvs are in the same folder before starting execution. 

# Final Project Scripts
This folder consists of 3 main models with modifications:
1. AC_ANN_Budget: This stands for Actor-Critic ANN implementation with Budget.It consists of 5 scripts.
   1. actor , critic , agent and helper scripts perform the same functionality as mentioned above.
   2. evaluate_budget.py : It includes the training and testing portion embedded with budget factor.
   
2. Q_ANN_Budget:This stands for Deep Q learning ANN with Budget model. It has 3 scripts
   1. agent.py : This script has the deep neural network which generates the Q values for all the action and then this script generates the policy
   2. helper.py : same functionality as mentioned before in AC model
   3. evaluate_budget.py : This script has the training, testing and Budget code
   
3. Q_RNN: This stands for Q-Learning RNN model. It consists of 4 scripts.
   1. agent, helper scripts perform the same functionality as mentioned in the Q_ANN_Budget model.
   2. train and test perform the same functionality as mentioned above in AC_ANN model.
