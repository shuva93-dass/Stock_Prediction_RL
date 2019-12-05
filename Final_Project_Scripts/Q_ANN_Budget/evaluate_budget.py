#****************************************************
#TRAINING
#****************************************************
from agent import *
from helper import *
import sys
import datetime
import pytz
import os
path = "/content/drive/My Drive/Colab Notebooks/QANNBModelss/"
#os.mkdir(path)

#stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

window_size = 100
episode_count = 15
agent = Agent(window_size)
data = helper.getStockData("training_data")
l = len(data) - 1
batch_size = 32
half_window = window_size // 2
initial_money = 10000
starting_money = initial_money

train_start = datetime.datetime.now(pytz.timezone('America/Chicago'))
print ('\033[95m'+'Start time for training:',train_start.strftime("%I:%M:%S %p") + '\033[0m')

f = open("train_profit.txt", "a+")
f.write('********************\n Model : Q Learning with ANN with budget\n Total Episode Count : '+ str(episode_count) +'\n Window Size : '+ str(window_size) + '\n Budget : '+ str(initial_money)+'\n ******************** \n')
f.close()

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    print (state)
    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1 and starting_money >= data[t] and t < (len(data) - half_window): # buy
            agent.inventory.append(data[t])
            starting_money -= data[t]
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            starting_money += data[t]
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        else:
            print("Hold the Stock")
        invest = ((starting_money - initial_money) / initial_money)
        done = True if t == l - 1 else False
        agent.memory.append((state, action, invest, next_state, starting_money < initial_money))
        state = next_state

        if done:
            epi_end = datetime.datetime.now(pytz.timezone('America/Chicago'))
            f = open("train_profit.txt", "a+")
            f.write('for episode '+ str(e) +'end train time:'+epi_end.strftime("%I:%M:%S %p")+ 'Total Profit : ' + formatPrice(total_profit)+'\n')
            f.close()
            print("--------------------------------")
            #print(")
            print ('Train Total Profit: ' + formatPrice(total_profit)+'\033[95m' + 'end train time for episode {} : {}:'.format(e,epi_end.strftime("%I:%M:%S %p"))+ '\033[0m')
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    
    agent.model.save("/content/drive/My Drive/Colab Notebooks//QANNBModelss/model_ep" + str(e)+'.h5')
train_end = datetime.datetime.now(pytz.timezone('America/Chicago'))
print ('\033[95m'+ 'End time for training:',train_end.strftime("%I:%M:%S %p")+ '\033[0m')

#****************************************************
# Testing 
#****************************************************

import keras
from keras.models import load_model

#from agent import Agent
#from functions import *
import sys
#### ---->  Do This: pass the last model name you got from your training in load_model
model = load_model("/content/drive/My Drive/Colab Notebooks/QANNBModelss/" + "model_ep1.h5")
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True)#, model_name)
test_data = helper.getStockData("testing_data")
l = len(test_data) - 1
batch_size = 32
initial_money = 10000
starting_money = initial_money

state = getState(test_data, 1, window_size + 1)
print (state)
total_profit = 0
agent.inventory = []

for t in range(l):
  action = agent.act(state)
  print(action)
	# sit
  next_state = getState(data, t + 1, window_size + 1)
  #reward = 0
  if action == 1 and initial_money >= test_data[t] and t < (test_data - half_window): # buy
    agent.inventory.append(test_data[t])
    starting_money -= test_data[t]
    print("Buy: " + formatPrice(test_data[t]))
  elif action == 2 and len(agent.inventory) > 0: # sell
    bought_price = agent.inventory.pop(0)
		#reward = max(data[t] - bought_price, 0)
    total_profit += data[t] - bought_price
    starting_money += test_data[t]
    print("Sell: " + formatPrice(test_data[t]) + " | Profit: " + formatPrice(test_data[t] - bought_price))

  else:
     print(action)
     print("Hold the Stock")
  invest = ((starting_money - initial_money) / initial_money)
  done = True if t == l - 1 else False
  agent.memory.append((state, action, invest, next_state, starting_money < initial_money))
  state = next_state

  if done:
    f = open("train_profit.txt", "a+")
    f.write('--------------------------------\nTest Total Profit : ' + formatPrice(total_profit)+'\n--------------------------------')
    f.close()
    print("--------------------------------")
    print("Test Total Profit : " + formatPrice(total_profit))
    print("--------------------------------")
    
    
    
#****************************************************
# Budget
#****************************************************

#BUDGET
def buy(initial_money,dataset):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        state = getState(dataset, 0, window_size + 1)
        l = len(dataset) - 1
        for t in range(l):
            action = agent.act(state)
            next_state = getState(dataset,t + 1,window_size + 1)
            
            if action == 1 and initial_money >= dataset[t] and t < (test_data - half_window):
                inventory.append(dataset[t])
                initial_money -= dataset[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f'% (t, dataset[t], initial_money))
            
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += dataset[t]
                states_sell.append(t)
                try:
                    invest = ((dataset[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, dataset[t], invest, initial_money)
                )
            else:
                print('day %d: hold the stock, total balance %f'% ( t, initial_money))
            state = next_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest
        
states_buy, states_sell, total_gains, invest = buy(initial_money,test_data)
   
#****************************************************
# Plotting
#****************************************************
  
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (15,5))
plt.plot(test_data, color='r', lw=2.)
plt.plot(test_data, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(test_data, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
plt.legend()
#plt.savefig('output/'+name+'.png')  Code for saving the graph
plt.show()
   
