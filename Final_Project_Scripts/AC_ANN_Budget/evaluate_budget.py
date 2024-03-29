# =============================================================================
# Training the data
# =============================================================================

import sys
import datetime
import pytz 
from helper import getStockData, getState

window_size = 100                         
batch_size = 32
agent = Agent(window_size, batch_size)  
data = getStockData("training_data")
l = len(data)-1
episode_count = 1
initial_money = 10000
train_start = datetime.datetime.now(pytz.timezone('America/Chicago'))
print ('\033[95m'+'Start time for training:',train_start.strftime("%I:%M:%S %p") + '\033[0m')
f = open("trainAC_profit.txt", "a+")
f.write('********************\n Model : Actor-Critc with ANN with budget\n Total Episode Count : '+ str(episode_count) +'\n Window Size : '+ str(window_size) + '\n Budget : '+ str(initial_money)+'\n ******************** \n')
f.close()
for e in range(episode_count):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    agent.inventory = []
    total_profit = 0
    starting_money = initial_money
    done = False

    for t in range(l):
        action = agent.act(state)
        print("action:",action)
        
        #state = np.reshape(state,(state.shape[0],state.shape[1],1))   
        action_prob = agent.actor_local.model.predict(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        
        if action == 1:
            agent.inventory.append(data[t])
            starting_money-=data[t]
            print("Buy:" + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            total_profit += data[t] - bought_price
            starting_money += data[t]
            print("sell: " + formatPrice(data[t]) + "| profit: " + 
              formatPrice(data[t] - bought_price))
        
        invest = ((starting_money - initial_money) / initial_money)
        if t == l - 1:
            done = True
        #print("next_state:",next_state)
        agent.step(action_prob, invest , next_state, starting_money < initial_money) #adding to the memory of experience tuples
        state = next_state

        if done:
            epi_end = datetime.datetime.now(pytz.timezone('America/Chicago'))
            f = open("trainAC_profit.txt", "a+")
            f.write('for episode '+ str(e) +'end train time:'+epi_end.strftime("%I:%M:%S %p")+ 'Total Profit : ' + formatPrice(total_profit)+'\n')
            f.close()
            print("------------------------------------------")
            print("Train Total Profit for episode "+ str(e) +'is: ' + formatPrice(total_profit))
            print ('\033[95m' + 'end train time for episode {} : {}:'.format(e,epi_end.strftime("%I:%M:%S %p"))+ '\033[0m')
            print("------------------------------------------")
                    
            
# =============================================================================
# Testing the data
# =============================================================================
test_data = getStockData("testing_data")
l_test = len(test_data) - 1
state = getState(test_data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
starting_money = initial_money
agent.is_eval = False
done = False
for t in range(l_test):
    action = agent.act(state)
    next_state = getState(test_data, t + 1, window_size + 1)
    reward = 0
    if action == 1:  
        agent.inventory.append(test_data[t])
        starting_money-=data[t]
        print("Buy: " + formatPrice(test_data[t]))
    elif action == 2 and len(agent.inventory) > 0: 
        bought_price = agent.inventory.pop(0)
        total_profit += test_data[t] - bought_price
        starting_money += test_data[t]
        print("Sell: " + formatPrice(test_data[t]) + " | profit: " + formatPrice(test_data[t] - bought_price))

    invest = ((starting_money - initial_money) / initial_money)
    if t == l_test - 1:
        done = True
    agent.step(action_prob, invest, next_state, starting_money < initial_money)
    state = next_state

    if done:
        f = open("trainAC_profit.txt", "a+")
        f.write('------------------------------------------\nTest Total Profit: ' + formatPrice(total_profit)+'\n------------------------------------------\n')
        f.close()
        print("------------------------------------------")
        print("Test Total Profit: " + formatPrice(total_profit))
        print("------------------------------------------")
        
        
# =============================================================================
# Including the Budget
# =============================================================================

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
            
            if action == 1 and initial_money >= dataset[t]:
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
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, dataset[t], invest, initial_money)
                )
            
            state = next_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest  
        
states_buy, states_sell, total_gains, invest = buy(initial_money,test_data)  #Evaluate with test data

# =============================================================================
# Plotting
# =============================================================================
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (15,5))
plt.plot(test_data, color='r', lw=2.)
plt.plot(test_data, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(test_data, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
plt.legend()
#plt.savefig('output/'+name+'.png')  Code for saving the graph
plt.show()

