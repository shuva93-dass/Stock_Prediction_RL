#Train script
import sys
import datetime
import pytz 

window_size = 100                         
batch_size = 32
agent = Agent(window_size, batch_size)  
data = df.iloc[:,4].tolist()
#print('daata',len(data))
l = len(data) -1
#l = l-1
episode_count = 300

train_start = datetime.datetime.now(pytz.timezone('America/Chicago')) 

print ('\033[95m'+'Start time for training:',train_start.strftime("%I:%M:%S %p"))

for e in range(episode_count):
  
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size+1)
    agent.inventory = []
    total_profit = 0
    done = False

    for t in range(l):
        action = agent.act(np.reshape(state,(state.shape[0],state.shape[1],1)))
        print("action:",action)
        
        state = np.reshape(state,(state.shape[0],state.shape[1],1))   
        action_prob = agent.actor_local.model.predict(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        
        if action == 1:
            agent.inventory.append(data[t])
            print('\033[94m'+"Buy the stock at Price:" + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print('\033[91m' +'Sell the stock at Price: '+ formatPrice(data[t]) + ' | '  + 'Updated Profit: ' + '\033[1m'+ formatPrice(data[t] - bought_price))
        else:
            print ('\033[1m' + 'Hold the Stock')

        if t == l - 1:
            done = True
        agent.step(action_prob, reward, next_state, done) #adding to the memory of experience tuples
        state = next_state

        if done:
            epi_end = datetime.datetime.now(pytz.timezone('America/Chicago')) 
            print('\033[1m' + '------------------------------------------')
            print('\033[1m' + 'Total Profit for  epoch in RNN: ' + formatPrice(total_profit))
            print ('\033[95m' + 'end train time for episode {} : {}:'.format(e,epi_end.strftime("%I:%M:%S %p")))
            print('\033[1m' + '------------------------------------------')
            
            
            
        
train_end = datetime.datetime.now(pytz.timezone('America/Chicago'))
print ('\033[95m'+ 'End time for training:',train_end.strftime("%I:%M:%S %p"))
