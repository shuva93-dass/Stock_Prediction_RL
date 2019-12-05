#TRAINING
import sys
import datetime
import pytz
import os
path = "/content/drive/My Drive/Colab Notebooks/RL projects/Modelss/"
os.mkdir(path)

#stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

window_size = 100
episode_count = 3
agent = Agent(window_size)
data = df.iloc[:,4].tolist()
l = len(data) - 1
batch_size = 32

train_start = datetime.datetime.now(pytz.timezone('America/Chicago'))
print ('\033[95m'+'Start time for training:',train_start.strftime("%I:%M:%S %p") + '\033[0m')
f = open("train_profit.txt", "a+")
f.write('Model : Q Learning with RNN (without budget)\n Total Episode Count : '+ str(episode_count) +'\n Window Size : '+ str(window_size) + '\n ')
f.close()
for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(np.reshape(state,(state.shape[0],state.shape[1],1)))
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1: # buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        else:
            print("Hold the Stock")
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            epi_end = datetime.datetime.now(pytz.timezone('America/Chicago'))
            f = open("train_profit.txt", "a+")
            f.write('for episode '+ str(e) +'end train time:'+epi_end.strftime("%I:%M:%S %p")+ 'Total Profit : ' + formatPrice(total_profit)+'\n')
            f.close()
            print("--------------------------------")
            print("Train Total Profit: " + formatPrice(total_profit))
            print ('\033[95m' + 'end train time for episode {} : {}:'.format(e,epi_end.strftime("%I:%M:%S %p"))+ '\033[0m')
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    
    agent.model.save("/content/drive/My Drive/Colab Notebooks/RL projects/Modelss/model_ep" + str(e)+'.h5')
train_end = datetime.datetime.now(pytz.timezone('America/Chicago'))
print ('\033[95m'+ 'End time for training:',train_end.strftime("%I:%M:%S %p")+ '\033[0m')
