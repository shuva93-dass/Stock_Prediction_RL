from agent import Agent
from helper import getStockData, getState
import sys


window_size = 100                         
batch_size = 32
agent = Agent(window_size, batch_size)  
data = getStockData("training_data")
l = len(data)-1
episode_count = 30
#BUDGET
initial_money = 10000

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
        
        invest = ((starting_money - initial_money) / initial_money) #invest treated as reward
        if t == l - 1:
            done = True
        #print("next_state:",next_state)
        agent.step(action_prob, invest , next_state, starting_money < initial_money) #adding to the memory of experience tuples
        state = next_state

        if done:
            print("------------------------------------------")
            print("Train Total Profit: " + formatPrice(total_profit))
            print("------------------------------------------")
