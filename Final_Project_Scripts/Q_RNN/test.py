#TESTING
import keras
from keras.models import load_model

#from agent.agent import Agent
from functions import *
import sys

##  Do This: pass the last model name you got from your training
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data_test = df_test.iloc[:,4].tolist()
l = len(data_test) - 1
batch_size = 32

state = getState(data_test, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data_test, t + 1, window_size + 1)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data_test[t])
		print("Buy: " + formatPrice(data_test[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data_test[t] - bought_price, 0)
		total_profit += data_test[t] - bought_price
		print("Sell: " + formatPrice(data_test[t]) + " | Profit: " + formatPrice(data_test[t] - bought_price))
  else:
      print("Hold the Stock")
	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print("--------------------------------")
		print( " Test Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")

