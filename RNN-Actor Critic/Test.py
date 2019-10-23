# =============================================================================
# Testing the data (run this after training part)
#=============================================================================
test_data = getStockData("test_data")
l_test = len(test_data) - 1
state = getState(test_data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
agent.is_eval = False
done = False
for t in range(l_test):
   action = agent.act(state)
   next_state = getState(test_data, t + 1, window_size + 1)
   reward = 0
   if action == 1:  
       agent.inventory.append(test_data[t])
	   print('\033[94m'+"Buy the stock at Price:" + formatPrice(test_data[t]) + '\033[0m')
   elif action == 2 and len(agent.inventory) > 0: 
       bought_price = agent.inventory.pop(0)
       reward = max(test_data[t] - bought_price, 0)
       total_profit += test_data[t] - bought_price
	   print('\033[91m' +'Sell the stock at Price: '+ formatPrice(test_data[t]) + ' | '  + 'Updated Profit: ' + '\033[1m'+ formatPrice(test_data[t] - bought_price) + '\033[0m')
	else:
       print ('\033[1m' + 'Hold the Stock'  + '\033[0m')
   if t == l_test - 1:
       done = True
   agent.step(action_prob, reward, next_state, done)
   state = next_state

   if done:
       epi_end = datetime.datetime.now(pytz.timezone('America/Chicago')) 
       print('\033[1m' + '------------------------------------------' + '\033[0m')
       print('\033[1m' + 'Total Profit for  epoch in RNN: ' + formatPrice(total_profit) + '\033[0m')
       print ('\033[95m' + 'end test time for episode {} : {}:'.format(e,epi_end.strftime("%I:%M:%S %p"))+ '\033[0m')
       print('\033[1m' + '------------------------------------------' + '\033[0m')
            
            
            
        
test_end = datetime.datetime.now(pytz.timezone('America/Chicago')) 
print ('\033[95m'+ 'End time for testing:',test_end.strftime("%I:%M:%S %p")+ '\033[0m')
