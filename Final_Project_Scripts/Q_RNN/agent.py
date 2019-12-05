import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras import regularizers
from keras import initializers
from keras.layers import *
from keras.models import *
from keras import layers, models, optimizers
from keras import backend as K
import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model("/models/" + model_name) if is_eval else self._model()

    def _model(self):
        input_tensor = Input((self.state_size,1,))
        x = input_tensor
        x = LSTM(8,return_sequences = True)(x)
        x = LSTM(16,return_sequences = True)(x)
        #x = LSTM(32*2**2,return_sequences = True)(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(self.action_size, activation='linear', kernel_initializer = 'normal')(x)
        model = Model(inputs=input_tensor, outputs=x)
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])
    
    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
            #print("mini_batch")

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            #print('brfe if')
            if not done:
                #print('in if before target')
                target = reward + self.gamma * np.amax(self.model.predict(np.reshape(next_state,(next_state.shape[0],next_state.shape[1],1)))[0])
            #print('out if after target')
            target_f = self.model.predict(np.reshape(state,(state.shape[0],state.shape[1],1)))
            target_f[0][action] = target
            self.model.fit(np.reshape(state,(state.shape[0],state.shape[1],1)), target_f, epochs=1, verbose=0)
            #print('after if')
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
