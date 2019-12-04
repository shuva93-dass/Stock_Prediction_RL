# Critic Script
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


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
       #LSTM Architecture
      # Define input layers 
        states=Input((self.state_size,1),name = 'states')
        actions=Input((self.action_size,1),name = 'actions')
        net_states = LSTM(16,kernel_regularizer=regularizers.l2(1e-6),return_sequences = True)(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        net_states = LSTM(32,kernel_regularizer=regularizers.l2(1e-6),return_sequences = True)(net_states)
        net_actions = LSTM(32,kernel_regularizer=regularizers.l2(1e-6))(actions)
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net) 
       
        Q_values = layers.Dense(units=1, name='q_values',kernel_initializer=initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        action_gradients = K.gradients(Q_values, actions)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
