# Actor Script
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import *
from keras.models import *
#actor class whose object takes in the parameters of the state size and action size
class Actor:
  # """Actor (policy) Model. """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()
        
   
    
    def build_model(self):
        
        #LSTM ARCHITECTURE
        states=Input((self.state_size,1,),name = 'states')
        net = LSTM(32*2**0,kernel_regularizer=layers.regularizers.l2(1e-6),return_sequences = True)(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = LSTM(32*2**1,kernel_regularizer=layers.regularizers.l2(1e-6),return_sequences = True)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = Flatten()(net)
        net = layers.Activation("relu")(net)
        actions = Dense(units=self.action_size, activation='softmax', name = 'actions')(net)
                
        self.model = models.Model(inputs=states, outputs=actions)
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        optimizer = optimizers.Adam(lr=.00001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
        #print(self.model.summary())
