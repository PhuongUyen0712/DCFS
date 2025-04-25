import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, LSTM, Flatten, Lambda, Subtract, Multiply, Add, concatenate, GRU
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import Model



class Agent:
    def __init__(self, internal_threshold, external_threshold, rnn_unit, env):
        self.csv_logger = CSVLogger('log.csv', append=True, separator=';')
        self.env = env
        self.input_dim = env.data.shape[1] - 1
        self.output_dim = env.data.shape[1] - 1
        self.rnn_unit = rnn_unit


        self.q_model = self.create_Q_model(forTraining=True) 
        self.predicting_q_model = self.create_Q_model(forTraining=False)
        self.target_model = self.create_Q_model(forTraining=False)
            
        self.full_model=self.create_full_model()
        self.full_model.summary()        
        
        self.internal_threshold = internal_threshold
        self.external_threshold = external_threshold

        
    def create_Q_model(self, forTraining=True):
        if forTraining == True:
            stateful = False
            batchSize = None
        else:
            stateful = True
            batchSize = 1

        model = Sequential()
        
        first_lstm = LSTM(self.rnn_unit,
                          batch_input_shape=(batchSize, None, self.input_dim),
                          kernel_initializer ='RandomNormal',
                          return_sequences=True,
                          activation='tanh',
                          stateful=stateful)
        

        model.add(first_lstm)
        model.add(Dense(self.output_dim, kernel_initializer='VarianceScaling', activation='linear', name='output'))

        return model

    def create_full_model(self):
        stateful = False
        batchSize = None
        input_state = Input((None, self.input_dim))
        action_id = Input((self.input_dim,1), dtype='uint8')
        predicted_Qvalues = self.q_model(input_state)
        
        leave_only_action_value = tf.one_hot(action_id, self.output_dim, axis=-1)
        leave_only_action_value = tf.squeeze(leave_only_action_value, axis=2)
        
        predicted_q_for_action = tf.reduce_sum(predicted_Qvalues * leave_only_action_value, axis=2, keepdims=True)
        return Model([input_state, action_id], predicted_q_for_action)
    
    
    def complie(self, l_r):
        adam = Adam(lr=l_r)
        self.full_model.compile(loss='logcosh', metrics=['mse'],
                           optimizer=adam)

    def act(self, state, policy_calc, eps, selected_actions):
        available_actions = np.ones(self.input_dim) - selected_actions
        state = np.array(state)
        state = state.reshape(1, 1, state.shape[0])
        Q_values = self.predicting_q_model.predict(state).flatten()
        avilable_actions_Q = available_actions * Q_values
        avilable_actions_Q_threshold = avilable_actions_Q - (selected_actions * 999)
        available_actions_unif_prob = available_actions / sum(available_actions)

        if policy_calc == 0:
            if np.random.rand() < eps:
                a = np.random.choice(self.output_dim, 1, p=available_actions_unif_prob)[0]
                a_Q = self.internal_threshold
            else:
                if len(avilable_actions_Q_threshold[avilable_actions_Q_threshold > self.internal_threshold]) > 0:
                    a = np.argmax(avilable_actions_Q_threshold)
                    a_Q = Q_values[a]
                else:
                    a = -999
                    a_Q = -999
        else:
            if len(avilable_actions_Q_threshold[avilable_actions_Q_threshold > self.external_threshold]) > 0:
                a = np.argmax(avilable_actions_Q_threshold)
                a_Q = Q_values[a]
            else:
                a = -999
                a_Q = -999

        return a, a_Q,

    def is_done(self, a_Q, selected_actions, policy_calc=0):  ### need to fix
        done = 0
        if policy_calc == 0:
            if a_Q < self.internal_threshold or sum(selected_actions == 1) == self.input_dim:
                done = 1
        else:
             if a_Q < self.external_threshold or sum(selected_actions == 1) == self.input_dim:
                done = 1           
        return done

    def train(self, X_batch, y_batch, batch_size):
        print("model train")
#         return self.model.train_on_batch(X_batch, y_batch)
        return self.full_model.fit(X_batch, y_batch, epochs=5, steps_per_epoch=1, batch_size=batch_size, verbose=0)#, callbacks=

    def update_target_network(self):
        self.target_model.set_weights(self.q_model.get_weights())

    def predict_target_network(self, X_batch):
        return self.target_model.predict(X_batch)
    
    def updae_model_for_infereance_weights(self):
        self.predicting_q_model.set_weights(self.q_model.get_weights())

    def clear_inference_model_state(self):
        self.predicting_q_model.reset_states()