import math
import os
import tensorflow as tf
from tensorflow.keras import layers, initializers, losses, optimizers
from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense
from keras.layers import Reshape, Lambda, concatenate
from keras.models import Sequential
from keras.engine.topology import Layer
from collections import deque
import numpy as np
import time
import random


SECONDS_PER_EPISODE = 100
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 2048
MINIBATCH_SIZE = 2048
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = 1
UPDATE_TARGET_EVERY = 5
MEMORY_FRACTION = 0.4
MIN_REWARD = -200
EPISODES = 32000
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.99975  # 0.9975 99975
MIN_EPSILON = 0.01
AGGREGATE_STATS_EVERY = 10
state_size = 3
action_size = 3

class MatMul(Layer):

    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`MatMul` layer should be called '
                             'on a list of inputs')
        if len(input_shape) != 2:
            raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('The dimensions of each element of inputs should be 3')

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatMul` layer should be called '
                             'on a list of inputs.')
        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)

class DQNAgent:
    
    def __init__(self, state_size, action_size, point):
        self.state_size = state_size
        self.action_size = action_size
        self.point = point
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        REPLAY_MEMORY_SIZE = 5_000
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0
        #self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def get_weight(self):

        w = self.model.get_weights()
        return w

    def predict(self, state):

        predict = self.model.predict(state.reshape((-1, 3)))
        return predict

    def save_model(self, name):
        self.model.save_weights("{}.h5".format(name))
        print("Saved model to disk")

    def create_model(self):
        input_points = Input(shape=(2048, 3))
        x = Conv1D(64, 1, activation='relu')(input_points)
        x = BatchNormalization()(x)
        x = Conv1D(128, 1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(1024, 1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2048)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
        input_T = Reshape((3, 3))(x)

        g = MatMul()([input_points, input_T])
        g = Conv1D(64, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Conv1D(64, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        f = Conv1D(64, 1, activation='relu')(g)
        f = BatchNormalization()(f)
        f = Conv1D(128, 1, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Conv1D(1024, 1, activation='relu')(f)
        f = BatchNormalization()(f)
        f = MaxPooling1D(pool_size=2048)(f)
        f = Dense(512, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Dense(256, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
        feature_T = Reshape((64, 64))(f)

        g = MatMul()([g, feature_T])
        g = Conv1D(64, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Conv1D(128, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Conv1D(1024, 1, activation='relu')(g)
        g = BatchNormalization()(g)

        global_feature = MaxPooling1D(pool_size=2048)(g)
        c = Dense(512, activation='relu')(global_feature)
        c = BatchNormalization()(c)
        c = Dropout(0.5)(c)
        c = Dense(256, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Dropout(0.5)(c)
        c = Dense(self.action_size, activation='softmax')(c)
        prediction = Flatten()(c)
        model = Model(inputs=input_points, outputs=prediction)
        model.summary()
        model.compile(optimizer=optimizers.Adam(lr=0.01),loss=losses.Huber())
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, Loss):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            self.terminate = True
            Loss.append(0)
            return
        minibatch = random.sample(self.replay_memory, 1)
        #minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        #states = np.array([transition[0] for transition in minibatch])
        for transition in minibatch:
             state = np.asarray(transition[0]).astype(np.float32)
        while state.shape[0] < 2048 and state.shape[0] != 1:
            temp = state.copy()
            state = np.concatenate((state, temp))

        while state.shape[1] < 2048 and state.shape[0] == 1:
            temp = state.copy()
            state = np.concatenate((state, temp),axis=1)
            state = np.reshape(state[:,:2048,:],(1,2048,3))

        while state.shape[0] > 2048 and state.shape[0] != 1:
            state = state[:2048,:]
        state = np.reshape(state[:2048,:],(-1,2048,3))

        while state.shape[1] > 2048 and state.shape[0] == 1:
            state = np.reshape(state[:,:2048,:],(-1,2048,3))
        current_states = state
        current_qs_list = self.model.predict(
            current_states, PREDICTION_BATCH_SIZE)

        for transition in minibatch:
             new_state = np.asarray(transition[3]).astype(np.float32)
        while new_state.shape[0] < 2048 and new_state.shape[0] != 1:
            temp = new_state.copy()
            new_state = np.concatenate((new_state, temp))

        while new_state.shape[1] < 2048 and new_state.shape[0] == 1:
            temp = new_state.copy()
            new_state = np.concatenate((new_state, temp),axis=1)
            new_state = np.reshape(new_state[:,:2048,:],(1,2048,3))

        while new_state.shape[0] > 2048 and new_state.shape[0] != 1:
            new_state = new_state[:2048,:]
        new_state = np.reshape(new_state[:2048,:],(-1,2048,3))

        while new_state.shape[1] > 2048 and new_state.shape[0] == 1:
            new_state = np.reshape(new_state[:,:2048,:],(-1,2048,3))
        new_current_states = new_state
        future_qs_list = self.target_model.predict(
            new_current_states, PREDICTION_BATCH_SIZE)
        X = []
        y = []
        for index, (state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            while state.shape[0] < 2048 and state.shape[0] != 1:
                temp = state.copy()
                state = np.concatenate((state, temp))

            while state.shape[1] < 2048 and state.shape[0] == 1:
                temp = state.copy()
                state = np.concatenate((state, temp),axis=1)
                state = np.reshape(state[:,:2048,:],(2048,3))

            while state.shape[0] > 2048 and state.shape[0] != 1:
                state = state[:2048,:]
            state = np.reshape(state[:2048,:],(2048,3))

            while state.shape[1] > 2048 and state.shape[0] == 1:
                state = np.reshape(state[:,:2048,:],(2048,3))
            X.append(state)
            y.append(current_qs)

        history = self.model.fit(np.array(X), np.array(
            y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)
        Loss.append(history.history['loss'][0])
        self.target_model.set_weights(self.model.get_weights())
        

    def get_qs(self, state):
        while state.shape[0] < 2048 and state.shape[0] != 1:
            temp = state.copy()
            state = np.concatenate((state, temp))

        while state.shape[1] < 2048 and state.shape[0] == 1:
            temp = state.copy()
            state = np.concatenate((state, temp),axis=1)
            state = np.reshape(state[:,:2048,:],(1,2048,3))

        while state.shape[0] > 2048 and state.shape[0] != 1:
            state = state[:2048,:]
        state = np.reshape(state[:2048,:],(-1,2048,3))

        while state.shape[1] > 2048 and state.shape[0] == 1:
            state = np.reshape(state[:,:2048,:],(-1,2048,3))
        return self.model.predict(state)[0]

    def train_in_loop(self, Loss):
        X = np.random.uniform(size=(2048, 3)).astype(np.float32)
        y = np.random.uniform(size=(self.action_size)).astype(np.float32)
        X = X[None, ...]
        y = y[None, ...]
        self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True
        print('Start Train')
        while True:
            if self.terminate:
                return
            self.train(Loss)
            time.sleep(0.01)
