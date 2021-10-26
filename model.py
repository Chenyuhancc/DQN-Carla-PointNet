import math
import os
import tensorflow as tf
from tensorflow.keras import layers, initializers, losses, optimizers
from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.xception import Xception
from keras.models import Sequential
from collections import deque
import numpy as np

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
        model_json = self.model.to_json()
        with open("{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("{}.h5".format(name))
        print("Saved model to disk")

    def create_model(self):
        inputs = tf.keras.Input(shape=(100000, 3))
        y = layers.Conv1D(64, 3, strides=1, padding='same')(inputs)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        y = layers.Conv1D(128, 3, strides=1, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        y = layers.Conv1D(1024, 3, strides=1, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        print('y0:', y.shape)
        y = layers.GlobalMaxPooling1D(data_format='channels_last')(y)
        print('y:', y.shape)
        y = tf.reshape(y, [-1,1024])
        y = layers.Dense(512)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        y = layers.Dense(256)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        y = layers.Dense(9)(y)
        temp = tf.convert_to_tensor(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))
        y = y + temp
        y = tf.reshape(y, [-1,3,3])
        x = tf.reshape(inputs, [-1,self.point,3])
        x = tf.linalg.matmul(x, y)
        x = tf.reshape(x, [-1,3,self.point])
        x = layers.Conv1D(64, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(128, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(1024, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalMaxPooling1D(data_format='channels_first')(x)
        x = tf.reshape(x, [-1,1024])
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(256)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        predictions = layers.Dense(self.action_size)(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(lr=0.01),loss=losses.Huber())
        model.summary()
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        global Loss
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            self.terminate = True
            Loss.append(0)
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(
            current_states, PREDICTION_BATCH_SIZE)
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(
            new_current_states, PREDICTION_BATCH_SIZE)
        X = []
        y = []
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        history = self.model.fit(np.array(X), np.array(
            y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)
        Loss.append(history.history['loss'][0])
        self.target_model.set_weights(self.model.get_weights())

    def get_qs(self, state):
        return self.model.predict(state.reshape((1, self.state_size)))[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, self.state_size)).astype(np.float32)
        y = np.random.uniform(size=(1, self.action_size)).astype(np.float32)
        print('X0:',X.shape,'y:',y.shape)
        X = X[None, ...]
        y = y[None, ...]
        print('X:',X.shape,'y:',y.shape)
        self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True
        print('Start Train')
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
