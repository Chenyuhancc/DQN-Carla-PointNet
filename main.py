import wandb
from model import DQNAgent
from car import CarEnv
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import datetime
import math
import numpy as np
from threading import Thread
from jupyterplot import ProgressPlot
import tensorflow as tf
from plotnine import *
import pandas as pd
from collections import deque
import datetime
import math
import time
import cv2 as cv
import random
import sys
import glob
import os
import carla


def save_every_5k(name, n):
    df.to_csv('{}_{}.csv'.format(name, n))
    agent.save_model(name+'_'+str(n))

def train():
    SECONDS_PER_EPISODE = 100
    REPLAY_MEMORY_SIZE = 5_000
    MIN_REPLAY_MEMORY_SIZE = 2048
    MINIBATCH_SIZE = 2048
    PREDICTION_BATCH_SIZE = 1
    TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
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

    FPS = 60
    town = 'town03'
    ep_rewards = []
    ep = []
    avg = 0
    av_loss = 0
    avg_loss = []
    avg_reward = []
    Step = []
    Loss = []
    Explore = []
    Steer = []
    Epsilon = []
    random.seed(1)
    np.random.seed(1)
    steer_amt = 0.3
    sleepy = 0.3

    agent = DQNAgent(state_size, action_size, 240)
    load_episode = 0

    env = CarEnv()
    agent.train_in_loop(Loss)
    agent.get_qs(np.ones((1, 2048, state_size)))

    Loss = []
    with tqdm(total=EPISODES-load_episode) as pbar:
        for episode in range(EPISODES-load_episode):
            env.collision_hist = []
            episode_reward = 0
            loss = 0
            step = 1
            explore = 0
            steer_ = 0
            current_state = env.reset()
            done = False
            episode_start = time.time()
            while True:
                spectator = env.world.get_spectator()
                spectator_transform = env.vehicle.get_transform()
                spectator_transform.location += carla.Location(x=-2, y=0, z=2.0)
                spectator.set_transform(spectator_transform)
                if current_state.shape[0] == 0:
                    break
                rand = np.random.random()
                if rand > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                    new_state, reward, done, _ = env.step(action)
                    time.sleep(1/FPS)
                else:
                    action = np.random.randint(0, action_size)
                    new_state, reward, done, _ = env.step(action)
                    explore += 1
                    time.sleep(1/FPS)
                episode_reward += reward
                agent.update_replay_memory(
                    (current_state, action, reward, new_state, done))
                current_state = new_state
                step += 1
                if done:
                    break
            agent.train(Loss)

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            


            print('Episode :{}, Step :{}, Epsilon :{} ,Reward :{}, Explore_rate :{}, loss :{} ,Steer :{}'
                  .format(episode + load_episode, step, epsilon, episode_reward, explore / step, Loss[episode],
                      steer_ / step))
            wandb.log({'Reward': episode_reward})
            ep_rewards.append(episode_reward)
            ep.append(episode + load_episode)
            Step.append(step)
            Explore.append(explore)
            Steer.append(steer_ / step)
            Epsilon.append(epsilon)
            avg = ((avg * (episode + load_episode) + episode_reward) /
               (episode + load_episode + 1))
            avg_reward.append(avg)
            av_loss = ((av_loss * (episode + load_episode) +
                    Loss[episode]) / (episode + load_episode + 1))
            wandb.log({'loss': av_loss})
            avg_loss.append(av_loss)
          
 

if __name__ == '__main__':
    wandb.init(name='car', project="carla-rl")
    train()
