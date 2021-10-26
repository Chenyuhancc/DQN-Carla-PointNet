import wandb
from model import DQNAgent
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
from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.xception import Xception
from keras.models import Sequential
from collections import deque
import datetime
import math
import time
import cv2 as cv
import random
import sys
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    sys.path.append(glob.glob('/home/chankahou/Downloads/CARLA_0.9.11/PythonAPI//carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    )
except IndexError:
    print("EGG not found")
pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla


def xxx():
    print('run here')
    for x in list(env.world.get_actors()):
        if 'vehicle' in x.type_id or 'sensor' in x.type_id:
            x.destroy()


def lidar_line(points, degree, width):
    angle = degree*(2*np.pi)/360
    points_l = points
    points_l = points_l[np.logical_and(
        points_l[:, 2] > -1.75, points_l[:, 2] < 1000)]  # z
    points_l = points_l[np.logical_and(np.tan(angle)*points_l[:, 0]+width*np.sqrt(1+np.tan(angle)**2) >=
                                       points_l[:, 1], np.tan(angle)*points_l[:, 0]-width*np.sqrt(1+np.tan(angle)**2) <= points_l[:, 1])]  # y
    if 180 > degree > 0:
        points_l = points_l[np.logical_and(
            points_l[:, 1] > 0, points_l[:, 1] < 1000)]  # y>0
    if 180 < degree < 360:
        points_l = points_l[np.logical_and(
            points_l[:, 1] < 0, points_l[:, 1] > -1000)]  # x
    if degree == 0 or degree == 360:
        points_l = points_l[np.logical_and(
            points_l[:, 0] > 0, points_l[:, 0] < 1000)]  # x
    if degree == 180:
        points_l = points_l[np.logical_and(
            points_l[:, 0] > -1000, points_l[:, 0] < 0)]
    return points_l

def save_every_n_episode(num_of_episode):
    n = cum = avg = 0
    avg_loss = []
    for i in Loss[1:]:
        n += 1
        cum += i
        avg = cum/n
        avg_loss.append(avg)

    df = pd.DataFrame({'Episode': ep, 'Reward': ep_rewards,
                       'avg_reward': avg_reward, 'Step': Step, 'Loss': Loss[1:],
                       'avg_loss': avg_loss,
                       'Explore': Explore, 'PCT_Explore': np.array(Explore)/np.array(Step)*100,
                       'Epsilon': Epsilon, 'Dist_stop': Dist_stop,
                       'Max_accel': Max_accel, 'Avg_accel': Avg_accel})
    if LOAD == True:
        df = pd.concat([df_load, df], ignore_index=True)

    name = 'full_deep_brake_cont_reward_limit_action@' + \
        str(num_of_episode)  # INSERT FILE NAME
    n = datetime.datetime.now()
    n = n.strftime('_%m%d%y_%H%M')

    file_path = "data/"
    df.to_csv(file_path+'{}.csv'.format(name))
    agent.save_model(file_path+name)

class CarEnv:
    global town
    actor_list = []
    collision_hist = []
    pt_cloud = []
    pt_cloud_filtered = []
    STEER_AMT = 1.0
    def __init__(self):
        self.high_res_capture = False
        if self.high_res_capture:
            self.IM_HEIGHT = 240
        else:
            self.IM_HEIGHT = 84
        self.IM_WIDTH = 84*2
        self.recording = True
        self.steer_ = 0
        try:
            self.client = carla.Client('202.175.25.142', 2000)
            self.client.set_timeout(10.0)
            #world = self.client.load_world(town)
            self.world = self.client.get_world()
            #spectator = self.world.get_spectator()
            #spectator.set_transform(carla.Transform(carla.Location(249, -120, 3), carla.Rotation(yaw=-90)))
            for x in list(self.world.get_actors()):
                if 'vehicle' in x.type_id or 'sensor' in x.type_id:
                    x.destroy()
            blueprint_library = self.world.get_blueprint_library()
            self.Isetta = blueprint_library.filter('model3')[0]
            spawn_point = random.choice(self.world.get_map().get_spawn_points())
            #spawn_point = carla.Transform(carla.Location(x=-115.4, y=4.0, z=1), carla.Rotation(pitch=0, yaw=180, roll=0))
            self.vehicle = self.world.spawn_actor(self.Isetta, spawn_point)
            self.place = 0
            self.start_position = self.get_position()
            #spectator = self.world.get_spectator()
            #spectator_transform = self.vehicle.get_transform()
            #spectator_transform.location += carla.Location(x = -2, y=0, z = 2.0)
            #spectator.set_transform(spectator_transform)
            #spectator.set_transform(carla.Transform(spectator_transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
            #time.sleep(0.1)
        except RuntimeError:
            print("Init phase failed, check server connection. Retrying in 30s")
            time.sleep(30)
        #self.collision_hist = []
        self.collision_hist = []
        self.actor_list = []
        self.pt_cloud = []
        self.pt_cloud_filtered = []
        '''
        if self.place == 0:
            transform = carla.Transform(carla.Location(
                249, -130, 0.1), carla.Rotation(0, -90, 0))
        else:
            transform = carla.Transform(carla.Location(
                self.lo_x, self.lo_y), carla.Rotation(0, -90, 0))
        '''

        self.actor_list.append(self.vehicle)
        self.lidar_sensor = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_sensor.set_attribute('points_per_second', '100000')
        self.lidar_sensor.set_attribute('channels', '64')
        self.lidar_sensor.set_attribute('range', '10000')
        self.lidar_sensor.set_attribute('upper_fov', '10')
        self.lidar_sensor.set_attribute('lower_fov', '-10')
        self.lidar_sensor.set_attribute('rotation_frequency', '60')
        transform = carla.Transform(carla.Location(x=0, z=1.9))
        time.sleep(0.01)
        self.sensor = self.world.spawn_actor(
            self.lidar_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_lidar(data))

        cam_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: self.process_img(data))

        cam_bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.s_camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.s_camera)
        self.s_camera.listen(lambda data: self.process_img_semantic(data))

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=1.0, brake=0.0))
        self.episode_start = time.time()
        time.sleep(0.4)
        transform2 = carla.Transform(carla.Location(x=2.5, z=0.7))
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(
            colsensor, transform2, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

    def set_location(self, x, y):
        self.lo_x, self.lo_y = x, y
        self.place = x, y

    def get_position(self):
        return self.vehicle.get_location()

    def Black_screen(self):
        settings = self.world.get_settings()
        settings.no_rendering_mode = True
        self.world.apply_settings(settings)

    def get_fps():
        world_snapshot = self.world.get_snapshot()
        fps = 1/world_snapshot.timestamp.delta_seconds
        return fps

    def get_speed(self):    # m/s
        velocity = self.vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def reset(self):
        self.flag = 0
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        #spawn_point = carla.Transform(carla.Location(x=-115.4, y=4.0, z=1), carla.Rotation(pitch=0, yaw=180, roll=0))
        #self.vehicle = self.world.spawn_actor(self.Isetta, spawn_point)
        #self.transform = carla.Transform(carla.Location(249, -130, 0.1), carla.Rotation(0, -90, 0))
        #self.vehicle = self.world.spawn_actor(self.Isetta, self.transform)
        self.vehicle.set_location(spawn_point.location)#self.start_position)
        #print('carLocation1', self.vehicle.get_location())
        #spectator = self.world.get_spectator()
        #spectator_transform = self.vehicle.get_transform()
        #spectator_transform.location += carla.Location(x = -2, y=0, z = 2.0)
        #spectator.set_transform(spectator_transform)
            #spectator.set_transform(carla.Transform(spectator_transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        #time.sleep(0.1)
        self.flag = 1
        # while self.distance_to_obstacle_f is None
        while (not hasattr(self,'distance_to_obstacle_f') or
          self.distance_to_obstacle_f is None or
          not hasattr(self,'distance_to_obstacle_r') or
          self.distance_to_obstacle_r is None or
          not hasattr(self,'distance_to_obstacle_l') or
          self.distance_to_obstacle_l is None):
            time.sleep(0.01)
        #print('carLocation1', self.vehicle.get_location())
        #spectator = self.world.get_spectator()
        #spectator_transform = self.vehicle.get_transform()
        #spectator_transform.location += carla.Location(x = -2, y=0, z = 2.0)
        #spectator.set_transform(spectator_transform)
            #spectator.set_transform(carla.Transform(spectator_transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        #time.sleep(0.1)
        self.episode_start = time.time()
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=1.0, brake=0.0))
        xx = self.distance_to_obstacle_f
        yy = self.distance_to_obstacle_r
        zz = self.distance_to_obstacle_l
        state_ = np.array(self.points)
        return state_

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, raw_image):
        self.image = np.array(raw_image.raw_data)
        self.image = self.image.reshape(
            (self.IM_HEIGHT, self.IM_WIDTH, 4))  # RGBA
        self.image = cv.cvtColor(self.image, cv.COLOR_RGBA2RGB)
        if self.high_res_capture and self.recording:
            self.image_array.append(image)
        cv.imshow('image', cv.resize(self.image, (500, 500)))
        cv.waitKey(1)
        # trim the top part
        self.image = self.image[int(self.IM_HEIGHT//2.4)::]
        return

    def process_img_semantic(self, data):
        i = np.array(data.raw_data)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))  # RGBA
        i2 = i2[int(self.IM_HEIGHT//2.4)::] # trim the top part
        self.semantic_image = i2[:, :, 2]
        return

    def process_lidar(self, raw):
        points_new = []
        points = np.frombuffer(raw.raw_data, dtype=np.dtype('f4'))
        for i in range(points.shape[0]//4):
            points_new.append(points[4*i])
            points_new.append(points[4*i+1])
            points_new.append(points[4*i+2])
        points_new = np.asarray(points_new)
        self.points = np.reshape(
            points_new, (int(points_new.shape[0] / 3), 3))*np.array([1, -1, -1])
        if self.points.shape[0] < 2048:
            temp = self.points.copy()
            self.points = np.concatenate((self.points, temp))
        self.points = self.points[:2048,:]
        lidar_f = lidar_line(self.points, 90, 2)
        lidar_r = lidar_line(self.points, 45, 2)
        lidar_l = lidar_line(self.points, 135, 2)

        if len(lidar_f) == 0:
            pass
        else:
            self.distance_to_obstacle_f = min(lidar_f[:, 1])-2.247148275375366

        if len(lidar_r) == 0:
            pass
        else:
            self.distance_to_obstacle_r = np.sqrt(
                min(lidar_r[:, 0]**2 + lidar_r[:, 1]**2))

        if len(lidar_l) == 0:
            pass
        else:
            self.distance_to_obstacle_l = np.sqrt(
                min(lidar_l[:, 0]**2 + lidar_l[:, 1]**2))

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if len(self.collision_hist) != 0:
            done = True
        else:
            done = False
        state_ = self.points
        return state_, reward, done, None
