from __future__ import division
from mujoco_py import load_model_from_path, MjSim
import random
import math
import time
import os
import numpy as np
from scipy.misc import imsave
import matplotlib as mpl
from scipy.misc.pilutil import imshow
import logging
import copy
import gym
import cv2

initial_pos = [0.1, -50 / 180 * math.pi, 61.03 / 180 * math.pi,
               0, 0, 0, # robotic arm
               0, 0, 0, 0,
               0, 0, 0, 0, # two fingers
               -0.62, 0.32, 0, math.cos(math.pi * 0.16), 0, 0, math.sin(math.pi * 0.16),
               -0.72, 0.38, 0, math.cos(math.pi * 0.18), 0, 0, math.sin(math.pi * 0.18),
               -0.835, 0.425, 0, math.cos(math.pi * 0.195), 0, 0, math.sin(math.pi * 0.195),
               -0.935, 0.46, 0, math.cos(math.pi * 0.23), 0, 0, math.sin(math.pi * 0.23)] # 4 cubes
joint_pos = [[29.70 / 180 * math.pi, -85 / 180 * math.pi, 115 / 180 * math.pi], # yellow
             [31.00 / 180 * math.pi, -78 / 180 * math.pi, 105 / 180 * math.pi],
             [30.55 / 180 * math.pi, -70 / 180 * math.pi, 99 / 180 * math.pi],
             [20 / 180 * math.pi, -45 / 180 * math.pi, 60 / 180 * math.pi]]
closed_pos = [1.12810781, -0.59798289, -0.53003607]
fig_size_1 = (214, 214) # For the workbench camera
fig_size_2 = (214, 214) # For the upper camera
gaussian_noise_parameters = (20, 20)
gaussian_blur_prarmeters = ((5, 5), 1.5)
safety_threshold = 0.01 # Used to determine the stablity of current joint positions
grasp_steps = 120 # The minimum steps in simulator for the gripper to close is 120
drop_steps = 60 # It takes roughly 12 steps to reach ground from a 0.3m high position, extra iterations are for the convergance of final postion
sensor_threshold = 2.0 # Used to judge whether a cube is firmly grasped
action_scale = 30

class vector():
    # 3D vector class
    def __init__(self, x=0, y=0, z=1):
        self.x, self.y, self.z = x, y, z

    def add(self, v):
        return vector(self.x + v.x, self.y + v.y, self.z + v.z)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def mul_vec(self, v):
        return vector(self.y*v.z-self.z*v.y, self.z*v.x-self.x*v.z, self.x*v.y-self.y*v.x)

    def mul_num(self, s):
        return vector(self.x * s, self.y * s, self.z * s)

class quaternion():
    # Quaternion class used for 3D rotation
    def __init__(self, w=0, v=vector(0,0,1)):
        self.v = v
        self.w = w

    def rev(self):
        N = math.sqrt(self.w*self.w + self.v.x*self.v.x + self.v.y*self.v.y + self.v.z*self.v.z)
        return quaternion(self.w/N, vector(-self.v.x/N, -self.v.y/N, -self.v.z/N))

    def mul(self, q):
        res_v = self.v.mul_vec(q.v).add(q.v.mul_num(self.w)).add(self.v.mul_num(q.w))
        res_w = self.w*q.w - self.v.dot(q.v)
        return quaternion(res_w, res_v)

class lab_env():
    def __init__(self, env, args):
        #super(lab_env, self).__init__(env)
        # The real-world simulator
        self.model = load_model_from_path('xmls/lab_env.xml')
        self.sim = MjSim(self.model)
        # Used to locate the gripper
        self.model2 = load_model_from_path('xmls/light_env.xml')
        self.sim2 = MjSim(self.model2)

    def reset(self, task_id):
        self.task = task_id
        self.grasping = -1
        self.last_grasp = -1
        # Configure gravity
        for i in range(4):
            self.sim.data.ctrl[i] = -1
        # Configure joint positions
        for i in range(42):
            self.sim.data.qpos[i] = initial_pos[i]
        for i in range(3):
            self.sim.data.qpos[i] = joint_pos[task_id][i]
        self.pos_forward()
        self.sim.forward()

        remapped_pos = [remap(self.sim.data.qpos[0], -30 / 180 * math.pi, 45 / 180 * math.pi, -1, 1),
                        remap(self.sim.data.qpos[1], -105 / 180 * math.pi, -50 / 180 * math.pi, -1, 1),
                        remap(self.sim.data.qpos[2], 0 / 180 * math.pi, 180 / 180 * math.pi, -1, 1), 0]

        return (remapped_pos,) + self.get_state() 

    def step(self, action):
        self.sim.data.qpos[0] = remap(action[0], -1, 1, -30 / 180 * math.pi, 45 / 180 * math.pi)
        self.sim.data.qpos[1] = remap(action[1], -1, 1, -105 / 180 * math.pi, -50 / 180 * math.pi)
        self.sim.data.qpos[2] = remap(action[2], -1, 1, 0 / 180 * math.pi, 180 / 180 * math.pi)

        self.pos_forward()
        self.sim.forward()

        if action[3] < self.last_grasp or self.grasping == -1:
            t = int(remap(action[3], -1, 1, 0, grasp_steps))
            for i in range(6, 14):
                self.sim.data.qpos[i] = 0
            self.sim.forward()
            self.grasping = -1
            self.sim.data.ctrl[4] = 1
            self.sim.data.ctrl[5] = 1
            backup = [self.sim.data.qpos[i] for i in [15, 16, 22, 23, 29, 30, 36, 37]]

            for i in range(t):
                self.sim.step()
                stop = False
                for j in range(4):
                    if self.sim.data.sensordata[j] > sensor_threshold:
                        self.grasping = j
                        self.pickuppos = [self.sim.data.qpos[i] for i in (list(range(6)) + list(range(14 + 7 * self.grasping, 21 + 7 * self.grasping)))]
                        stop = True
                        break
                for i in range(4):
                    for j in range(2):
                        self.sim.data.qpos[15 + 7 * i + j] = backup[i * 2 + j]
                if stop:
                    break
            self.gripper_sync()
            self.sim.forward()

            self.sim.data.ctrl[4] = 0
            self.sim.data.ctrl[5] = 0

        self.last_grasp = action[3]

        return self.get_state()

    # Ensure that the gripper is always heading down and is parallar to the desk edge
    def pos_forward(self):
        self.sim.data.qpos[3] = math.pi * 1.5 - self.sim.data.qpos[1] - self.sim.data.qpos[2]
        self.sim.data.qpos[4] = math.pi * 1.5
        self.sim.data.qpos[5] = math.pi * 1.25 + self.sim.data.qpos[0]
        self.gripper_sync()

        if self.grasping != -1:
            current_xyz = pos_to_xyz(self.sim.data.qpos[0: 6])
            old_xyz = pos_to_xyz(self.pickuppos[0: 6])
            for i in range(3):
                self.sim.data.qpos[14 + 7 * self.grasping + i] = self.pickuppos[6 + i] + current_xyz[i] - old_xyz[i]
            '''
            old_quat = quaternion(self.pickuppos[9], vector(self.pickuppos[10], self.pickuppos[11], self.pickuppos[12]))
            rotate_quat = quaternion(math.cos(self.sim.data.qpos[0] - self.pickuppos[0]), vector(0, 0, math.sin(self.sim.data.qpos[0] - self.pickuppos[0])))
            new_quat = rotate_quat.mul(old_quat)
            self.sim.data.qpos[17 + 7 * self.grasping] = new_quat.w
            self.sim.data.qpos[18 + 7 * self.grasping] = new_quat.v.x
            self.sim.data.qpos[19 + 7 * self.grasping] = new_quat.v.y
            self.sim.data.qpos[20 + 7 * self.grasping] = new_quat.v.z
            '''

    def gripper_sync(self):
        self.sim.data.qpos[9] = gripper_consistent(self.sim.data.qpos[6: 9])
        self.sim.data.qpos[13] = gripper_consistent(self.sim.data.qpos[10: 13])

    def get_state(self):
        sync(self.sim, self.sim2, 6)
        # Locate the gripper, render twice to overcome bugs in mujoco
        image_1 = copy.deepcopy(self.sim.render(width = 960, height = 720, camera_name = 'workbench_camera'))
        image_1 = copy.deepcopy(self.sim.render(width = 960, height = 720, camera_name = 'workbench_camera'))
        image_2 = copy.deepcopy(self.sim.render(width = 960, height = 720, camera_name = 'upper_camera'))
        image_2 = copy.deepcopy(self.sim.render(width = 960, height = 720, camera_name = 'upper_camera'))
        image_3 = copy.deepcopy(self.sim2.render(width = 960, height = 720, camera_name = 'workbench_camera'))
        image_3 = copy.deepcopy(self.sim2.render(width = 960, height = 720, camera_name = 'workbench_camera'))
        x1, y1 = get_x_y(image_3)
        image_4 = copy.deepcopy(self.sim2.render(width = 960, height = 720, camera_name = 'upper_camera'))
        image_4 = copy.deepcopy(self.sim2.render(width = 960, height = 720, camera_name = 'upper_camera'))
        x2, y2 = get_x_y(image_4)
        # Crop gripper images and add noise
        image_1 = cv2.GaussianBlur(gaussian_noise(crop(image_1, x1 + fig_size_1[0] // 2, y1, *fig_size_1), *gaussian_noise_parameters), *gaussian_blur_prarmeters).transpose((2, 0, 1))
        image_2 = cv2.GaussianBlur(gaussian_noise(crop(image_2, x2 + fig_size_2[0] // 2, y2, *fig_size_2), *gaussian_noise_parameters), *gaussian_blur_prarmeters).transpose((2, 0, 1))

        danger = int(self.safety_check() or math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2))
        return [x1, y1, x2, y2, int(self.grasping == self.task), danger], (image_1, image_2)

    def safety_check(self):
        # return 0 if safe, otherwise 1
        backup = [self.sim.data.qpos[i] for i in range(14)]
        self.sim.step()
        s = 0
        for i in range(6):
            s += abs(backup[i] - self.sim.data.qpos[i])
            self.sim.data.qpos[i] = backup[i]
        return s > safety_threshold

    def force_close(self):
        for i in range(2):
            for j in range(3):
                self.sim.data.qpos[6 + i * 4 + j] = closed_pos[j]
        self.gripper_sync()
        self.sim.forward()

def get_x_y(fig):
    gray_fig = fig.sum(axis = 2)
    x, y = np.where(gray_fig > 0)
    x_mean = x.mean() if x.shape[0] > 0 else math.nan
    y_mean = y.mean() if y.shape[0] > 0 else math.nan
    return x_mean, y_mean

def pos_to_xyz(pos):
    x = 0.425 * math.cos(pos[1]) + 0.39225 * math.cos(pos[1] + pos[2]) - 0.09465 * math.sin(pos[1] + pos[2] + pos[3])\
    + 0.0823 * math.cos(pos[1] + pos[2] + pos[3]) * math.sin(pos[4])
    y = 0.10915 + 0.0823 * math.cos(pos[4])
    c, s = math.cos(pos[0] + 0.75 * math.pi), math.sin(pos[0] + 0.75 * math.pi)
    z = 0.089159 - 0.425 * math.sin(pos[1]) - 0.39225 * math.sin(pos[1] + pos[2]) - 0.09465 * math.cos(pos[1] + pos[2] + pos[3])\
    - 0.0823 * math.sin(pos[1] + pos[2] + pos[3]) * math.sin(pos[4])
    x, y = x * c - y * s, x * s + y * c
    return x, y, z

def crop(fig, x, y, height, width):
    Height, Width, _ = fig.shape
    x = constrain(x, height // 2, Height - height // 2)
    y = constrain(y, width // 2, Width - width // 2)
    return fig[x - height // 2: x + height // 2, y - width // 2: y + width // 2, :]

def constrain(x, lower_bound, upper_bound):
    x = (upper_bound + lower_bound) / 2 if math.isnan(x) else x
    x = upper_bound if x > upper_bound else x
    x = lower_bound if x < lower_bound else x
    return int(round(x))

def gaussian_noise(fig, mu, sigma):
    noise = mu + np.random.randn(*fig.shape) * sigma
    target = fig + noise
    target[target < 0] = 0
    target[target > 255] = 255
    return target

# Synchronize sim2 with sim1
def sync(sim1, sim2, length):
    for i in range(length):
        sim2.data.qpos[i] = sim1.data.qpos[i]
    sim2.forward()

def gripper_consistent(angle):
    x = -0.006496 + 0.0315 * math.sin(angle[0]) + 0.04787744772 * math.cos(angle[0] + angle[1] - 0.1256503306) - 0.02114828598 * math.sin(angle[0] + angle[1] + angle[2] - 0.1184899592)
    y = -0.0186011 - 0.0315 * math.cos(angle[0]) + 0.04787744772 * math.sin(angle[0] + angle[1] - 0.1256503306) + 0.02114828598 * math.cos(angle[0] + angle[1] + angle[2] - 0.1184899592)
    return math.atan2(y, x) + 0.6789024115

def create_env(env_id, args):
    env = env_id #gym.make(env_id)
    if env == 'Goal_LfD':
        env = lab_env(env, args)
    return env

def remap(x, lb, ub, LB, UB):
    return (x - lb) / (ub - lb) * (UB - LB) + LB

if __name__ == '__main__':
    env = create_env('Goal_LfD', None)
    angles, state, images = env.reset(0)
    print(state)
    state, images = env.step([0.2, -0.2, 0.4, -1])
    #imshow(images[0])
    #imshow(images[1])
    fig1 = env.sim.render(width = 960, height = 720, camera_name = 'workbench_camera')
    fig1 = env.sim.render(width = 960, height = 720, camera_name = 'workbench_camera')
    #imshow(fig1)
    fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
    fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
    imshow(fig2)
    state, images = env.step([0.2, -0.2, 0.4, 1])
    print(state)
    fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
    fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
    imshow(fig2)
    state, images = env.step([0.2, 0, -0.2, 1])
    print(state)
    fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
    fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
    imshow(fig2)
    state, images = env.step([0.2, 0, -0.2, 0.9])
    print(state)
    fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
    fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
    imshow(fig2)

    '''
    video = cv2.VideoWriter("sample.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (1920, 720), True)
    for i in range(20):
        env.step([0, 0, 0, 0])
        fig1 = env.sim.render(width = 960, height = 720, camera_name = 'workbench_camera')
        fig1 = env.sim.render(width = 960, height = 720, camera_name = 'workbench_camera')
        fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
        fig2 = env.sim.render(width = 960, height = 720, camera_name = 'upper_camera')
        video.write(cv2.cvtColor(np.concatenate((fig1, fig2), axis=1), cv2.COLOR_BGR2RGB))
    video.release()
    '''