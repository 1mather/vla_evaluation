import json
import os
import numpy as np
import random
import mediapy
from tqdm import tqdm
from VLABench.envs import load_env
from VLABench.utils.utils import euler_to_quaternion
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.envs import load_env
from VLABench.robots import *
from VLABench.tasks import *
from VLABench.configs import name2config
from VLABench.utils.utils import find_key_by_value

from datetime import datetime
import collections
import dataclasses
import logging
import math
import websocket_client_policy as _websocket_client_policy
import image_tools
from scipy.spatial.transform import Rotation as R

def process_ee_state(ee_state):
    ee_pos, ee_quat, gripper = ee_state[:3], ee_state[3:7], ee_state[-1]
    ee_euler = quat2euler(ee_quat)
    ee_pos -= np.array([0, -0.4, 0.78])
    return np.concatenate([ee_pos, ee_euler, np.array([gripper]).reshape(-1)])

def quat2euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles

def save_video(frames, save_dir):
    frames_to_save = [] 
    for frame in frames:
        frames_to_save.append(np.vstack([np.hstack(frame[:2]), np.hstack(frame[2:4]), np.hstack(frame[4:6])]))
    mediapy.write_video(save_dir, 
                        frames_to_save, fps=10) 

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

args = Args
client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

max_episode_length=200
control_mode = 'ee'
task_name="select_fruit"
eval_unseen = False
max_substeps = 10
tolerance = 1e-2

# env = load_env(task_name, robot='franka', random_init=True, eval=eval_unseen)
success_index = []
save_dir = "/root/workspace/shiyan/Jiarui_VLABench/vlabench/test_yjj/saved_mp4/pi0_apple_horizon8_model29999_100cameraBCDE_cameraindex_refix_randompos_refix"

def eval_one_episode(episode_id, seed=42, save_dir=save_dir):
    np.random.seed(seed)
    random.seed(seed)
    env = load_env(task_name, config=None)
    success = False
    frames_to_save = []
    # save_dir = "/root/workspace/shiyan/Jiarui_VLABench/vlabench/test_yjj/saved_mp4/pi0_apple_horizon10_model29999_100onlycameraBCDE"
    env.reset()
    action_plan = collections.deque()
    for i in range(max_episode_length):
        observation = env.get_observation()
        observation["instruction"] = env.task.get_instruction()
        img = image_tools.resize_with_pad(observation['rgb'][4], 224, 224)
        image_0 = image_tools.resize_with_pad(observation['rgb'][2], 224, 224)
        image_1 = image_tools.resize_with_pad(observation['rgb'][3], 224, 224)
        image_2 = image_tools.resize_with_pad(observation['rgb'][0], 224, 224)
        image_3 = image_tools.resize_with_pad(observation['rgb'][1], 224, 224)
        wrist_img = image_tools.resize_with_pad(observation['rgb'][5], 224, 224)
        state = process_ee_state(observation['ee_state'])
        instruction = observation["instruction"]
        # instruction = "put the apple into the plate_unseen"
        print(instruction)
        frames_to_save.append(observation["rgb"])

        if not action_plan:
            element = {
                        "observation/image": img,
                        # "observation/wrist_image": wrist_img,
                        "observation/image_1": image_1,
                        "observation/image_2": image_2,
                        "observation/image_3": image_3,
                        "observation/state": state,
                        "prompt": instruction,
                    }
            # Query model to get action
            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[:8])
        cur_action = action_plan.popleft()
        pos, euler, gripper_open = cur_action[:3], cur_action[3:6], cur_action[-1]

        #动作处理
        quat = euler_to_quaternion(*euler)
        pos = np.array(pos, copy=True)
        pos += np.array([0, -0.4, 0.78])

        action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)[:7]
        # gripper_state = np.ones(2)*0.1 if gripper_open >= 0.03 else np.zeros(2)  # FIXME 闭合阈值由0.1改为0.03
        gripper_state = np.ones(2)*0.04 if gripper_open >= 0.1 else np.zeros(2)
        action = np.concatenate([action, gripper_state])
    
        for _ in range(max_substeps):
            timestep = env.step(action)
            if timestep.last():
                success=True
                break
            current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
            if np.max(current_qpos-np.array(action)[:7]) < tolerance \
                and np.min(current_qpos - np.array(action)[:7]) > -tolerance:
                break
        if success:
            break
    env.reset()
    print("success:",success)
    if success:
        success_index.append(episode_id)
    os.makedirs(os.path.join(save_dir, task_name), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_video(frames_to_save, os.path.join(save_dir, task_name, f"{episode_id}_{timestamp}.mp4"))


    if success:
        return 1
    return 0    

score = 0
total_episode = 25
for i in range(total_episode):
    score += eval_one_episode(i)
print("eval score:", score)
with open(save_dir+'/results.txt', 'w') as file:
    file.write(f'score={score}\n, total_episode={total_episode}\n')
    file.write('success_index=' + ', '.join(map(str, success_index)) + '\n')

# env.close()





