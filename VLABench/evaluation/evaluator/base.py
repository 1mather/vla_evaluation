import json
import os
import numpy as np
import random
import mediapy
from tqdm import tqdm
from VLABench.envs import load_env
from VLABench.utils.utils import euler_to_quaternion
from scipy.spatial.transform import Rotation as R
import pdb
import copy
from tutorials.test_client import send_test_request
from VLABench.evaluation.model.policy.client import RemoteAgentClient
OBSERVATION={
    "observation.image_0":2,
    "observation.image_1":3,
    "observation.image_2":0,
    "observation.image_3":1,
    "observation.image_4":4,
    "observation.image_wrist":5,
}
CAMERA_VIEW_INDEX={
    "select_painting": 1,
    "put_box_on_painting": 1,
    "select_chemistry_tube":2,
    "find_unseen_object":2,
    "texas_holdem": 2,
    "cluster_toy": 2,
    "select_fruit":2,
}

tasktoconfig={
    "select_fruit_table0":"configs/task_related/task_specific_config/select_apple/task_config_1_pos_200_table_0.json",
    "select_fruit_table1":"configs/task_related/task_specific_config/select_apple/task_config_1_pos_200_table_1.json",
    "select_fruit_table2":"configs/task_related/task_specific_config/select_apple/task_config_1_pos_200_table_2.json",
    "select_fruit_table3":"configs/task_related/task_specific_config/select_apple/task_config_1_pos_200_table_3.json",
    "select_fruit_table4":"configs/task_related/task_specific_config/select_apple/task_config_1_pos_200_table_4.json",


    "select_fruit_difficult_table0":"configs/task_related/task_specific_config/select_apple_difficult/task_config_1_pos_200_table_0.json",
    "select_fruit_difficult_table1":"configs/task_related/task_specific_config/select_apple_difficult/task_config_1_pos_200_table_1.json",
    "select_fruit_difficult_table2":"configs/task_related/task_specific_config/select_apple_difficult/task_config_1_pos_200_table_2.json",
    "select_fruit_difficult_table3":"configs/task_related/task_specific_config/select_apple_difficult/task_config_1_pos_200_table_3.json",
    "select_fruit_difficult_table4":"configs/task_related/task_specific_config/select_apple_difficult/task_config_1_pos_200_table_4.json",

    "select_fruit_random_position":"configs/task_related/task_specific_config/select_apple_ood/task_config_rand_pos.json",


    "add_condiment":"configs/task_related/task_specific_config/add_condiment/task_config_1_pos_200.json",
    "insert_flower":"configs/task_related/task_specific_config/insert_flower/task_config_1_pos_200.json",
    "select_chemistry_tube":"configs/task_related/task_specific_config/select_chemistry_tube/task_config_1_pos_200.json",

    "select_fruit_difficult":"configs/task_related/task_specific_config/select_apple_difficult/task_config_1_pos_200.json",
    "add_condiment_difficult":"configs/task_related/task_specific_config/add_condiment_difficult/task_config_1_pos_200.json",
    "insert_flower_difficult":"configs/task_related/task_specific_config/insert_flower_difficult/task_config_1_pos_200.json",
    "select_chemistry_tube_difficult":"configs/task_related/task_specific_config/select_chemistry_tube_difficult/task_config_1_pos_200.json",

    "select_fruit_ood_camera":"configs/task_related/task_specific_config/select_apple_ood_camera/task_config_1_pos_200.json",

    "get_coffee_simple":"",

    "texas_holdem":""
}
def quat2euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles
class Evaluator:
    def __init__(self, 
                 tasks,
                 n_episodes,
                 episode_config=None,
                 max_substeps=50,
                 tolerance=1e-2,
                 metrics=["success_rate"],
                 #metrics=['progress_score'],
                 save_dir=None,
                 visulization=False,
                 **kwargs
                 ):
        """
        Basic evaluator of policy
        params:
            tasks: list of task names to evaluate, e.g. ["task1", "task2"]
            n_episodes: number of episodes to evaluate in each task
            episode_config: dict or path of config file for episode generation
            max_substeps: maximum number of substeps for env.step
            metrics: list of metrics to evaluate
            save_dir: directory to save the evaluation results
            visulization: whether to visualize the evaluation progress as videos
        """
        if isinstance(episode_config, str):
            with open(episode_config, "r") as f:
                self.episode_config = json.load(f)
        else:self.episode_config = episode_config
        if self.episode_config is None:
            print("Load the task episodes by task, instead of episodes")
            #import pdb; pdb.set_trace()
            episode_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),tasktoconfig.get(tasks[0]))
            with open(episode_path, "r") as f:
                self.episode_config = json.load(f)
            if '_table' in tasks[0]:
                tasks[0]=tasks[0].split("_table")[0]
            if "difficult" in tasks[0]:
                tasks[0]=tasks[0].split("_difficult")[0]
            if "random_position" in tasks[0]:
                tasks[0]=tasks[0].split("_random_position")[0]
            if "ood_camera" in tasks[0]:
                tasks[0]=tasks[0].split("_ood_camera")[0]
        else:
            Warning(f"The number of episodes should be less than the number of configurations, {len(self.episode_config)} >= {n_episodes}")

        self.eval_tasks = tasks
        self.n_episodes = n_episodes 
        
        self.max_substeps = max_substeps
        self.tolerance = tolerance
        self.target_metrics = metrics
        
        # log, store and visualization
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.visulization = visulization

        if kwargs.get("observation_images",None) is not None:
            self.observation_images = kwargs["observation_images"]
        else:
            self.observation_images = None
        
    def evaluate(self, agent):
        """
        Evaluate the agent on all tasks defined in the evaluator.
        """   
        metrics = {}
        instruction={}
        # 创建 episode_config 的深拷贝
       
        
        for task in self.eval_tasks:
            task_infos = []
            instructions=[]
            for i in tqdm(range(self.n_episodes), desc=f"Evaluating {task} of {agent.name}"):
                kwargs = {
                    "unnorm_key": task
                }
                episode_config_copy = copy.deepcopy(self.episode_config) if self.episode_config is not None else None
                if episode_config_copy is None:
                    info,obs = self.evaluate_single_episode(agent, task, i, None, seed=42+i, **kwargs)
                else: 
                    # 为每个评估创建新的深拷贝
                    config_for_episode = copy.deepcopy(episode_config_copy)
                    info,obs= self.evaluate_single_episode(agent, task, i, config_for_episode, **kwargs)
                if obs["instruction"] is not None:
                    print(obs["instruction"])
                else:
                    print("there is no instruction")
                task_infos.append(info)
                instructions.append(obs["instruction"])
            metric_score = self.compute_metric(task_infos)       
            metrics[task] = metric_score
            instruction[task]=instructions
            
        if self.save_dir is not None:
            os.makedirs(os.path.join(self.save_dir, agent.name),exist_ok=True)
            with open(os.path.join(self.save_dir, agent.name, "metrics.json"), "w") as f:
                json.dump(metrics, f)
            with open(os.path.join(self.save_dir, agent.name, "instruction.json"), "w") as f:
                json.dump(instruction, f)
        return metrics
        
    def evaluate_single_episode(self, agent, task_name, episode_id, episode_config, seed=42, max_episode_length=300, **kwargs):
        """
        If episode_config is given, the task and scene will load deterministically.
        params:
            agent: policy to evaluate
            task_name: name of the task
            episode_id: id of the episode
            episode_config: configuration of the task
            seed: seed for the random number generator, if episode_config is None
            max_episode_length: maximum length of the episode
        """
        if episode_config is None: # use random seed to ditermine the task
            np.random.seed(seed)
            random.seed(seed)
        env = load_env(task_name, config=episode_config)
        env.reset()
        send_test_request(None,None,is_reset=True)
        #pdb.set_trace()
        success = False
        info = {}
        frames_to_save = []
        view_of_model=[]
        for i in range(max_episode_length):
        #for i in range(1):
            observation = env.get_observation()
            observation["instruction"] = env.task.get_instruction()
            if self.save_dir is not None and self.visulization:
                
                frame_width=observation["rgb"][0].shape[1]
                frame_height=observation["rgb"][0].shape[0]
                # 将所有帧堆叠到一个大图像中
                num_cols = 2
                num_rows = 3
                stacked_image = np.zeros((frame_height * num_rows, frame_width * num_cols, 3), dtype=np.uint8)
                for i, frame in enumerate(observation["rgb"]):
                    row = i // num_cols
                    col = i % num_cols
                    y1 = row * frame_height
                    y2 = y1 + frame_height
                    x1 = col * frame_width
                    x2 = x1 + frame_width
                    # 确保尺寸一致
                    # 复制到堆叠图像中
                    stacked_image[y1:y2, x1:x2] = frame               
                # 写入堆叠后的帧
                frames_to_save.append(stacked_image)
                cam_index = CAMERA_VIEW_INDEX.get(task_name)
                if self.observation_images is not None:
                    cam_index = [OBSERVATION.get(img) for img in self.observation_images]

                indices = cam_index if isinstance(cam_index, list) else [cam_index]
                view_of_model_frame=[observation["rgb"][idx] for idx in indices]

                view_of_model.append(np.vstack(view_of_model_frame))
            if agent.control_mode == "ee":
                #pdb.set_trace()
                #但是注意这里输出的应该是delta_action, 具体实现在VLABench/evaluation/model/policy/Openvla.py
            
                """
                这里可以添加多个接口用于测评不同的模型
                """
                if isinstance(agent, RemoteAgentClient):
                    ee_state = observation["ee_state"]
                    ee_pos, ee_quat, gripper = ee_state[:3], ee_state[3:7], np.array([ee_state[7]])
                    print(f"the current pose is {ee_pos}")
                    ee_euler = quat2euler(ee_quat)
                    ee_pos -= np.array([0, -0.4, 0.78])
                    print(ee_pos,ee_euler,gripper)
                    ee_state = np.concatenate([ee_pos, ee_euler, gripper], axis=0)
                    observation_images_tosend = {}
                    for img in self.observation_images:
                        observation_images_tosend[img] = observation["rgb"][OBSERVATION[img]]
                    #observation_images_tosend["observation.state"]=ee_state
                    
                    try:
                        pos, euler, gripper_state, view_index = send_test_request(observation_images_tosend,ee_state,is_reset=False)
                        distance_to_current=np.linalg.norm(pos-ee_pos)

                        print(f"the target pose is {pos}")
                        print("distance_to_current",distance_to_current)
                        # if distance_to_current>0.2:
                        #     continue
                    except Exception as e:
                        print(e)
                        continue
                else:
                    pos, euler, gripper_state, view_index = agent.predict(observation, **kwargs)
                quat = euler_to_quaternion(*euler)
                #quat=euler_to_quaternion(np.pi,0,-0.5*np.pi)
                action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)[:7]#delta关节角度
                action = np.concatenate([action[1], gripper_state])
            elif agent.control_mode == "joint":
                qpos, gripper_state = agent.predict(observation, **kwargs)
                action = np.concatenate([qpos, gripper_state])
            else:
                raise NotImplementedError(f"Control mode {agent.control_mode} is not implemented")    
            

            if i >=2:
                for _ in range(self.max_substeps):
                    timestep = env.step(action)
                    if timestep.last():
                        success=True
                        break
                    current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                    if np.max(current_qpos-np.array(action)[:7]) < self.tolerance \
                        and np.min(current_qpos - np.array(action)[:7]) > -self.tolerance:
                        break
            
            if success:
                break
            #这里进行的操作是：


        info["task"] = task_name
        info["success"] = success
        info["consumed_step"] = i
        #info["intention_score"] = env.get_intention_score()
        #info["progress_score"] = env.get_task_progress()
        
        env.close()
        if self.save_dir is not None and self.visulization:
            os.makedirs(os.path.join(self.save_dir, agent.name, task_name), exist_ok=True)
            episode_id=str(episode_id)+"view"+str(view_index)
            if success:
                self.save_video(frames_to_save, os.path.join(self.save_dir, agent.name, task_name, f"{episode_id}_success.mp4"))
            else:
                self.save_video(frames_to_save, os.path.join(self.save_dir, agent.name, task_name, f"{episode_id}_fail.mp4"))
            mediapy.write_video(os.path.join(self.save_dir, agent.name, task_name, f"{episode_id}_view_of_model.mp4"),view_of_model, fps=10) 
        return info,observation
        
    def compute_metric(self, infos):
        """
        Compute the metric scores for the evaluation
        param:
            infos: list of episode information
        """
        metric = {}
        for key in self.target_metrics:
            if key == "success_rate": # compute the success rate
                success = [info["success"] for info in infos]
                sucess_rate = np.mean(success)
                metric["success_rate"] = sucess_rate
            elif key == "intention_score":
                intention_score = [info["intention_score"] for info in infos]
                avg_intention_score = np.mean(intention_score)
                metric["intention_score"] = avg_intention_score
            elif key == "progress_score":
                progress_score = [info["progress_score"]for info in infos]
                #avg_progress_score = np.mean(progress_score)
                metric["progress_score"] = progress_score
            else:
                #raise NotImplementedError(f"Metric {key} is not implemented")
                continue
        return metric
    
    def save_video(self, frames, save_dir):
        frames_to_save = [] 
        for frame in frames:
            frames_to_save.append(np.vstack([np.hstack(frame[:2]), np.hstack(frame[2:4])]))
        mediapy.write_video(save_dir, 
                            frames_to_save, fps=10) 