

from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.evaluation.model.policy.client import RemoteAgentClient
from VLABench.tasks import *
from VLABench.robots import *
import transformers
demo_tasks = ["add_condiment"]
unseen = False
save_dir = "/mnt/data/310_jiarui/VLABench_YJJ/vlabench/test_yjj/saved_eval_results/"
#/home/tyj/Documents/310_jiarui/openvla/log/train_log/openvla-7b+h5py_dataset+b1+lr-0.0005+lora-r8+dropout-0.0
# model_ckpt = "/mnt/data/310_jiarui/VLABench/model_parameter/base/openvla-7b+vlabench_dataset+b80+lr-0.0005+lora-r16+dropout-0.0--time-20250408-13"
# lora_ckpt ="/mnt/data/310_jiarui/VLABench/model_parameter/adapter/openvla-7b+vlabench_dataset+b80+lr-0.0005+lora-r16+dropout-0.0--time-20250408-13"


from huggingface_hub import login
from pathlib import Path
import os
import websocket_client_policy as _websocket_client_policy

os.environ["MUJOCO_GL"] = "egl"
evaluator = Evaluator(
    tasks=demo_tasks,
    n_episodes=6,     #配置评测次数
    episode_config="/mnt/data/310_jiarui/VLABench_YJJ/vlabench/VLABench/configs/task_related/task_specific_config/add_condiment/task_config_1_pos_100.json",
    max_substeps=10,   
    save_dir=save_dir,
    visulization=True,
    observation_images=["observation/image_1"]#, "observation/image_2", "observation/image", "observation/image_3"]  # FIXME 可以传入任意的image，但请与训练的时候保持一致。
)

client = _websocket_client_policy.WebsocketClientPolicy("0.0.0.0", 8001)

policy = RemoteAgentClient(model="Pi0_cameraB_add_condiment_1_200_model10000_test_speed")
result = evaluator.evaluate(policy, client)

