

from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.evaluation.model.policy.client import RemoteAgentClient
from VLABench.tasks import *
from VLABench.robots import *
import transformers

import argparse
tasks=[
    "select_fruit_table0",#the pretrain task
    "select_fruit_table1",
    "select_fruit_table2",
    "select_fruit_table3",
    "select_fruit_table4",


    "select_fruit_difficult_table0",#the pretrain task
    "select_fruit_difficult_table1",
    "select_fruit_difficult_table2",
    "select_fruit_difficult_table3",
    "select_fruit_difficult_table4",

    "select_fruit_random_position",#randomness in [-10,-5] [5,10] cm


    "add_condiment",
    "insert_flower",
    "select_chemistry_tube",

    "select_fruit_difficult",
    "add_condiment_difficult",
    "insert_flower_difficult",
    "select_chemistry_tube_difficult",

    #position ood for camera
    "select_fruit_ood_camera",

    "set_study_table",
]


def main(args):
        
    demo_tasks = [args.task]

    save_dir = args.save_dir

    from huggingface_hub import login
    from pathlib import Path
    import os
    os.environ["MUJOCO_GL"] = "egl"
    evaluator = Evaluator(
        tasks=demo_tasks,
        n_episodes=args.n_episodes,    
        max_substeps=args.max_substeps,   
        save_dir=save_dir,
        visulization=True,
        observation_images=["observation.image_1","observation.image_2","observation.image_3","observation.image_4"]  
        #observation_images= ["observation.image_0"]# the observation image input to the policy
    )
    policy = RemoteAgentClient(model="VQ_BET")

    result = evaluator.evaluate (policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a policy")
    parser.add_argument("--task", type=str, default="select_fruit_difficult", help="Task name")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--max_substeps", type=int, default=10, help="Maximum number of substeps")
    parser.add_argument("--save_dir", type=str, default="/mnt/data/310_jiarui/VLABench/logs/select_fruit_difficult", help="Save directory")
    args = parser.parse_args()
    main(args)

# python evaluation.py --task set_study_table --n_episodes 50 --max_substeps 10 --save_dir /mnt/data/310_jiarui/VLABench/logs/set_table