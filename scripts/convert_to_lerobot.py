from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import os
import numpy as np
import argparse
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

def get_all_hdf5_files(directory):
    """
    Get all HDF5 files in a directory and its subdirectories.
    """
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

def create_lerobot_dataset_from_hdf5(args):
    dataset = LeRobotDataset.create(
        repo_id=args.dataset_name,
        robot_type="franka",
        fps=10,
        features={
            "observation.image_0":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.image_1":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.image_2":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.image_3":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.image_4":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.image_wrist":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.state":{
                "dtype": "float",
                "shape": (7,),
                "names": ["state"]
            },
            "action":{
                "dtype": "float",
                "shape": (7,),
                "names": ["actions"]
            },

        },
        image_writer_processes=5,
        image_writer_threads=10
    )
    
    if args.task_list is None:
        tasks = os.listdir(args.dataset_path)
    else:
        tasks = args.task_list
    print("Task to process:", tasks)
    h5py_files = list()
    for task in tasks:
        h5py_files.extend(get_all_hdf5_files(os.path.join(args.dataset_path, task))[:args.max_files])
    print("File numbers:", len(h5py_files))
    for file in h5py_files:
        with h5py.File(file, "r") as f:
            for timestamp in f["data"].keys():
                images = f["data"][timestamp]["observation"]["rgb"][()]
                ee_state = f["data"][timestamp]["observation"]["ee_state"][()]
                q_state = f["data"][timestamp]["observation"]["q_state"][()]
                actions = f["data"][timestamp]["trajectory"]
                ee_pos, ee_quat, gripper = ee_state[:, :3], ee_state[:, 3:7], ee_state[:, 7]
                ee_euler = np.array([quat2euler(q) for q in ee_quat])
                ee_pos -= np.array([0, -0.4, 0.78])
                ee_state = np.concatenate([ee_pos, ee_euler, gripper.reshape(-1, 1)], axis=1)
                assert images.shape[0] == ee_state.shape[0] == q_state.shape[0] == actions.shape[0]
                for i in range(images.shape[0]):
                    action = actions[i]
                    if actions[i][-1] > 0.03:
                        action = np.concatenate([action[:6], np.array([1])])
                    else:
                        action = np.concatenate([action[:6], np.array([0])])
                    dataset.add_frame(
                        {
                                "observation.image_0": images[i][2], #left low
                                "observation.image_1": images[i][3], #left high
                                "observation.image_2": images[i][0], #right low
                                "observation.image_3": images[i][1], #right high
                                "observation.image_4": images[i][4], #front
                                "observation.image_wrist": images[i][5], #wrist 
                                "observation.state": ee_state[i],
                                "action": action,
                        }
                    )
                dataset.save_episode(task=np.array(f["data"][timestamp]["instruction"])[0].decode("utf-8"))
    dataset.consolidate(run_compute_stats=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LeRobot dataset")
    parser.add_argument("--dataset-name", type=str, default="test", help="Name of the dataset")
    parser.add_argument("--dataset-path", type=str, default="/media/shiduo/LENOVO_USB_HDD/dataset/VLABench/select_billiards", help="Path to the dataset")
    parser.add_argument("--max-files", type=int, default=500, help="Maximum number of files to process")
    parser.add_argument("--task-list", type=str, nargs="+", default=None, help="List of tasks to process")
    args = parser.parse_args()

    create_lerobot_dataset_from_hdf5(args)