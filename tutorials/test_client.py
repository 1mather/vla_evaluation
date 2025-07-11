import asyncio
import numpy as np
import websockets
import cv2
import base64
import logging
import traceback
import sys
import msgpack_numpy
import msgpack
import torch
import os
import numpy as np
from PIL import Image
import cv2
import time

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DEBUG = True
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def save_image(key,value):
    save_dir = "images_sent_to_model"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{key}_.png"
    filepath = os.path.join(save_dir, filename)
    if isinstance(value, np.ndarray):

        if value.ndim == 3 and value.shape[2] == 3:  # RGB image
            cv2.imwrite(filepath, cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
        elif value.ndim == 2 or (value.ndim == 3 and value.shape[2] == 1):  # gray image
            cv2.imwrite(filepath, value)
        else:
            cv2.imwrite(filepath, value)
    elif hasattr(value, 'save'):  # PIL Image
        value.save(filepath)
    else:
        try:
            img_array = np.array(value)
            cv2.imwrite(filepath, img_array)
        except Exception as e:
            print(f"can not save image {key}: {e}")
    print(f"saved image: {filepath}")

def preprocess_image(image):
    """
    convert image to [C, H, W] format from [H, W, C] 
    """
    image = np.array(image, dtype=np.float32)
    image_chw = image.transpose(2, 0, 1)
    image_chw = image_chw / 255.0  # normalize to [0,1]
    return image_chw
def send_test_request(images, ee_state,is_reset=False):
    async def _async_send_request(images, ee_state):
        uri = "ws://127.0.0.1:8000"
        async with websockets.connect(uri) as websocket:
            observation = {}
            if is_reset:
                await websocket.send(b"reset")
                return None
            else:
                for key,value in images.items():
                    if DEBUG:
                        save_image(key,value)
                    observation[key] = np.array(preprocess_image(value),dtype=np.float32)
                state = np.array(ee_state,dtype=np.float32)
            
            observation["observation.state"] = state
            packed_data = msgpack_numpy.packb(observation, use_bin_type=True)
            await websocket.send(packed_data)
            
            response = await websocket.recv()
            try:
                unpacked_response = msgpack_numpy.unpackb(response, raw=False)
                logging.info(f"Received msgpack response: {unpacked_response}")
                if len(unpacked_response) == 1:
                    action = unpacked_response[0]
                else:
                    action = unpacked_response
                target_pos = action[:3]+np.array([0, -0.4, 0.78]) 
                target_euler = action[3:6]
                gripper_open = action[-1]
                gripper_state = np.ones(2)*0.04 if gripper_open >= 0.2 else np.zeros(2)
                view_index = 0

                return target_pos, target_euler, gripper_state, view_index
            except Exception as e:
                logging.error(f"Failed to unpack response: {e}")
                logging.error(f"Raw response: {response}")
                return None
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(_async_send_request(images, ee_state))
    return result  #retrun the result of the coroutine,not the coroutine itself
if __name__ == "__main__":
    try:
        result = asyncio.run(send_test_request())
        if result is not None:
            logging.info(f"Test completed successfully with result: {result}")
        else:
            logging.error("Test failed")
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        logging.error(traceback.format_exc())