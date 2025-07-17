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
import image_tools

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def preprocess_image(image):
    """
    将图像从[H, W, C]格式转换为[C, H, W]格式
    
    Args:
        image: 形状为[H, W, C]的numpy数组
        
    Returns:
        形状为[C, H, W]的numpy数组
    """
    # 确保图像是正确的类型
    image = np.array(image, dtype=np.float32)
    
    # 执行转置
    image_chw = image.transpose(2, 0, 1)
    
    # 可选：标准化图像
    # image_chw = image_chw / 255.0  # 归一化到[0,1]范围
    
    return image_chw
def send_test_request(client, images, ee_state, instruction):
    """同步版本的send_test_request函数"""
    # 定义内部异步函数
    async def _async_send_request(images, ee_state, instruction):
        # uri = "ws://127.0.0.1:8000"
        uri = "ws://0.0.0.0:8000"
        async with websockets.connect(uri) as websocket:
            observation = {}
            for key,value in images.items():
                observation[key] = image_tools.resize_with_pad(value, 224, 224)
            state = np.array(ee_state,dtype=np.float32)
            # 创建观察数据，使用正确的键名
            observation["observation/state"] = state
            observation["prompt"] = instruction
            

            packed_data = msgpack_numpy.packb(observation, use_bin_type=True)
            await websocket.send(packed_data)
            
            # 接收响应
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
                print(f"the current pose is {observation['observation.state']}")
                print(f"the target pose is {target_pos},{target_euler},{gripper_state}")
                # 解析响应...
                return target_pos, target_euler, gripper_state, view_index
            except Exception as e:
                logging.error(f"Failed to unpack response: {e}")
                logging.error(f"Raw response: {response}")
                return None
    
    # 使用事件循环执行异步函数并获取结果
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(_async_send_request(images, ee_state, instruction))
    return result  # 返回实际结果，而不是协程

def send_test_request(client, images, ee_state, instruction):
    observation = {}
    for key,value in images.items():
        observation[key] = image_tools.resize_with_pad(value, 224, 224)
    state = np.array(ee_state,dtype=np.float32)
    # 创建观察数据，使用正确的键名
    observation["observation/state"] = state
    observation["prompt"] = instruction
    # Query model to get action
    action_chunk = client.infer(observation)["actions"]
    return action_chunk

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