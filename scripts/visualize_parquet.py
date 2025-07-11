import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm
import sys
import traceback

def extract_images_to_video(parquet_path, output_path, fps=30):

    try:
        print(f"process file: {parquet_path}")
        

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        

        df = pd.read_parquet(parquet_path)
        print(f"data frame size: {df.shape}")
        

        image_columns = []
        for col in df.columns:
            if len(df) > 0:
                first_val = df[col].iloc[0]
                if isinstance(first_val, dict) and 'bytes' in first_val:
                    image_columns.append(col)
        
        if not image_columns:
            print("no image data column found!")
            return
        
        print(f"find image columns: {image_columns}")
        
        first_frames = []
        for col in image_columns:
            first_image_data = df[col].iloc[0]
            if isinstance(first_image_data, dict) and 'bytes' in first_image_data:
                first_image_bytes = first_image_data['bytes']
                frame = cv2.imdecode(np.frombuffer(first_image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    first_frames.append(frame)
        
        if not first_frames:
            print("can not decode any image, please check the data format")
            return
        
        frame_height, frame_width = first_frames[0].shape[:2]
        num_cols = 2  
        num_rows = 3  
        
        stacked_width = frame_width * num_cols
        stacked_height = frame_height * num_rows
        
        print(f"single frame size: {frame_width}x{frame_height}, stacked size: {stacked_width}x{stacked_height}")
        

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (stacked_width, stacked_height))
        #video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            print(f"can not create video writer, path: {output_path}")
            return
        
        frame_count = 0

        for idx in tqdm(range(len(df))):

            frames = []
            for col in image_columns:
                image_data = df[col].iloc[idx]
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image_bytes = image_data['bytes']
                    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        frames.append(frame)
            
            if frames:
                stacked_image = np.zeros((stacked_height, stacked_width, 3), dtype=np.uint8) 
                for i, frame in enumerate(frames):
                    row = i // num_cols
                    col = i % num_cols
                    y1 = row * frame_height
                    y2 = y1 + frame_height
                    x1 = col * frame_width
                    x2 = x1 + frame_width
                    
                    if frame.shape[:2] != (frame_height, frame_width):
                        frame = cv2.resize(frame, (frame_width, frame_height))
                    
                    stacked_image[y1:y2, x1:x2] = frame
                
                video_writer.write(stacked_image)
                frame_count += 1
        
        video_writer.release()
        
        if frame_count > 0:
            print(f"successfully write {frame_count} frames to video: {output_path}")
            if os.path.exists(output_path):
                print(f"confirm the file is created, size: {os.path.getsize(output_path)} bytes")
            else:
                print(f"warning: the file seems not be created: {output_path}")
        else:
            print("no frames are written to the video!")
            
    except Exception as e:
        print(f"error when process file {parquet_path}: {str(e)}")
        traceback.print_exc()

def process_all_parquet_files(data_dir, output_dir, fps=30):

    os.makedirs(output_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        print(f"no parquet files found in {data_dir}")
        return
    print(f"find {len(parquet_files)} parquet files")
    
    for parquet_file in parquet_files:
        parquet_path = os.path.join(data_dir, parquet_file)
        output_filename = f"{os.path.splitext(parquet_file)[0]}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        extract_images_to_video(parquet_path, output_path, fps)


if __name__ == "__main__":

    data_dir = os.path.expanduser("/home/tyj/.cache/huggingface/lerobot/set_study_table_simple_100/data/chunk-000")
    output_dir = os.path.expanduser("/home/tyj/.cache/huggingface/lerobot/set_study_table_simple_100_video")
    os.makedirs(output_dir, exist_ok=True)
    print(f"input directory: {data_dir}")
    print(f"output directory: {output_dir}")
    process_all_parquet_files(data_dir, output_dir)