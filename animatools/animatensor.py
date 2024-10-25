import torch
import sys
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output
from PIL import Image
import torchvision
import torch
# from torchvision.datasets import HMDB51
import diffusers
# from diffusers.models import AutoencoderKL
# from diffusers import StableDiffusionPipeline


from . import animavid, animaimg, animahelpers
import torch

# Takes NP array (frames, height, width, channels)
# Returns Tensor (frames, channels, height, width)
def np_2_tensor(clip, debug=False):
    if debug:
        print("NP Shape:       ", clip.shape, clip.min(), clip.max())
    clip = torch.tensor(clip, dtype=torch.float32)
    clip = clip.permute(0,3,1,2)
    clip = clip / 255.0
    if debug:
        print("Tensor Shape:   ", clip.shape, clip.min(), clip.max())
    return clip


# Takes (batch x channels x height x width) tensor and returns tile at pos x,y
def get_tensor_tile(clip_tensor, x, y, tile_size=64):
    # print(clip_tensor.shape)
    temp = torch.reshape(clip_tensor, (1, -1, clip_tensor.shape[2], clip_tensor.shape[3]))
    # print(temp.shape)
    return temp[:, :, y*tile_size:y*tile_size+tile_size, x*tile_size:x*tile_size+tile_size]

# takes square clip (frames, channels, height, width)
# reshapes batch into channels and puts tiles in batch
# returns (tiles, frames&channels, height, width)
def tensor_2_tiles(clip_tensor, tile_size=64, debug=False):
    count = 0
    axis_tiles = int((clip_tensor.shape[2] // tile_size))
    num_channels = int( clip_tensor.shape[0] * clip_tensor.shape[1] )
    tiles = torch.zeros(axis_tiles**2, num_channels, tile_size, tile_size)
    for i in range(0, axis_tiles):
        for j in range(0, axis_tiles):
            tiles[count] = get_tensor_tile(clip_tensor, j, i)
            count += 1
    if debug:
        print("Tiles Shape:    ", tiles.shape, tiles.min(), tiles.max())
    return tiles

# takes tiles and returns original tensor
def tiles_2_tensor(tiles, tensor_size=256, debug=False):
    axis_tiles = int(tensor_size // tiles.shape[2])
    tile_size  = tiles.shape[2]
    tiled       = torch.zeros(1, tiles.shape[1], tiles.shape[2]*axis_tiles, tiles.shape[2]*axis_tiles)
    frame_count = tiles.shape[1] // 3
    
    for i in range(0, axis_tiles):
        for j in range(0, axis_tiles):
            # print(i, j, i*axis_tiles+j, test_tiles[i*axis_tiles+j].shape)
            tiled[:, :, i*tile_size:i*tile_size+tile_size, j*tile_size:j*tile_size+tile_size] = tiles[i*axis_tiles+j]
    temp = torch.zeros(tiles.shape[1] // 3, 3, tensor_size, tensor_size)
    for i in range(0, frame_count):
        temp[i] = tiled[0, i*3:i*3+3, :, :].clone()
    if debug:
        print("Tiled Shape:    ", tiled.shape, tiled.min(), tiled.max())
        print("Tensor Shape:   ", temp.shape, temp.min(), temp.max())
    return temp


def add_normal_noise(tensor, noise_level=0.1, offset=0):
    noise = torch.randn(tensor.size()) * noise_level
    # print(noise.min(), noise.max())
    noised_tensor = torch.clamp(tensor + float(offset) + noise, tensor.min(), tensor.max())
    # print(noised_tensor.min(), noised_tensor.max())
    return noised_tensor



def load_videos_to_tensor(folder_path, num_frames_per_clip=24, stride=1, max_clips=50, scale=512, crop=True):
    
    if not os.path.isdir(folder_path):
        print("Folder path does not exist")
        return
    if max_clips < 1:
        print("max_clips must be greater than 0")
        return 
    

    clips_tensor = torch.empty(max_clips, num_frames_per_clip, 3, 512, 512)
    count = 0
    
    for filename in os.listdir(folder_path):
        
        if count >= max_clips:
            break
        if not filename.endswith(".mp4"):
            continue

        # specifies current file path
        file_path = os.path.join(folder_path, filename)
        print("\n Processing: ", file_path)
        # loads to np from file
        clip_np = animavid.load_video_to_np(file_path, crop=crop, scale=scale)
        # drops every other frame
        drop_frame_np = clip_np[::stride]
        # trims to specified number of frames
        trimmed_np = drop_frame_np[0:num_frames_per_clip]
        clip_tensor = np_2_tensor(trimmed_np, debug=True)
        # transform tensor to -1 to 1
        clips_tensor[count] = clip_tensor * 2 - 1
        count += 1
    # return tensor with only filled rows
    trimmed_clips_tensor = clips_tensor.narrow(0, 0, count)
    return trimmed_clips_tensor

def encode_tensor_to_latent_tensor(tensor, pipe):
    print("Encoding tensors to latent tensors")
    print("Tensor shape: ", tensor.shape)
    latents_tensor = torch.empty(tensor.shape[0], tensor.shape[1], 4, 64, 64)
    print("Latent shape: ", latents_tensor.shape)
    
    for clip, i in zip(tensor, range(tensor.shape[0])):
        print("\n")
        print("Encoding Clip: ", i)
        print("Clip shape: ", clip.shape)
        for frame, j in zip(clip, range(clip.shape[0])):
            print("    Encoding Frame: ", j, ", ", frame.shape)
            latents_tensor[i][j] = 0.18215 * pipe.vae.encode(frame.unsqueeze(0)).latent_dist.mean
            torch.mps.empty_cache()
    return latents_tensor

def decode_latent_tensor_to_tensor(tensor, pipe):
    print("Decoding tensors to latent tensors")
    print("Tensor shape: ", tensor.shape)
    clips_tensor = torch.empty(tensor.shape[0], tensor.shape[1], 3, 512, 512)
    print("Clips shape: ", clips_tensor.shape)
    
    for clip, i in zip(tensor, range(tensor.shape[0])):
        print("\n")
        print("Decoding Clip: ", i)
        print("Clip shape: ", clip.shape)
        for frame, j in zip(clip, range(clip.shape[0])):
            print("    Decoding Frame: ", j, ",", frame.shape)
            clips_tensor[i][j] = 0.18215 * pipe.vae.encode(frame.unsqueeze(0)).latent_dist.mean
            torch.mps.empty_cache()

            
    return clips_tensor
    
    
    

        

def display_video_tensor(video_tensor):
    for frame in video_tensor:
        frame = frame.permute(1, 2, 0).numpy()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
def display_2img_tensor(before_tensor, after_tensor):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(before_tensor.permute(1, 2, 0).numpy())
        axes[0].set_title('Before')
        axes[0].axis('off')
        axes[1].imshow(after_tensor.permute(1, 2, 0).numpy())
        axes[1].set_title('After')
        axes[1].axis('off')
        plt.show()
        
def save_video_tensor(tensor, file_path, fps=24):
    # Convert tensor to numpy array
    video_np = tensor.permute(0, 2, 3, 1).numpy()
    # Scale the values back to 0-255 range
    video_np = np.clip(video_np, 0.0, 1.0) * 255
    
    # Convert the numpy array to OpenCV video format
    video_cv = video_np.astype(np.uint8)
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (video_cv.shape[2], video_cv.shape[1]))

    # Write each frame to the video file
    for frame in video_cv:
        writer.write(frame)

    # Release the VideoWriter object
    writer.release()
    
    