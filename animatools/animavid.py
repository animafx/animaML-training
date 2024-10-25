import cv2
import numpy as np
import matplotlib.pyplot as plt
from animatools import animaimg, animahelpers
import os

# Displays 3D + 3Ch Numpy Array as Video
def display_video(video_np):
    for frame in video_np:
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
# Loads Video as Time x H x W x 3Ch Numpy Array
# If Crop=True crops largest centred square
# If scale=512, rescales image to have height=512 (maintains width ratio)
# If scale=(512,720), rescales video to indicated size
def load_video_to_np(video_path, crop=False, scale=(0)):
    print("\n" + "Loading Video: " + video_path, ", Crop: ", crop, ", Scale: ", scale)
    cap = cv2.VideoCapture(video_path)
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_height = output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_width  = output_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Input Shape:   (" + str(frame_count) + ", " + str(input_height) + ", " + str(input_width) + ", 3)")
    
    if isinstance(scale, tuple) == True:
        output_height = int(scale[0])
        output_width  = int(scale[1])
    elif scale != 0:
        output_height = int(scale)
        output_width  = int((output_height / input_height) * input_width)
    # crop_side = min(output_height, output_width)
    # print("IN_HEIGHT:  ", input_height,  "IN_WIDTH:  ", input_width)
    # print("OUT_HEIGHT: ", output_height, "OUT_WIDTH: ", output_width)
    
    if crop==True:
        video_data = np.zeros((frame_count, min(output_height,output_width), min(output_height,output_width), 3), dtype=np.uint8)
        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                
                cropped = animaimg.crop_to_center_square(frame)
                video_data[i] = cv2.resize(cropped, dsize=(min(output_height,output_width), min(output_height,output_width)))
            else:
                break
    else:
        video_data = np.zeros((frame_count, output_height, output_width, 3), dtype=np.uint8)
        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                video_data[i] = cv2.resize(frame, dsize=(output_width, output_height))
            else:
                break
    
    cap.release()
    print("Output Shape: ", video_data.shape)
    return video_data

# Loads PNG Sequence as Time x H x W x 3Ch Numpy Array
# If Crop=True crops largest centred square
# If scale=512, rescales image to have height=512 (maintains width ratio)
# If scale=(512,720), rescales video to indicated size
def load_seq_to_np(folder_path, crop=False, scale=(0)):
    print("\n" + "Loading Sequence: " + folder_path, ", Crop: ", crop, ", Scale: ", scale)
    files = sorted(os.listdir(folder_path))
    frame = cv2.imread(os.path.join(folder_path, files[1]), cv2.IMREAD_UNCHANGED)
    # print(os.path.join(folder_path, files[1]))
    frame_count = len(files) - 1
    input_height, input_width, _ = frame.shape
    output_height, output_width  = input_height, input_width
    print("Input Shape:   (" + str(frame_count) + ", " + str(input_height) + ", " + str(input_width) + ", " + str(_) + ")")

    if isinstance(scale, tuple) == True:
        output_height = int(scale[0])
        output_width  = int(scale[1])
    elif scale != 0:
        output_height = int(scale)
        output_width  = int((output_height / input_height) * input_width)
    
    if crop:
        video_data = np.zeros((frame_count, min(output_height, output_width), min(output_height, output_width), 4), dtype=np.uint8)
        for i, file in enumerate(files):
            frame = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_UNCHANGED)
            if frame is not None:
                cropped         = animaimg.crop_to_center_square(frame)
                video_data[i-1] = cv2.resize(cropped, dsize=(min(output_height,output_width), min(output_height,output_width)))
            else:
                print(f"Unable to read file: {file}")
                # break
    else:
        video_data = np.zeros((frame_count, output_height, output_width, 3), dtype=np.uint8)
        for i, file in enumerate(files):
            frame = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_UNCHANGED)
            if frame is not None:
                video_data[i-1] = cv2.resize(frame, dsize=(output_width, output_height))
            else:
                print(f"Unable to read file: {file}")
                # break
            
    print("Output Shape: ", video_data.shape)
    return video_data

# Takes   frames x height x width x 4channels input np array
# Returns frames x height x width x 3channels output mask (alpha -> red, green=0, blue=0)
def mask_alpha_red(video_np):
    print("\n" + "Flattening Alpha Mask to Red Mask")
    print("Input Shape:  ", video_np.shape)
    frame_count, height, width, _ = video_np.shape
    video_data = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    
    for i, frame in enumerate(video_np):
        b, g, r, a = cv2.split(frame)
        image = cv2.merge((np.zeros_like(a), np.zeros_like(a), a))
        video_data[i] = image
    
    print("Output Shape: ", video_data.shape)
    return video_data

# Converts frames x height x width x channels input np array to target frame rate
def convert_fps(video_np, fps, output_fps):
    print("\n" + "Converting FPS: ", fps, " to ", output_fps)
    print("Input Shape:  ", video_np.shape)
    frame_count, height, width, channels = video_np.shape    
       # Calculate frame ratio for fps conversion
    frame_ratio = output_fps / fps
    
    # Calculate new frame count based on the frame ratio
    new_frame_count = int(frame_count * (fps / output_fps))
    
    # Initialize output video data array
    video_data = np.zeros((new_frame_count, height, width, channels), dtype=np.uint8)
    
    # Copy frames with proper interval to achieve target fps
    for i in range(new_frame_count):
        # Calculate the source frame index
        source_frame_index = int(i * frame_ratio)
        # Copy the frame data
        video_data[i] = video_np[source_frame_index]
    
    print("Output Shape: ", video_data.shape)
    return video_data

# Saves 3D + 3Ch Numpy array as video in output_video_path
def save_video(output_video_path, video_np, fps=24):
    print("\n" + "Saving Video To: ", output_video_path)
    frame_height, frame_width, _ = video_np[0].shape
    codec = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, codec, fps, (frame_width, frame_height))

    for frame in video_np:
        out.write(frame)

    out.release()
    print("Done!")

# Takes 2 frames x height x width x channels Numpy arrays (Videos) and Outputs the fg overlayed on top of bg
def comp_vid(bg_np, fg_np, mask=None):
    
    if mask is None:
        mask = fg_np[..., -1]  # Assuming alpha channel is the last channel

    # Perform composition
    composite = bg_np.copy()  # Make a copy of background array
    for i in range(3):  # Loop through RGB channels
        composite[..., i] = (fg_np[..., i] * (mask / 255.0) +
                             bg_np[..., i] * (1.0 - mask / 255.0))
        
    return composite
 
# Takes 2 Video/PNG Seq Paths and Saves the fg overlayed on top of bg and a mask on the red channel in output_video_path
def from_seq_comp_mask(seq_path, bg_path, output_video_path, mask=None, fps=24, crop=False, scale=(0)):
    
    # Loading sequence and background from path
    if os.path.isdir(seq_path):
        fg_np = load_seq_to_np(seq_path, crop=crop, scale=scale)
    elif os.path.isfile(seq_path):
        fg_np = load_video_to_np(seq_path, crop=crop, scale=scale)
    else:
        raise ValueError("Invalid sequence path")
    
    if os.path.isdir(bg_path):
        bg_np = load_seq_to_np(bg_path, crop=crop, scale=scale)
    elif os.path.isfile(bg_path):
        bg_np = load_video_to_np(bg_path, crop=crop, scale=scale)
    else:
        raise ValueError("Invalid background plate path")
    
    # Trimming clips to same length
    min_frames = min(fg_np.shape[0], bg_np.shape[0])
    fg_np = fg_np[:min_frames]
    bg_np = bg_np[:min_frames]
    if mask is not None:
        mask = mask[:min_frames]
    print("Trimmed FG: ", fg_np.shape)
    print("Trimmed BG: ", bg_np.shape)
    
    # Comping and Masking
    red_mask_np = mask_alpha_red(fg_np)
    comp_np = comp_vid(bg_np, fg_np, mask=mask)
    comp_output_path = animahelpers.generate_next_filename(output_video_path + "comp", ".mp4")
    mask_output_path = animahelpers.generate_next_filename(output_video_path + "mask", ".mp4")
    print("Mask Shape: ", red_mask_np.shape)
    print("Comp Shape: ", comp_np.shape)

    # Saving Files
    save_video(comp_output_path, comp_np, fps=fps)
    save_video(mask_output_path, red_mask_np, fps=fps)
    return comp_np, red_mask_np