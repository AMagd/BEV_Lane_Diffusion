import os
import imageio
import re
import torch
import numpy as np
import os
import re
import imageio
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from PIL import ImageOps
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import time
from PIL import Image, ImageDraw, ImageFont

import sys

COLOR_MAP =  {0: [255,255,255], 1: [107,131,167], 2: [167,101,110], 3: [103,142,117]}
# COLOR_MAP =  {0: [255,255,255], 1: [167,131,107], 2: [110,101,167], 3: [117,142,103]}
STROKE_WIDTH = 5

def rgb_to_label(img, color_map):
    img = img.to(dtype=torch.int32).numpy().transpose(1, 2, 0)
    label_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for label, rgb in color_map.items():
        matching_pixels = (img == np.array(rgb)).all(axis=-1)
        label_map[matching_pixels] = label
    return torch.from_numpy(label_map)

def apply_thicker_stroke(segmentation_tensor, color_map, stroke_width=3):
    result_tensor = segmentation_tensor.clone()

    # Create a binary mask for each label and apply dilation
    for label, color in color_map.items():
        if label in [1, 2, 3]:
            mask = (torch.stack([(segmentation_tensor[i] == color[i]) for i in range(segmentation_tensor.shape[0])]).sum(0) > 2)
            dilated_mask = torch.nn.functional.max_pool2d(mask.float().unsqueeze(0).unsqueeze(0), kernel_size=stroke_width, stride=1, padding=stroke_width // 2)
            
            # Resizing the dilated_mask back to the original mask size
            dilated_mask = torch.nn.functional.interpolate(dilated_mask, size=mask.shape, mode='nearest') > 0

            # Assign the color to the regions in the result tensor where the dilated mask is True
            for i in range(len(color)):
                result_tensor[i, dilated_mask[0, 0]] = color[i]

    return result_tensor



def get_seg(dd):
    _, max_class_index = torch.max(dd, dim=0)

    # Create an empty tensor for the RGB image
    output = torch.zeros(3, dd.shape[1], dd.shape[2])

    # Assign each pixel the color of its class with the highest probability
    for i in range(dd.shape[0]):
        class_color = torch.tensor(COLOR_MAP[i]).reshape(3, 1, 1)
        output += (max_class_index == i).float() * class_color

    # Rearrange the dimensions to get a tensor of shape [h, w, 3]
    return output

def calculate_mIoU(target, pred, mIoU_list, lanes_list, crosswalks_list, road_boundary_list):
    n_classes = 4
    pred = rgb_to_label(pred, COLOR_MAP)
    target = rgb_to_label(target, COLOR_MAP)
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes): 
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item() 
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            iou = float('nan')
            ious.append(iou)  # If there is no ground truth, do not include in evaluation
        else:
            iou = float(intersection) / float(union)
            ious.append(iou)

        if cls == 1:
            lanes_list.append(iou)
        elif cls == 2:
            crosswalks_list.append(iou)
        elif cls == 3:
            road_boundary_list.append(iou)
    
    miou = np.nanmean(ious)
    mIoU_list.append(miou)

    return mIoU_list, lanes_list, crosswalks_list, road_boundary_list

# The main directory
main_dir = './Test_Results_ddpm_40_used_in_thesis'
transform = T.ToPILImage()
# Get all subdirectories
subdirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

# Loop over all subdirectories
for subdir in subdirs:
    # Filter images
    if int(subdir.split('/')[-1]) != 103:
        continue
    sample_images = [f for f in os.listdir(subdir) if re.match(r'Sample_\d{4}\.pt', f)]
    
    # Sort images
    sample_images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]), reverse=True)
    
    # List of images for the gif
    images = []
    
    # Read images into memory
    for kk, image_file in enumerate(sample_images):

        text = f"t = {image_file.split('_')[-1].split('.')[0]}"
        blank_space_height = 20
        # images.append(imageio.imread(os.path.join(subdir, image_file)))
    
    # Add the Sample_filtered.pt file to the end
    # if 'Sample_filtered.pt' in os.listdir(subdir):
        img_label = torch.load(os.path.join(subdir, image_file))
        img_label = get_seg(img_label)
        img_label = img_label.clamp(0, 255)
        if kk > 55:
            img_label = apply_thicker_stroke(img_label, color_map = COLOR_MAP, stroke_width=STROKE_WIDTH)
        # img_label_tensor = img_label
        img_label = transform(img_label.byte()).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        width, height = img_label.size
        new_image = Image.new('RGB', (width, height + blank_space_height), 'white')
        new_image = Image.new('RGB', (width, height + blank_space_height), 'white')
        new_image.paste(img_label, (0, blank_space_height))
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
        draw = ImageDraw.Draw(new_image)
        # Calculate width of the text and find the center position
        text_width, _ = draw.textsize(text, font=font)
        text_x = (width - text_width) // 2
        draw.text((text_x, 5), text, font=font, fill='black')

        images.append(new_image)
    
    # # Add last image multiple times to create a pause
    # for _ in range(int(5 / 0.5)):  # 5 seconds / frame duration
    #     if 'Sample_filtered.pt' in os.listdir(subdir):
    #         images.append(imageio.imread(os.path.join(subdir, 'Sample_filtered.pt')))

    # Write gif file
    imageio.mimsave(os.path.join("", f"Diffusion_{os.path.basename(subdir)}.gif"), images, duration=0.5, subrectangles=True)