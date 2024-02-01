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
import sys

import wandb

# run = wandb.init(project="Visualization")

# def filter_image(image, tolerance=50):
#     # RGB color codes
#     red = torch.tensor([255, 0, 0])
#     green = torch.tensor([0, 255, 0])
#     blue = torch.tensor([0, 0, 255])

#     # Convert tensor to Numpy array and filter colors
#     np_img = image.numpy()
#     red_pixels = np.all(np.abs(np_img - red.numpy()) < tolerance, axis=-1)
#     green_pixels = np.all(np.abs(np_img - green.numpy()) < tolerance, axis=-1)
#     blue_pixels = np.all(np.abs(np_img - blue.numpy()) < tolerance, axis=-1)
#     np_img[~(red_pixels | green_pixels | blue_pixels)] = 0
#     filtered_image = Image.fromarray(np_img.astype(np.uint8))

#     return filtered_image

# Get the class index with the maximum probability for each pixel
# COLOR_MAP =  {0: background, 1: lanes, 2: crosswalks, 3: road boundary}
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

# def calculate_mIoU(img1, img2, color_map, mIoU_list, lanes_list, crosswalks_list, road_boundary_list):
#     num_classes = len(color_map)
    
#     # Convert images to class labels
#     labels1 = torch.zeros(img1.shape[1:]).long()
#     labels2 = torch.zeros(img2.shape[1:]).long()
#     for i, color in color_map.items():
#         labels1[(img1 == torch.tensor(color, dtype=torch.uint8).view(3, 1, 1)).all(dim=0)] = i
#         labels2[(img2 == torch.tensor(color, dtype=torch.uint8).view(3, 1, 1)).all(dim=0)] = i
    
#     # Calculate IoU for each class
#     ious = []
#     for i in range(1, num_classes):  # exclude the background class
#         intersection = ((labels1 == i) & (labels2 == i)).float().sum()
#         union = ((labels1 == i) | (labels2 == i)).float().sum()

#         if union == 0:  # correct absence of the class in both prediction and ground truth
#             iou = 1.0
#         elif intersection == 0:  # no overlap
#             iou = 0.0
#         else:  # partial or perfect overlap
#             iou = (intersection / union).item()

#         ious.append(iou)
#         if i == 1:
#             lanes_list.append(iou)
#         elif i == 2:
#             crosswalks_list.append(iou)
#         elif i == 3:
#             road_boundary_list.append(iou)
    
#     # Calculate mean IoU
#     if not ious:
#         print("########################################### IoUs IS EMPTY ###########################################")
#     mIoU = sum(ious) / len(ious) if ious else 0
#     mIoU_list.append(mIoU)
    
#     return mIoU_list, lanes_list, crosswalks_list, road_boundary_list


with torch.no_grad():
    # The main directory
    main_dir = './Test_Results_pretrained_bev'

    # Get all subdirectories
    subdirs = [os.path.join(main_dir, d) for d in sorted(os.listdir(main_dir)) if os.path.isdir(os.path.join(main_dir, d))]

    images_combined = []
    transform = T.ToPILImage()
    # Loop over all subdirectories

    # Open and resize the car image
    car_image_path = "./figs/lidar_car.png"  # replace with the path to your car image
    car_image = Image.open(car_image_path)
    car_image = car_image.resize((16, 16), Image.ANTIALIAS)
    car_image2 = car_image

    # Convert the PIL Image to a NumPy array
    car_image_np = np.array(car_image)

    # Separate the alpha channel
    alpha_channel = car_image_np[:, :, 3]

    # Convert the image from RGB to HSL
    car_image_np = cv2.cvtColor(car_image_np, cv2.COLOR_RGBA2BGR)
    car_image_hsl = cv2.cvtColor(car_image_np, cv2.COLOR_BGR2HLS).astype(float)

    # Shift the hue to blue (approximately 120-240 in OpenCV's 0-180 scale)
    blue_hue = 120
    car_image_hsl[:, :, 0] = blue_hue

    # Convert the image back to RGB
    car_image_blue = cv2.cvtColor(car_image_hsl.astype('uint8'), cv2.COLOR_HLS2BGR)

    # Add the alpha channel back
    car_image_blue = cv2.cvtColor(car_image_blue, cv2.COLOR_BGR2BGRA)
    car_image_blue[:, :, 3] = alpha_channel

    # Convert the NumPy array back to a PIL Image
    car_image = Image.fromarray(car_image_blue)

    border_width = 1
    border_color = "black"

    mIoU_list, lanes_list, crosswalks_list, road_boundary_list = [], [], [], []
    mIoU_list_cal, lanes_list_cal, crosswalks_list_cal, road_boundary_list_cal = [], [], [], []

    pbar = tqdm(total=len(subdirs), bar_format='{l_bar}{bar}| {percentage:3.0f}% ')
    print(f"num_labels = {len(subdirs)}")

    for kk, subdir in enumerate(subdirs):
        # if kk == 0:
        #     continue
        # Read the label tensor
        label_path = os.path.join(subdir, 'seg_label.pt')
        # Read the prediction tensor
        pred_path = os.path.join(subdir, 'Sample_0000.pt')

        # Read the prediction tensor
        pred_cal_path = os.path.join(subdir, 'cal.pt')

        cams_img_path = os.path.join(subdir[:-4]+f"{int(subdir[-4:])+1:04}"+"_cams_img.pt")

        if os.path.isfile(label_path) and os.path.isfile(pred_path):
            img_label = torch.load(label_path)
            img_label = get_seg(img_label)
            img_label = img_label.clamp(0, 255)
            img_label = apply_thicker_stroke(img_label, color_map = COLOR_MAP, stroke_width=STROKE_WIDTH)
            img_label_tensor = img_label
            img_label = transform(img_label.byte())
    
            img_pred = torch.load(pred_path)
            img_pred = get_seg(img_pred)
            img_pred = img_pred.clamp(0, 255)
            img_pred = apply_thicker_stroke(img_pred, color_map = COLOR_MAP, stroke_width=STROKE_WIDTH)
            img_pred_tensor = img_pred
            img_pred = transform(img_pred.byte())
    
            img_pred_cal = torch.load(pred_cal_path)
            img_pred_cal = get_seg(img_pred_cal)
            img_pred_cal = img_pred_cal.clamp(0, 255)
            img_pred_cal = apply_thicker_stroke(img_pred_cal, color_map = COLOR_MAP, stroke_width=STROKE_WIDTH)
            img_pred_cal_tensor = img_pred_cal
            img_pred_cal = transform(img_pred_cal.byte())

            mIoU_list, lanes_list, crosswalks_list, road_boundary_list = calculate_mIoU(img_label_tensor, img_pred_tensor, mIoU_list, lanes_list, crosswalks_list, road_boundary_list)
            mIoU_list_cal, lanes_list_cal, crosswalks_list_cal, road_boundary_list_cal = calculate_mIoU(img_label_tensor, img_pred_cal_tensor, mIoU_list_cal, lanes_list_cal, crosswalks_list_cal, road_boundary_list_cal)

            cams_img = torch.load(cams_img_path)
            cams_img = transform(cams_img.permute(2,0,1))
            width, height = cams_img.size
            new_height = 256
            new_width = int(new_height * width / height)
            cams_img = cams_img.resize((new_width, new_height))

            draw = ImageDraw.Draw(img_label)
            # draw.point((0, 0), fill=tuple(car_image.getpixel((5,5))[:3]))  # use the color of the top-left pixel of car_image
            img_label.paste(car_image, (128//2-16//2, 256//2-16//2), car_image)

            draw = ImageDraw.Draw(img_pred)
            # draw.point((0, 0), fill=tuple(car_image.getpixel((5,5))[:3]))
            img_pred.paste(car_image, (128//2-16//2, 256//2-16//2), car_image)

            draw = ImageDraw.Draw(img_pred_cal)
            # draw.point((0, 0), fill=tuple(car_image.getpixel((5,5))[:3]))
            img_pred_cal.paste(car_image, (128//2-16//2, 256//2-16//2), car_image)


        # After processing each image
        img_label = ImageOps.expand(img_label, border=border_width, fill=border_color)
        img_pred = ImageOps.expand(img_pred, border=border_width, fill=border_color)
        img_pred_cal = ImageOps.expand(img_pred_cal, border=border_width, fill=border_color)
        cams_img = ImageOps.expand(cams_img, border=border_width, fill=border_color)

        # Before combining the images
        img_combined_width = new_width + img_label.width + img_pred.width + img_pred_cal.width + border_width * 4
        img_combined = Image.new('RGB', (img_combined_width, img_label.height + border_width * 2))
        img_combined.paste(cams_img, (border_width, border_width))
        img_combined.paste(img_label, (new_width + border_width * 2, border_width))
        img_combined.paste(img_pred, (new_width + img_label.width + border_width * 3, border_width))
        img_combined.paste(img_pred_cal, (new_width + img_label.width + img_pred_cal.width + border_width * 4, border_width))

        # Append to images list
        # if kk<2:
        #     images_combined.append(img_combined)

        # Log the image
        # wandb.log({"my_image": wandb.Image(img_combined)})
        img_combined.save(f"combined_results_bev_pretrained/combined_{kk:04}.png")


        # clear the line and print metrics
        metrics = [
            f"mIoU: {np.nanmean(mIoU_list)*100:.1f}, length = {len(mIoU_list)}",
            f"lanes: {np.nanmean(lanes_list)*100:.1f}, length = {len(lanes_list)}",
            f"crosswalks: {np.nanmean(crosswalks_list)*100:.1f}, length = {len(crosswalks_list)}",
            f"road_boundary: {np.nanmean(road_boundary_list)*100:.1f}, length = {len(road_boundary_list)}",
            f"mIoU_cal: {np.nanmean(mIoU_list_cal)*100:.1f}, length = {len(mIoU_list_cal)}",
            f"lanes_cal: {np.nanmean(lanes_list_cal)*100:.1f}, length = {len(lanes_list_cal)}",
            f"crosswalks_cal: {np.nanmean(crosswalks_list_cal)*100:.1f}, length = {len(crosswalks_list_cal)}",
            f"road_boundary_cal: {np.nanmean(road_boundary_list_cal)*100:.1f}, length = {len(road_boundary_list_cal)}",
        ]

        # print metrics
        for metric in metrics:
            sys.stdout.write("\033[K")  # clear line
            print('\r'+metric)

        # move cursor up 8 lines
        for _ in range(8):
            sys.stdout.write("\033[F")  # back to previous line

        pbar.update(1)

        if kk == 0:

            # Get the height and width of the images
            width, height = img_combined.size

            # Create a VideoWriter object
            out = cv2.VideoWriter('output_bev_pretrained.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))

    # for image in images_combined:
        # Convert the image from RGB to BGR
        image = cv2.cvtColor(np.array(img_combined), cv2.COLOR_RGB2BGR)

        # Write the image to the video
        out.write(image)

    # Release the VideoWriter
    out.release()

    # print("mIoU: ", np.mean(mIoU_list))
    # print("lanes: ", np.mean(lanes_list))
    # print("crosswalks: ", np.mean(crosswalks_list))
    # print("road_boundary: ", np.mean(road_boundary_list))
    # print("mIoU: ", np.mean([np.mean(lanes_list), np.mean(crosswalks_list), np.mean(road_boundary_list)]))

    # print("mIoU_cal: ", np.mean(mIoU_list_cal))
    # print("lanes_cal: ", np.mean(lanes_list_cal))
    # print("crosswalks_cal: ", np.mean(crosswalks_list_cal))
    # print("road_boundary_cal: ", np.mean(road_boundary_list_cal))
    # print("mIoU_cal: ", np.mean([np.mean(lanes_list_cal), np.mean(crosswalks_list_cal), np.mean(road_boundary_list_cal)]))
    # Write gif file
    # imageio.mimsave(os.path.join(main_dir, "Temporal_combined2.gif"), images_combined, duration=10)


    # run.finish()
    metrics = [
        f"mIoU: {np.nanmean(mIoU_list)*100:.1f}, length = {len(mIoU_list)}",
        f"lanes: {np.nanmean(lanes_list)*100:.1f}, length = {len(lanes_list)}",
        f"crosswalks: {np.nanmean(crosswalks_list)*100:.1f}, length = {len(crosswalks_list)}",
        f"road_boundary: {np.nanmean(road_boundary_list)*100:.1f}, length = {len(road_boundary_list)}",
        f"mIoU_cal: {np.nanmean(mIoU_list_cal)*100:.1f}, length = {len(mIoU_list_cal)}",
        f"lanes_cal: {np.nanmean(lanes_list_cal)*100:.1f}, length = {len(lanes_list_cal)}",
        f"crosswalks_cal: {np.nanmean(crosswalks_list_cal)*100:.1f}, length = {len(crosswalks_list_cal)}",
        f"road_boundary_cal: {np.nanmean(road_boundary_list_cal)*100:.1f}, length = {len(road_boundary_list_cal)}",
    ]

    # print metrics
    for metric in metrics:
        print(metric)