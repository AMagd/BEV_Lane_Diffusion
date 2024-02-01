import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def total_variation_loss(image, scale_factor=1.0):
    diff_i = torch.abs(image[:, :, 1:] - image[:, :, :-1])
    diff_j = torch.abs(image[:, :, 1:] - image[:, :, :-1])
    tv = torch.sum(diff_i) + torch.sum(diff_j)
    eps = 1e-8
    tv = torch.where(tv < eps, torch.zeros_like(tv), tv)
    loss = torch.where(tv > 0, 1.0 / (scale_factor * tv), torch.zeros_like(tv))
    return loss

def thickness_loss(image, thickness_threshold):
    # Convert to torch tensor if it's not
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        
    # Create a thickness map
    filter_size = thickness_threshold
    filter = torch.ones((filter_size, filter_size))
    
    # Ensure image and filter are of correct dimensions
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    if len(filter.shape) == 2:
        filter = filter.unsqueeze(0).unsqueeze(0)
    
    # Use convolution to simulate dilation
    thickness_map = F.conv2d(image, filter, padding=filter_size // 2)

    # Normalized thickness map by filter area
    thickness_map = thickness_map / (filter_size ** 2)

    # Calculate mask for thickness > 1
    mask = thickness_map > 250
    over_threshold = thickness_map * mask

    # The loss is the sum of the over_threshold map
    loss = over_threshold.sum()
    
    # Apply exponential function to the areas over the threshold
    scale_factor = 0.000001  # Modify this value to control the steepness of the increase
    loss = torch.exp(scale_factor * loss) - 1.0

    return loss, thickness_map, mask



size = (1, 1, 200, 200)
image = np.zeros(size, dtype=np.uint8)

torch.manual_seed(0)
for k in range(20):
    image = np.zeros(size, dtype=np.uint8)
    print("Thickness: ", k+1)
    for i in range(10):
        start_point = tuple(torch.randint(0, 199, (2,)).tolist())
        end_point = tuple(torch.randint(0, 199, (2,)).tolist())
        thickness = k+1
        cv2.line(image[0,0], start_point, end_point, 255, thickness)

    # Create a mask for thickness > 7
    image_mask = np.copy(image[0, 0])
    kernel = np.ones((7,7),np.uint8)
    image_mask = cv2.dilate(image_mask, kernel, iterations = 1)

    image_mask[image_mask < 255] = 0
    image_mask[image_mask == 255] = 1

    # Convert back to torch tensor
    image = torch.from_numpy(image).float()
    image_mask = torch.from_numpy(image_mask).float()

    filter_size = 3
    filter = torch.ones((filter_size, filter_size)) / (filter_size ** 2)
    thickness_map = F.conv2d(image, filter.unsqueeze(0).unsqueeze(0), padding=filter_size // 2)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    loss_thickness, thickness_map, image_mask = thickness_loss(image.numpy(), thickness_threshold=7)

    axs[0].imshow(image[0, 0].cpu().numpy(), cmap='gray')
    axs[0].set_title('Lane Map')

    axs[1].imshow(thickness_map.squeeze().cpu().numpy(), cmap='jet')
    axs[1].set_title('Thickness Map')

    axs[2].imshow(image_mask.squeeze().cpu().numpy(), cmap='gray')
    axs[2].set_title('Mask (Thickness > 7)')

    total_variation = total_variation_loss(image)
    loss_variation = total_variation

    # thickness_threshold = 7
    # thickness_estimate = torch.sqrt((thickness_map * thickness_map).sum(dim=1, keepdim=True))
    # over_threshold = torch.clamp(thickness_estimate - thickness_threshold, min=0)
    # penalty = torch.exp(0.0015 * over_threshold).mean()

    # print("Total Variation Loss: ", loss_variation.item())
    print("Thickness Penalty Loss: ", loss_thickness.item())

plt.tight_layout()
plt.show()

