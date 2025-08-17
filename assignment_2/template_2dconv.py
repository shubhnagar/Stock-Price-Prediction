import numpy as np
from helper import *

def movePatchOverImg(image, filter_size, apply_filter_to_patch):
    if len(image.shape) == 3: 
        image = (0.299 * image[:, :, 0] +  
                           0.587 * image[:, :, 1] + 
                           0.114 * image[:, :, 2])   
        
    pad_size = filter_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    output_image = np.zeros_like(image)
    
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            patch = padded_image[i:i + filter_size, j:j + filter_size]
            output_image[i, j] = apply_filter_to_patch(patch)
    
    return output_image

def detect_vertical_edge(image_patch):
    vert_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    outputval = np.sum(image_patch * vert_kernel)
    return outputval

def detect_horizontal_edge(image_patch):
    hori_kernel = np.array([
        [1,  2,  1],
        [0,  0,  0],
        [-1, -2, -1]
    ])
    
    outputval = np.sum(image_patch * hori_kernel)
    return outputval

def detect_all_edges(image_patch):
    all_edges_kernel = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ])
    
    outputval = np.sum(image_patch * all_edges_kernel)
    return outputval

def remove_noise(image_patch):
    outputval = np.median(image_patch)
    return outputval

def create_gaussian_kernel(size, sigma):
    """
    Generates a Gaussian kernel of the specified size and sigma.
    
    Parameters:
    - size (int): The size of the kernel (must be an odd number).
    - sigma (float): The standard deviation of the Gaussian distribution.
    
    Returns:
    - output_kernel (np.array): A normalized 2D Gaussian kernel.
    """
    center = size // 2
    
    output_kernel = np.zeros((size, size), dtype=np.float32)

    c = 1 / (2 * np.pi * sigma**2)
    for x in range(size):
        for y in range(size):
            x_dist = (x - center) ** 2
            y_dist = (y - center) ** 2
            output_kernel[x, y] = c * np.exp(-(x_dist + y_dist) / (2 * sigma ** 2))
    
    output_kernel /= np.sum(output_kernel)
    
    return output_kernel


def gaussian_blur(image_patch):
    """
    Applies Gaussian blur to a given image patch.
    
    Parameters:
    - image_patch (np.array): A 2D square patch of the image.
    
    Returns:
    - outputval (float): The blurred pixel value after applying Gaussian blur.
    """
    kernel = create_gaussian_kernel(size=25, sigma=1)
    
    return np.sum(image_patch * kernel)


def unsharp_masking(image, scale):
    """
    Perform unsharp masking on an image using Gaussian blur to enhance sharpness.
    
    Parameters:
    - image (np.array): Input grayscale image.
    - scale (float): Scaling factor for sharpening effect.
    
    Returns:
    - out (np.array): Sharpened image.
    """
    if len(image.shape) == 3: 
        image = (0.299 * image[:, :, 0] +  
                           0.587 * image[:, :, 1] + 
                           0.114 * image[:, :, 2])  
    
    blurred_image = movePatchOverImg(image, 25, gaussian_blur)
    sharpened_image = image + scale * (image - blurred_image)
    out = np.clip(sharpened_image, 0, 255)
    
    return out

#TASK 1  
# img=load_image("cutebird.png")
img=load_image("assgmt2/cutebird.png")
filter_size=3 #You may change this to any appropriate odd number
hori_edges = movePatchOverImg(img, filter_size, detect_horizontal_edge)
save_image("hori.png",hori_edges)
print("Horizantal edge detection done")
filter_size=3 #You may change this to any appropriate odd number
vert_edges = movePatchOverImg(img, filter_size, detect_vertical_edge)
save_image("vert.png",vert_edges)
print("Vertical edge detection done")
filter_size=3 #You may change this to any appropriate odd number
all_edges = movePatchOverImg(img, filter_size, detect_all_edges)
save_image("alledge.png",all_edges)
print("All edge detection done")

#TASK 2
# noisyimg=load_image("noisycutebird.png")
noisyimg=load_image("assgmt2/noisycutebird.png")
filter_size=3 #You may change this to any appropriate odd number
denoised = movePatchOverImg(noisyimg, filter_size, remove_noise)
save_image("denoised.png",denoised)
print("Denoising done")

#TASK 3
scale= 1.5#You may use any appropriate positive number (ideally between 1 and 3)
save_image("unsharpmask.png",unsharp_masking(img,scale))
print("Upsharping done")
