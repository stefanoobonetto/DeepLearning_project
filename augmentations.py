import cv2
import numpy as np
import os
import random 

def get_random_imagenet_a_image(dataset_dir):
    # imagenet-a
    # ├── class1
    # │   ├── image1.jpg
    # │   └── ...
    # ├── class2
    # │   ├── image1.jpg
    # │   └── ...
    # └── ...

    classes = os.listdir(dataset_dir)
    random_class = random.choice(classes)
    class_dir = os.path.join(dataset_dir, random_class)
    images = os.listdir(class_dir)

    random_image = random.choice(images)

    random_image_path = os.path.join(class_dir, random_image)

    return random_image_path

def rotation(image, angle_range=(-10, 10)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def zoom(image, scale_range=(0.8, 1.2)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, 0, scale_factor)
    zoomed_image = cv2.warpAffine(image, M, (width, height))
    return zoomed_image

def flip_horizontal(image):
    return cv2.flip(image, 1)

def flip_vertical(image):
    return cv2.flip(image, 0)

def greyscale(image):
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey_img, cv2.COLOR_GRAY2BGR)

def inverse(image):
    return cv2.bitwise_not(image)

def blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def crop(image, crop_size=(100, 100)):
    return image[:crop_size[0], :crop_size[1]]

def affine(image, pts1=None, pts2=None):
    rows, cols = image.shape[:2]  # Get the dimensions of the input image
    if pts1 is None:
        pts1 = np.float32([[50,50],[200,50],[50,200]])
    if pts2 is None:
        pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(image,M,(cols,rows))

def change_gamma(image, gamma_range=(0.5, 2.0)):
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def translation(image, shift_range=(-10, 10)):
    x_shift = np.random.randint(shift_range[0], shift_range[1])
    y_shift = np.random.randint(shift_range[0], shift_range[1])
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(image, M, (cols, rows))

def scale(image, scale_factor_range=(0.5, 1.5)):
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    height, width = image.shape[:2]
    scaled_height, scaled_width = int(height * scale_factor), int(width * scale_factor)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))
    return scaled_image

def brightness(image, brightness_range=(-50, 50)):
    brightness_value = np.random.randint(brightness_range[0], brightness_range[1])
    return np.clip(image.astype(int) + brightness_value, 0, 255).astype(np.uint8)

def histogram_eq(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    return image_rgb

def salt_e_pepper(image, amount=0.01):
    noise = np.random.rand(*image.shape)
    salt = noise > 1 - amount / 2
    pepper = noise < amount / 2
    noisy_image = image.copy()
    noisy_image[salt] = 255
    noisy_image[pepper] = 0
    return noisy_image

def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def poisson_noise(image):
    noisy_image = np.random.poisson(image / 255.0 * 25) / 25 * 255
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def speckle_noise(image):
    noise = np.random.randn(*image.shape)
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def contrast(image, alpha_range=(0.5, 1.5)):
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    return np.clip(image.astype(int) * alpha, 0, 255).astype(np.uint8)

augmentations = [
    rotation,
    zoom,
    flip_horizontal,
    flip_vertical,
    greyscale,
    inverse,
    blur,
    crop,
    affine,
    change_gamma,
    translation,
    scale,
    brightness,
    histogram_eq,
    salt_e_pepper,
    gaussian_blur,
    poisson_noise,
    speckle_noise,
    contrast,
]
