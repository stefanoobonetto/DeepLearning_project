import cv2
import torch
import random
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torchvision.transforms as T

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Prepare data transformations for the train loader
transform = T.Compose([
    T.Resize((256, 256)),                                                   # Resize each PIL image to 256 x 256
    T.RandomCrop((224, 224)),                                               # Randomly crop a 224 x 224 patch
    T.ToTensor(),                                                           # Convert Numpy to Pytorch Tensor
    T.Normalize(mean, std)                                                  # Normalize with ImageNet mean
])


def get_dataset(batch_size, img_root, num_workers=2):
    # Load data
    imagnet_dataset_4test = torchvision.datasets.ImageFolder(root=img_root, transform=transform)
    imagnet_dataset_4memo = torchvision.datasets.ImageFolder(root=img_root)

    # Initialize dataloaders
    test_loader_4test = torch.utils.data.DataLoader(imagnet_dataset_4test, batch_size, shuffle=False, num_workers=num_workers)

    return test_loader_4test, imagnet_dataset_4memo

def apply_augmentations(img, num_aug):
    ret_images = []
    ret_names = []

    ret_images.append(img)
    ret_names.append("Original")

    for i in range(num_aug):
        img_copy = img.copy()

        # Convert PIL image to NumPy array
        img_np = np.array(img_copy)

        n = random.randint(0, len(augmentations)-1)

        # Apply augmentation
        img_aug_np = augmentations[n](img_np)

        # Convert NumPy array back to PIL image
        img_aug = Image.fromarray(img_aug_np)

        ret_images.append(img_aug)
        ret_names.append(augmentations_names[n])

    return ret_images, ret_names

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

augmentations_names = [
    "Rotation",
    "Zoom",
    "Horizontal flip",
    "Vertical flip",
    "Greyscale",
    "Inverse",
    "Blur",
    "Crop",
    "Affine",
    "Change gamma",
    "Translation",
    "Scale",
    "Brightness",
    "Histogram equalization",
    "Salt and pepper",
    "Gaussian blur",
    "Poisson noise",
    "Speckle noise",
    "Contrast",
]

def transform_images(images):
    if isinstance(images, list):
        transformed_images = []
        for img in images:
            transformed_img = transform(img)
            transformed_images.append(transformed_img)
        return transformed_images
    else:
        return transform(images)

def segment_images(aug, mask_generator):

    ret_images = []

    for image in aug:
        image_np = np.array(image)
        masks = mask_generator.generate(image_np)
        # print(f"Generated {len(masks)} masks")

        min_mask_size = 500  # Puoi regolare questo valore in base alle tue esigenze

        # Filtra le maschere troppo piccole e calcola le aree delle maschere valide
        filtered_masks = [(i, mask, np.sum(mask["segmentation"])) for i, mask in enumerate(masks) if np.sum(mask["segmentation"]) >= min_mask_size]
        # print(f"Filtered to {len(filtered_masks)} masks larger than {min_mask_size} pixels")

        # Ordina le maschere per area in ordine decrescente
        filtered_masks.sort(key=lambda x: x[2], reverse=True)

        # Tieni solo le prime 10 maschere
        top_masks = filtered_masks[:10]
        # print(f"Selected top {len(top_masks)} masks")

        # Creare una versione sfocata dell'intera immagine
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=15))
        blurred_image_np = np.array(blurred_image)

        # Loop through each top mask and create separate images with blurred background
        for i, (original_index, mask, mask_size) in enumerate(top_masks):
            mask_np = mask["segmentation"]

            # Crea una maschera inversa
            inverse_mask_np = np.logical_not(mask_np)

            # Applica la maschera inversa all'immagine sfocata
            segmented_image = np.where(inverse_mask_np[:, :, None], blurred_image_np, image_np)
            segmented_image_pil = Image.fromarray(segmented_image)
            ret_images.append(segmented_image_pil)

    return ret_images




