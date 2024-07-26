import cv2
import torch
import copy
import random
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter, map_coordinates

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Prepare data transformations for the train loader
transform = T.Compose([
    T.Resize((224, 224)),                                                   # Resize each PIL image to 256 x 256
    # T.RandomCrop((224, 224)),                                               # Randomly crop a 224 x 224 patch
    T.ToTensor(),                                                           # Convert Numpy to Pytorch Tensor
    T.Normalize(mean, std)                                                  # Normalize with ImageNet mean
])

def get_WholeDataset(batch_size, img_root, num_workers=2):
    # Load data
    imagnet_dataset_4test = torchvision.datasets.ImageFolder(root=img_root, transform=transform)
    imagnet_dataset_4memo = torchvision.datasets.ImageFolder(root=img_root)

    # Initialize dataloaders
    test_loader_4test = torch.utils.data.DataLoader(imagnet_dataset_4test, batch_size, shuffle=False, num_workers=num_workers)

    return test_loader_4test, imagnet_dataset_4memo

def get_SubDataset(batch_size, img_root, subset_size, random_seed, num_workers=2):

    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Load the full dataset
    imagnet_dataset = torchvision.datasets.ImageFolder(root=img_root)

    indices = np.random.choice(len(imagnet_dataset), size=subset_size, replace=False)
    imagnet_subset = torch.utils.data.Subset(imagnet_dataset, indices)

    # Initialize dataloader
    test_loader = torch.utils.data.DataLoader(imagnet_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # print("full dataset: ", len(imagnet_dataset))
    # print("subset dataset: ", len(imagnet_subset))  

    return test_loader, imagnet_subset

def apply_augmentations(img, num_aug, centroid):
    ret_images = []
    ret_names = []

    ret_images.append(img)
    ret_names.append("Original")

    applied_augmentations = set()

    while len(ret_images) - 1 < num_aug:  # Exclude original image from count
        img_copy = img.copy()
        img_np = np.array(img_copy)

        n = random.randint(0, len(augmentations) - 1)
        aug_name = augmentations_names[n]

        # Check if the augmentation has been applied before
        if aug_name not in applied_augmentations:
            applied_augmentations.add(aug_name)

            # Apply augmentation
            if aug_name == "Crop":
                img_aug_np = augmentations[n](img_np, center=centroid)
            else:
                img_aug_np = augmentations[n](img_np)

            # Convert NumPy array back to PIL image
            img_aug = Image.fromarray(img_aug_np)

            ret_images.append(img_aug)
            ret_names.append(aug_name)

    # print("aug_applied: ", ret_names)
    return ret_images, ret_names

def rotation(image,  angle_range=(-10, 10)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def zoom(image, scale_range=(0.5, 1.5)):
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

def blur(image, kernel_size=(20, 20)):
    return cv2.blur(image, kernel_size)

def crop(image, crop_size=(160, 160), center=None):

    height, width = image.shape[:2]
    crop_height, crop_width = crop_size

    if center is None:
        center = (height // 2, width // 2)

    centroid_x, centroid_y = center
    centroid_resized = (int(centroid_x * image.shape[1] / 224), int(centroid_y * image.shape[0] / 224))

    center_x, center_y = centroid_resized

    y1 = max(0, center_y - crop_height // 2)
    y2 = min(height, center_y + crop_height // 2)
    x1 = max(0, center_x - crop_width // 2)
    x2 = min(width, center_x + crop_width // 2)

    if y2 - y1 < crop_height:
        if y1 == 0:
            y2 = min(height, y1 + crop_height)
        else:
            y1 = max(0, y2 - crop_height)

    if x2 - x1 < crop_width:
        if x1 == 0:
            x2 = min(width, x1 + crop_width)
        else:
            x1 = max(0, x2 - crop_width)


    return image[y1:y2, x1:x2]

def affine(image, pts1=None, pts2=None):
    rows, cols = image.shape[:2]
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

def translation(image, shift_range=(-100, 100)):
    x_shift = np.random.randint(shift_range[0], shift_range[1])
    y_shift = np.random.randint(shift_range[0], shift_range[1])
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(image, M, (cols, rows))

def elastic_transform(image, alpha = 1000, sigma = 20, alpha_affine = 20, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, 
                       [center_square[0]+square_size, center_square[1]-square_size], 
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted_image = np.empty_like(image)
    for i in range(shape[2]):
        distorted_image[..., i] = map_coordinates(image[..., i], indices, order=1, mode='reflect').reshape(shape[:2])
    
    return distorted_image

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

def gaussian_blur(image, kernel_size=(25, 25), sigma=0):
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
    elastic_transform,
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
    "Elastic transform",
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

def segment_every_aug_blur(aug, mask_generator):

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

def segment_original_blur(aug, mask_generator):
    ret_images = []

    for elem in aug:
        ret_images.append(aug)

    image = aug[0]
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


def segment_original_crop_plus_augmentation(aug, mask_generator):
    ret_images = []

    for elem in aug:
        ret_images.append(aug)

    image = aug[0]
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
    top_masks = filtered_masks[:5]
    # print(f"Selected top {len(top_masks)} masks")

    # Loop through each top mask and create separate images with blurred background
    for i, (original_index, mask, mask_size) in enumerate(top_masks):
        mask_np = mask["segmentation"]

        # Trova i bounding box della maschera
        mask_indices = np.argwhere(mask_np)
        y_min, x_min = mask_indices.min(axis=0)
        y_max, x_max = mask_indices.max(axis=0)
        
        # Ritaglia l'immagine originale utilizzando il bounding box
        cropped_image_np = image_np[y_min:y_max+1, x_min:x_max+1]
        segmented_image_pil = Image.fromarray(cropped_image_np)
        ret_images.append(segmented_image_pil)

        # # Crea una maschera inversa
        # inverse_mask_np = np.logical_not(mask_np)

        # # Applica la maschera inversa all'immagine sfocata
        # segmented_image = np.where(inverse_mask_np[:, :, None], blurred_image_np, image_np)
        # segmented_image_pil = Image.fromarray(segmented_image)
        # ret_images.append(segmented_image_pil)

    return ret_images



def segment_original_crop_only_segmentation(aug, mask_generator):
    ret_images = []

    # for elem in aug:
    #     ret_images.append(aug)
    ret_images.append(aug[0])

    image = aug[0]
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
    top_masks = filtered_masks[:3]
    # print(len(top_masks))
    # print(f"Selected top {len(top_masks)} masks")

    # Loop through each top mask and create separate images with blurred background
    for i, (original_index, mask, mask_size) in enumerate(top_masks):
        mask_np = mask["segmentation"]

        # Trova i bounding box della maschera
        mask_indices = np.argwhere(mask_np)
        y_min, x_min = mask_indices.min(axis=0)
        y_max, x_max = mask_indices.max(axis=0)
        
        # Ritaglia l'immagine originale utilizzando il bounding box
        cropped_image_np = image_np[y_min:y_max+1, x_min:x_max+1]
        segmented_image_pil = Image.fromarray(cropped_image_np)
        ret_images.append(segmented_image_pil)

        # # Crea una maschera inversa
        # inverse_mask_np = np.logical_not(mask_np)

        # # Applica la maschera inversa all'immagine sfocata
        # segmented_image = np.where(inverse_mask_np[:, :, None], blurred_image_np, image_np)
        # segmented_image_pil = Image.fromarray(segmented_image)
        # ret_images.append(segmented_image_pil)
    cv2.imwrite('/home/sagemaker-user/DeepLearning_project/output.jpg', image_np)
    return ret_images



def segment_original_blackBG_only_segmentation(aug, mask_generator):
    ret_images = []

    # for elem in aug:
    #     ret_images.append(aug)
    ret_images.append(aug[0])

    image = aug[0]
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
    top_masks = filtered_masks[:3]
    # print(len(top_masks))
    # print(f"Selected top {len(top_masks)} masks")
    black_background_np = np.zeros_like(image_np)
    # Loop through each top mask and create separate images with blurred background
    for i, (original_index, mask, mask_size) in enumerate(top_masks):
        mask_np = mask["segmentation"]

        inverse_mask_np = np.logical_not(mask_np)
        segmented_image = np.where(inverse_mask_np[:, :, None], black_background_np, image_np)
        
        segmented_image_pil = Image.fromarray(segmented_image)
        ret_images.append(segmented_image_pil)

        # # Crea una maschera inversa
        # inverse_mask_np = np.logical_not(mask_np)

        # # Applica la maschera inversa all'immagine sfocata
        # segmented_image = np.where(inverse_mask_np[:, :, None], blurred_image_np, image_np)
        # segmented_image_pil = Image.fromarray(segmented_image)
        # ret_images.append(segmented_image_pil)
    cv2.imwrite('/home/sagemaker-user/DeepLearning_project/output.jpg', image_np)

    return ret_images





def segment_original_blackBG_only_segmentation_andGC(aug, mask_generator, centroid):
    ret_images = []
    original_image = aug[0]
    ret_images.append(original_image)
    
    # Convert and resize image
    image_np = np.array(original_image)
    masks = mask_generator.generate(image_np)

    min_mask_size = 500  # Adjustable minimum mask size

    # Filter out small masks and calculate areas of valid masks
    filtered_masks = [(i, mask, np.sum(mask["segmentation"])) for i, mask in enumerate(masks) if np.sum(mask["segmentation"]) >= min_mask_size]

    # Sort masks by area in descending order
    filtered_masks.sort(key=lambda x: x[2], reverse=True)

    # Keep only the top 10 masks
    top_masks = filtered_masks[:10]

    # Draw centroid on the resized image
    centroid_x, centroid_y = centroid
    centroid_resized = (int(centroid_x * image_np.shape[1] / 224), int(centroid_y * image_np.shape[0] / 224))
    cv2.circle(image_np, centroid_resized, radius=5, color=(0, 255, 0), thickness=-1)
    # cv2.imwrite('/home/sagemaker-user/DeepLearning_project/output.jpg', image_np)

    black_background_np = np.zeros_like(image_np)
    mask_found = False
    closest_mask = None
    min_distance = float('inf')

    # Process each mask
    for i, (original_index, mask, mask_size) in enumerate(top_masks):
        mask_np = mask["segmentation"]
        
        # Check if the centroid is within the mask
        if mask_np[centroid_resized[1], centroid_resized[0]]:
            mask_found = True
            inverse_mask_np = np.logical_not(mask_np)
            segmented_image = np.where(inverse_mask_np[:, :, None], black_background_np, image_np)
            segmented_image_pil = Image.fromarray(segmented_image)
            ret_images.append(segmented_image_pil)
        
        # Calculate the centroid of the mask
        mask_indices = np.argwhere(mask_np)
        if mask_indices.size > 0:
            mask_centroid = mask_indices.mean(axis=0)
            distance = np.linalg.norm(np.array(centroid_resized) - mask_centroid)
            if distance < min_distance:
                min_distance = distance
                closest_mask = (original_index, mask_np)

    # If no mask contains the centroid, select the closest and largest mask
    if not mask_found and closest_mask:
        _, mask_np = closest_mask
        inverse_mask_np = np.logical_not(mask_np)
        segmented_image = np.where(inverse_mask_np[:, :, None], black_background_np, image_np)
        segmented_image_pil = Image.fromarray(segmented_image)
        ret_images.append(segmented_image_pil)

    return ret_images


def segment_original_blackBG_only_segmentation_and_GC_bBOX(aug, mask_generator, centroid):
    ret_images = []
    original_image = aug[0]
    ret_images.append(original_image)
    
    # Convert and resize image
    image_np = np.array(original_image)
    masks = mask_generator.generate(image_np)

    min_mask_size = 500  # Adjustable minimum mask size

    # Filter out small masks and calculate areas of valid masks
    filtered_masks = [(i, mask, np.sum(mask["segmentation"])) for i, mask in enumerate(masks) if np.sum(mask["segmentation"]) >= min_mask_size]

    # Sort masks by area in descending order
    filtered_masks.sort(key=lambda x: x[2], reverse=True)

    # Keep only the top 10 masks
    top_masks = filtered_masks[:10]
    
    black_background_np = np.zeros_like(image_np)

    # Draw centroid on the resized image
    centroid_x, centroid_y = centroid
    centroid_resized = (int(centroid_x * image_np.shape[1] / 224), int(centroid_y * image_np.shape[0] / 224))
    image_copy = copy.deepcopy(image_np)
    cv2.circle(image_copy, centroid_resized, radius=10, color=(0, 255, 0), thickness=-1)
    cv2.imwrite('/home/sagemaker-user/DeepLearning_project/output.jpg', image_copy)

    mask_found = False
    closest_mask = None
    min_distance = float('inf')
    finded_masks = []

    # Process each mask
    for i, (original_index, mask, mask_size) in enumerate(top_masks):
        mask_np = mask["segmentation"]
        
        # Check if the centroid is within the mask
        if mask_np[centroid_resized[1], centroid_resized[0]]:
            mask_found = True
            finded_masks.append(mask_np)
        
        # Calculate the centroid of the mask
        mask_indices = np.argwhere(mask_np)
        if mask_indices.size > 0:
            mask_centroid = mask_indices.mean(axis=0)
            distance = np.linalg.norm(np.array(centroid_resized) - mask_centroid)
            if distance < min_distance:
                min_distance = distance
                closest_mask = (original_index, mask_np)

    # If no mask contains the centroid, select the closest and largest mask
    if not mask_found and closest_mask:
        _, mask_np = closest_mask
        finded_masks.append(mask_np)
    
    for i, elem in enumerate(finded_masks):
        mask_np = np.array(elem)
        
        # Find the bounding rectangle for the mask
        mask_indices = np.argwhere(mask_np)
        if mask_indices.size > 0:
            y_min, x_min = mask_indices.min(axis=0)
            y_max, x_max = mask_indices.max(axis=0)
            
            # Create an empty black background
            black_background_np = np.zeros_like(image_np)
            
            # Copy the content of the bounding box from the original image
            black_background_np[y_min:y_max+1, x_min:x_max+1] = image_np[y_min:y_max+1, x_min:x_max+1]
            
            # Convert to PIL Image and append to ret_images
            ret_images.append(Image.fromarray(black_background_np))
    
    return ret_images