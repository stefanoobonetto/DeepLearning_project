from utils import *
from model import *
from gradcam import *
from functions import *

import os
import csv
import torch
from tqdm import tqdm
import torch.optim as optim
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import matplotlib.pyplot as plt
from PIL import Image
import shutil

# Check for the appropriate device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")



def summation(lst):
    """
    Sums up the first elements of tuples in a list.
    
    Args:
        lst (list): List of tuples.
        
    Returns:
        int: The sum of the first elements.
    """
    return sum(elem[0] for elem in lst)

def main(colab=False):
    """
    Main function to run the training and evaluation process.
    
    Args:
        colab (bool): Flag to indicate if running on Google Colab.
    """
    # Define paths and directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pathDatasetImagenetA = os.path.join(current_dir, "datasets/imagenet-a")
    checkpoint_path = os.path.join(current_dir, "weights/sam_vit_b_01ec64.pth")

    # Load dataset
    _, test_data = get_WholeDataset(batch_size=1, img_root=pathDatasetImagenetA)

    # Initialize ResNet50 model and save initial weights
    model = ModelVitb16().to(device)
    target_layers = [model.vitb.encoder.layers[-1].ln_1]
    
    initial_weights_path = os.path.join(current_dir, "weights/weights_model_in_use.pth")
    torch.save(model.state_dict(), initial_weights_path)

    # Define optimizer
    lr = 0.00025
    weight_decay = 0.0
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Load segmentation model
    model_type = "vit_b"
    segmentation_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(segmentation_model)

    # Lists to store results
    correct_before_memo = []
    correct_after_memo = []
    correct_after_memo_plus = []

    # Process test data
    pbar = tqdm(range(len(test_data)))  # Progress bar
    for i in pbar:
        # Load an image and its target
        image, target = test_data[i]

        model.load_state_dict(torch.load(initial_weights_path))
        model.eval()

        # Test the model before applying MEMO
        correct_before_memo.append(test_model(image=image, target=target, model=model))

        # Create Grad-CAM before MEMO
        gradcam_initial, centroid, regions = create_gradcam(np.float32(image) / 255, model, target_layers)

        # Tune the model using MEMO
        augmentation, name_aug = tune_model(image=image, model=model, mask_generator=mask_generator, optimizer=optimizer, cost_function=marginal_entropy, num_aug=8, flag_memo_plus=False, centroid=centroid)

        # Test the model after applying MEMO
        correct_after_memo.append(test_model(image=image, target=target, model=model))

        # Create Grad-CAM after MEMO
        gradcam_memo, centroid, regions = create_gradcam(np.float32(image) / 255, model, target_layers)

        # Tune the model using MEMO_PLUS
        segmentation, name_aug_plus = tune_model(image=image, model=model, mask_generator=mask_generator, optimizer=optimizer, cost_function=marginal_entropy, num_aug=8, flag_memo_plus=True, centroid=centroid)

        # Test the model after applying MEMO_PLUS
        correct_after_memo_plus.append(test_model(image=image, target=target, model=model))

        # Create Grad-CAM after MEMO_PLUS
        gradcam_memo_plus, _, _ = create_gradcam(np.float32(image) / 255, model, target_layers)

        # Calculate accuracies
        accuracy_before_memo = (summation(correct_before_memo) / len(correct_before_memo)) * 100
        accuracy_after_memo = (summation(correct_after_memo) / len(correct_after_memo)) * 100
        accuracy_after_memo_plus = (summation(correct_after_memo_plus) / len(correct_after_memo_plus)) * 100

        # Update progress bar with current accuracy
        pbar.set_description(f'Before MEMO: {accuracy_before_memo:.2f}%  After MEMO: {accuracy_after_memo:.2f}% After MEMO_PLUS: {accuracy_after_memo_plus:.2f}%')

    # Print final results
    print(f'Before MEMO: {accuracy_before_memo:.2f}%  After MEMO: {accuracy_after_memo:.2f}% After MEMO_PLUS: {accuracy_after_memo_plus:.2f}%')



if __name__ == "__main__":
    colab = False
    main(colab=colab)
