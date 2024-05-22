from utils import *
from model import *
from functions import *

import os
from tqdm import tqdm
import torch
import torch.optim as optim
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def main(colab=False):
    # Define paths and directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pathDatasetImagenetA = os.path.join(current_dir, "datasets/imagenet-a")
    checkpoint_path = os.path.join(current_dir, "weights/sam_vit_b_01ec64.pth")
    
    # List to store augmentation results
    total_aug = []

    # Load dataset
    _, test_data = get_dataset(batch_size=1, img_root=pathDatasetImagenetA)

    # Number of augmentations
    num_aug = 8

    # Initialize ResNet50 model and save initial weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    
    initial_weights_path = os.path.join(current_dir, "weights/resnet50_weights.pth")
    torch.save(model.state_dict(), initial_weights_path)
    initial_weights = model.state_dict()

    # Define optimizer
    lr = 0.00025
    weight_decay = 0.0
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Load segmentation model
    model_type = "vit_b"
    segmentation_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(segmentation_model)

    #  # Instantiates dataloaders
    # batch_size = 1
    # test_loader, _ = get_dataset(batch_size=batch_size, img_root=pathDatasetImagenetA)
    # # Instantiates the model
    # model = ModelResNet().to(device)
    # # Instantiates the cost function
    # cost_function = torch.nn.CrossEntropyLoss()

    # # Run a single test step beforehand and print metrics
    # print("Before training:")
    # test_loss, test_accuracy = test_all_dataset(model, test_loader, cost_function)
    # print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")

    # Lists to store results
    correct_before_memo = []
    correct_after_memo = []

    # Process test data
    pbar = tqdm(range(77))  # Adjust the range as necessary
    for i in pbar:
        # Reset model to initial weights before each iteration
        #model = ModelResNet().to(device)
        model.load_state_dict(torch.load(initial_weights_path))
        model.eval()
        
        # Load an image and its target
        image, target = test_data[i]

        # Test the model before applying MEMO
        correct_before_memo.append(test_model(image=image, target=target, model=model))

        # Tune the model using MEMO
        tune_model(image=image, model=model, mask_generator=mask_generator, optimizer=optimizer, cost_function=marginal_entropy, num_aug=num_aug)

        # Test the model after applying MEMO
        correct_after_memo.append(test_model(image=image, target=target, model=model))

        # print(correct_before_memo[i], correct_after_memo[i])

        # Update progress bar with current accuracy
        pbar.set_description(f'Before MEMO accuracy: {np.mean(correct_before_memo)*100:.2f}%  after MEMO accuracy: {np.mean(correct_after_memo)*100:.2f}%')

    # Print final results
    print(f'Before MEMO accuracy: {(sum(correct_before_memo)/len(correct_before_memo))*100:.2f}%  after MEMO accuracy: {np.mean(correct_after_memo)*100:.2f}%')
    print(correct_before_memo)
    print(correct_after_memo)

if __name__ == "__main__":
    colab = False
    main(colab=colab)
