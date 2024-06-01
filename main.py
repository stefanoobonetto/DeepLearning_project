from utils import *
from model import *
from functions import *

import os
import csv
import torch
from tqdm import tqdm
import torch.optim as optim
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

#sudo apt install libgl1-mesa-glx

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Funzione per salvare i dati di accuratezza in un file CSV
def save_accuracy_to_csv(accuracy_classes, output_path, accuracy_before_memo,accuracy_after_memo,accuracy_after_memo_plus):
    # Creare il file se non esiste
    file_exists = os.path.isfile(output_path)
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Accuracy", "Result_for_image", "Augmentation_for_image" ])
        for elem in accuracy_classes:
            if len(accuracy_classes[elem]["prediction"]) > 0:
                # accuracy = round((sum(accuracy_classes[elem]["prediction"]) / len(accuracy_classes[elem]["prediction"])) * 100, 2)
                accuracy = round(np.mean(accuracy_classes[elem]["prediction"])*100, 2)
                writer.writerow([elem, accuracy,accuracy_classes[elem]['prediction'],accuracy_classes[elem]['augmentation']])
                
        writer.writerow([accuracy_before_memo, accuracy_after_memo, accuracy_after_memo_plus])

def main(colab=False):
    # Define paths and directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pathDatasetImagenetA = os.path.join(current_dir, "datasets/imagenet-a")
    checkpoint_path = os.path.join(current_dir, "weights/sam_vit_b_01ec64.pth")
    output_csv_path = os.path.join(current_dir, "Results/test_5_all_dataset_VITB_before_memo_after_segmentation_only.csv")
    
    # List to store augmentation results
    total_aug = []

    # Load dataset
    _, test_data = get_dataset(batch_size=1, img_root=pathDatasetImagenetA)

    # Number of augmentations
    num_aug = 8

    # Initialize ResNet50 model and save initial weights
    #model = ModelVitb16.to(devsegment_original_cropice)
    model = ModelVitb16().to(device)
    
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
    accuracy_classes = {}
    for i in range(200): 
        accuracy_classes[f"{i}_before_MEMO"] = {"prediction":[], "augmentation":[]}
        accuracy_classes[f"{i}_after_MEMO"] = {"prediction":[], "augmentation":[]}
        accuracy_classes[f"{i}_after_MEMO_PLUS"] = {"prediction":[], "augmentation":[]}

    

    # Process test data
    pbar = tqdm(range(len(test_data)))  # Adjust the range as necessary
    for i in pbar:
        # Reset model to initial weights before each iteration
        model.load_state_dict(torch.load(initial_weights_path))
        model.eval()
        
        # Load an image and its target
        image, target = test_data[i]

        # Test the model before applying MEMO
        correct_before_memo.append(test_model(image=image, target=target, model=model))

        # Tune the model using MEMO
        augmentation = tune_model(image=image, model=model, mask_generator=mask_generator, optimizer=optimizer, cost_function=marginal_entropy, num_aug=num_aug, flag_memo_plus=False)

        # Test the model after applying MEMO
        correct_after_memo.append(test_model(image=image, target=target, model=model))

        # model.load_state_dict(torch.load(initial_weights_path))
        # model.eval()

        # Tune the model using MEMO
        augmentation_plus = tune_model(image=image, model=model, mask_generator=mask_generator, optimizer=optimizer, cost_function=marginal_entropy, num_aug=num_aug, flag_memo_plus=True)

        # Test the model after applying MEMO
        correct_after_memo_plus.append(test_model(image=image, target=target, model=model))
        

        # accuracy
        accuracy_before_memo = np.mean(correct_before_memo)*100
        accuracy_after_memo = np.mean(correct_after_memo)*100
        accuracy_after_memo_plus = np.mean(correct_after_memo_plus)*100
        #accuracy_before_memo = (sum(correct_before_memo)/len(correct_before_memo))*100
        #accuracy_after_memo = (sum(correct_after_memo)/len(correct_after_memo))*100


        accuracy_classes[f"{target}_before_MEMO"]["prediction"].append(correct_before_memo[i])
        accuracy_classes[f"{target}_after_MEMO"]["prediction"].append(correct_after_memo[i])
        accuracy_classes[f"{target}_after_MEMO_PLUS"]["prediction"].append(correct_after_memo_plus[i])

        accuracy_classes[f"{target}_after_MEMO"]["augmentation"].append(augmentation)
        accuracy_classes[f"{target}_after_MEMO_PLUS"]["augmentation"].append(augmentation_plus)
        
        # Salvataggio dei dati di accuratezza in un file CSV ogni 100 epoche
        if( i % 5 == 0):
            save_accuracy_to_csv(accuracy_classes, output_csv_path, accuracy_before_memo,accuracy_after_memo,accuracy_after_memo_plus)

        # Update progress bar with current accuracy
        pbar.set_description(f'Before MEMO: {accuracy_before_memo:.2f}%  after MEMO: {accuracy_after_memo:.2f}% after MEMO_PLUS: {accuracy_after_memo_plus:.2f}%')



    # Print final results
    print(f'Before MEMO: {accuracy_before_memo:.2f}%  after MEMO: {accuracy_after_memo:.2f}% after MEMO_PLUS: {accuracy_after_memo_plus:.2f}%')

    # Salvataggio dei dati di accuratezza in un file CSV
    save_accuracy_to_csv(accuracy_classes, output_csv_path, accuracy_before_memo,accuracy_after_memo,accuracy_after_memo_plus)
    print(f"Accuracy results saved to {output_csv_path}")


if __name__ == "__main__":
    colab = False
    main(colab=colab)
