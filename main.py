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

# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# sudo apt install libgl1-mesa-glx
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
# mkdir -p datasets
# tar -xf imagenet-a.tar -C datasets

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




def save_report_image(image=None, augmentation=None, segmentation=None, gradcam_original=None, gradcam_memo=None, gradcam_memo_plus=None, output_path=None, n_image = 0):
    # Create directory if it does not exist
    output_path = f"{output_path}/image_{n_image}"
    # print(output_path)
 
    os.makedirs(output_path, exist_ok=True)
    

    if augmentation:
        # Determine the size of each augmentation image (assuming they are all the same size)
        width, height = augmentation[0].size

        # Create a figure to contain the 9 augmentation images in a 3x3 grid with spaces between them
        fig, axes = plt.subplots(3, 3, figsize=(width*3/100, height*3/100), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

        # Hide axes
        for ax in axes.flatten():
            ax.axis('off')

        # Paste each augmentation image into the grid
        for i, aug_img in enumerate(augmentation):
            row = i // 3
            col = i % 3
            axes[row, col].imshow(aug_img)

        # Save the combined image (the grid of augmentations)
        grid_output_path = os.path.join(os.path.dirname(output_path), f'image_{n_image}/augmentation_grid.png')
        # print(grid_output_path)
        plt.savefig(grid_output_path, bbox_inches='tight')
        plt.close(fig)

    if segmentation:
        # Determine the number of segmentation images
        num_seg_images = len(segmentation)
        rows = (num_seg_images + 1) // 2  # 2 images per row

        # Create a figure to contain the segmentation images in a grid with spaces between them
        fig, axes = plt.subplots(rows, 2, figsize=(width*2/100, height*rows/100), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

        # If there's only one row, axes might not be a 2D array
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        # Hide axes
        for ax in axes.flatten():
            ax.axis('off')

        # Paste each segmentation image into the grid
        for i, seg_img in enumerate(segmentation):
            row = i // 2
            col = i % 2
            axes[row, col].imshow(seg_img)

        # Save the combined image (the grid of segmentation images)
        seg_output_path = os.path.join(os.path.dirname(output_path), f'image_{n_image}/segmentation_grid.png')
        plt.savefig(seg_output_path, bbox_inches='tight')
        plt.close(fig)

    # Save the other images if they are provided
    if image:
        image.save(os.path.join(os.path.dirname(output_path), f'image_{n_image}/original_image.png'))
    if gradcam_original:
        gradcam_original.save(os.path.join(os.path.dirname(output_path), f'image_{n_image}/gradcam_original.png'))
    if gradcam_memo:
        gradcam_memo.save(os.path.join(os.path.dirname(output_path), f'image_{n_image}/gradcam_memo.png'))
    if gradcam_memo_plus:
        gradcam_memo_plus.save(os.path.join(os.path.dirname(output_path), f'image_{n_image}/gradcam_memo_plus.png'))

    # Move the output images to the specified path
    shutil.move('/home/sagemaker-user/DeepLearning_project/output.jpg', f'/home/sagemaker-user/DeepLearning_project/Results/test1/image_{n_image}/output.jpg')
    


def create_gradcam(image, model, target_layers):

    input_tensor = transform(Image.fromarray((image * 255).astype(np.uint8))).unsqueeze(0).to(device)
    
    cam_algorithm = GradCam(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale_cam = cam_algorithm(input_tensor=input_tensor)[0]

    # Save the grayscale CAM image
    heatmap = (np.uint8(255 * grayscale_cam))

    

    # Threshold the heatmap to identify the most important region
    threshold_value = np.max(heatmap) * 1
    important_region = np.where(heatmap == threshold_value)

    # Calculate the centroid of the important region
    centroid_x = np.mean(important_region[1])
    centroid_y = np.mean(important_region[0])
    centroid = (int(centroid_x), int(centroid_y))


    cam_image = show_cam_on_image(image, grayscale_cam)
    return Image.fromarray(cam_image), centroid, important_region

def summation(lst):
    ret = 0
    for elem in lst:
        ret += elem[0]
    return ret


def main(colab=False):
    # Define paths and directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pathDatasetImagenetA = os.path.join(current_dir, "datasets/imagenet-a")
    checkpoint_path = os.path.join(current_dir, "weights/sam_vit_b_01ec64.pth")
    output_csv_path = os.path.join(current_dir, "Results/test2.csv")
    
    # List to store augmentation results
    total_aug = []

    # Load dataset
    _, test_data = get_dataset(batch_size=1, img_root=pathDatasetImagenetA)

    # Number of augmentations
    num_aug = 8

    # Initialize ResNet50 model and save initial weights
    # model = ModelVitb16.to(devsegment_original_cropice)
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
        # print(type(image))
        # tryn = np.float32(image) / 255
        # print(type(tryn))

        # Test the model before applying MEMO
        correct_before_memo.append(test_model(image=image, target=target, model=model))

        gradcam_initial, centroid, regions = create_gradcam(np.float32(image) / 255, model, target_layers)

        # Tune the model using MEMO
        augmentation, name_aug = tune_model(image=image, model=model, mask_generator=mask_generator, optimizer=optimizer, cost_function=marginal_entropy, num_aug=num_aug, flag_memo_plus=False, centroid=centroid)

        # Test the model after applying MEMO
        correct_after_memo.append(test_model(image=image, target=target, model=model))

        gradcam_memo, centroid, regions= create_gradcam(np.float32(image) / 255, model, target_layers)


        # Reset model to initial weights before each iteration
        # model.load_state_dict(torch.load(initial_weights_path))
        # model.eval()
        
        # Tune the model using MEMO
        segmentation, name_aug_plus = tune_model(image=image, model=model, mask_generator=mask_generator, optimizer=optimizer, cost_function=marginal_entropy, num_aug=num_aug, flag_memo_plus=True, centroid=centroid)

        # Test the model after applying MEMO
        correct_after_memo_plus.append(test_model(image=image, target=target, model=model))
        
        gradcam_memo_plus,_,_ = create_gradcam(np.float32(image) / 255, model, target_layers)

        # accuracy
        accuracy_before_memo = np.mean(correct_before_memo)*100
        accuracy_after_memo = np.mean(correct_after_memo)*100
        accuracy_after_memo_plus = np.mean(correct_after_memo_plus)*100


        accuracy_before_memo = (summation(correct_before_memo)/len(correct_before_memo))*100
        accuracy_after_memo = (summation(correct_after_memo)/len(correct_after_memo))*100
        accuracy_after_memo_plus = (summation(correct_after_memo_plus)/len(correct_after_memo))*100



        accuracy_classes[f"{target}_before_MEMO"]["prediction"].append(correct_before_memo[i])
        accuracy_classes[f"{target}_after_MEMO"]["prediction"].append(correct_after_memo[i])
        accuracy_classes[f"{target}_after_MEMO_PLUS"]["prediction"].append(correct_after_memo_plus[i])

        accuracy_classes[f"{target}_after_MEMO"]["augmentation"].append(name_aug)
        accuracy_classes[f"{target}_after_MEMO_PLUS"]["augmentation"].append(name_aug_plus)

        # Save the augmentation results
        # save_report_image(augmentation = augmentation, output_path = os.path.join(current_dir, f"Results/{i}"))
        save_report_image(image=image, augmentation=augmentation, 
                          segmentation=segmentation, gradcam_original= gradcam_initial, 
                          gradcam_memo=gradcam_memo, gradcam_memo_plus=gradcam_memo_plus, 
                          output_path = os.path.join(current_dir, f"Results/test1"), n_image = i)
        
        # Salvataggio dei dati di accuratezza in un file CSV ogni 100 epoche
        if( i % 50 == 0):
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
