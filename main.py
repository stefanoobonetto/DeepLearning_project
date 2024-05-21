from utils import *
from model import *
from functions import *


from tqdm import tqdm
import torch.optim as optim
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def main(colab=False):


    
    if colab:
        pathDatasetImagenetA = "/datasets/imagenet-a"
    else:
        pathDatasetImagenetA = "/home/sagemaker-user/DeepLearning_project/datasets/imagenet-a"
        #apt install libgl1-mesa-glx

    # # Instantiates dataloaders
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
    
    total_aug = []

    #take data
    _, test_data = get_dataset(batch_size=1, img_root=pathDatasetImagenetA)
    #number of augmentations
    num_aug = 5
    #model
    model = ModelResNet().to(device)
    #optimizer
    lr = 0.00025
    weight_decay = 0.0
    optimizer = optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


    # Scegli il tipo di modello, ad esempio 'vit_b' per il modello ViT-B
    model_type = "vit_b"
    # Percorso al checkpoint scaricato
    checkpoint_path = "/home/sagemaker-user/DeepLearning_project/weights/sam_vit_b_01ec64.pth"
    # Registra e carica il modello
    segmentation_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    # Crea il generatore di maschere automatico
    mask_generator = SamAutomaticMaskGenerator(segmentation_model)

    correct = []
    # with tqdm(total=len(test_data)) as pbar:
    with tqdm(total=100) as pbar:
        for i in range(100):
            model = ModelResNet().to(device)

            image, target = test_data[i]
            names = tune_model(image=image, model=model, mask_generator=mask_generator , optimizer=optimizer, cost_function=marginal_entropy, num_aug=num_aug)
            correct.append(test_model(image=image, target=target, model=model))

            accuracy = np.mean(correct)*100
            pbar.set_description(f'MEMO accuracy: {np.mean(correct)*100:.2f}%')
            pbar.update(1)

            total_aug.append(names)

    for i, item in enumerate(total_aug):
        item = item[1:]
        #print("\nAugmentations applied at image " + str(i) + " ---> " + str(item))

    print(f'\nFinal MEMO accuracy: {np.mean(correct)*100:.2f}%')
    


if __name__ == "__main__":
    colab = False
    main(colab=colab)
