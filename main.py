from utils import *
from model import *
from functions import *

import os
from tqdm import tqdm
import torch.optim as optim
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def main(colab=False):
   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pathDatasetImagenetA = f"{current_dir}/datasets/imagenet-a"
    checkpoint_path = f"{current_dir}/weights/sam_vit_b_01ec64.pth"
    #apt install libgl1-mesa-glx
    
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
    # Registra e carica il modello
    segmentation_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    # Crea il generatore di maschere automatico
    mask_generator = SamAutomaticMaskGenerator(segmentation_model)

    correct_after_memo = []
    correct_before_memo = []
    # with tqdm(total=len(test_data)) as pbar:
    with tqdm(total=100) as pbar:
        for i in range(100):
            model = ModelResNet().to(device)
            image, target = test_data[i]


            correct_before_memo.append(test_model(image=image, target=target, model=model))
            names = tune_model(image=image, model=model, mask_generator=mask_generator , optimizer=optimizer, cost_function=marginal_entropy, num_aug=num_aug)
            correct_after_memo.append(test_model(image=image, target=target, model=model))

            pbar.set_description(f'Before MEMO accuracy: {np.mean(correct_before_memo)*100:.2f}%  after MEMO accuracy: {np.mean(correct_after_memo)*100:.2f}%')
            pbar.update(1)

            total_aug.append(names)

    for i, item in enumerate(total_aug):
        item = item[1:]
        #print("\nAugmentations applied at image " + str(i) + " ---> " + str(item))

    print(f'Before MEMO accuracy: {np.mean(correct_before_memo)*100:.2f}%  after MEMO accuracy: {np.mean(correct_after_memo)*100:.2f}%')
    


if __name__ == "__main__":
    colab = False
    main(colab=colab)
