from utils import *
from model import *
from functions import *

from gradcam import *
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from model import *

device = 'cuda' if torch.cuda.is_available() else 'mps'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transformGradCam = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean, std)
])


if __name__ == '__main__':
    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    model = ModelVitb16().to(device)
    # target_layers = [model.layer4]
    reshape_transform = reshape_transform
    # model = create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
    # print(model)
    target_layers = [model.vitb.encoder.layers[-1].ln_1]
    path_img = "/home/disi/DeepLearning_project/Results/test1/image_1/original_image.png"  # Ensure this path is correct

    rgb_img = cv2.imread(path_img)
    if rgb_img is None:
        raise FileNotFoundError(f"Image not found at path: {path_img}")

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = transform(Image.fromarray((rgb_img * 255).astype(np.uint8))).unsqueeze(0).to(device)
    
    cam_algorithm = GradCam(model=model, target_layers=target_layers, 
                            reshape_transform=reshape_transform)
    grayscale_cam = cam_algorithm(input_tensor=input_tensor)[0]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite('/home/disi/DeepLearning_project/GRADCAM_output.jpg', cam_image)
