import cv2
import torch
import numpy as np
from torchvision import models
import torchvision.transforms as T
from timm import create_model
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

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model.eval()
        self.target_layers = target_layers
        self.activations = []
        self.gradients = []
        self.handles = []
        self.reshape_transform = reshape_transform
        for target_layer in self.target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))
        
    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
    
    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        #print(grad_output)
        # print(grad)
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        # print(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients
    
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

class GradCam:
    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model.eval()
        self.target_layers = target_layers
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)
        self.reshape_transform = reshape_transform

    def forward(self, input_tensor, targets=None):
        output = self.activations_and_grads(input_tensor)
        target_category = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[:, target_category].sum()
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, target_category)
        return self.aggregate_multi_layers(cam_per_layer)

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result, target_size=(224, 224))
    
    
    def scale_cam_image(self, cam, target_size=None):
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-7)
        if target_size:
            cam = [cv2.resize(np.float32(img), target_size) for img in cam]
        return np.float32(cam)
    
    def get_cam_image(self, activations, grads):
        weights = np.mean(grads, axis=(2, 3))
        weighted_activations = weights[:, :, None, None] * activations if len(activations.shape) == 4 else weights[:, :, None, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    def compute_cam_per_layer(self, input_tensor, target_category):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = (input_tensor.size(2), input_tensor.size(3))

        cam_per_target_layer = []
        for i, target_layer in enumerate(self.target_layers):
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads = grads_list[i] if i < len(grads_list) else None

            cam = self.get_cam_image(layer_activations, layer_grads)
            cam = np.maximum(cam, 0)
            cam_per_target_layer.append(self.scale_cam_image(cam, target_size)[:, None, :])

        return cam_per_target_layer

    def __call__(self, input_tensor, targets=None, aug_smooth=False):
        return self.forward(input_tensor, targets)

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == '__main__':
    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    model = ModelVitb16().to(device)
    print(model)
    # target_layers = [model.layer4]
    reshape_transform = reshape_transform
    # model = create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
    # print(model)
    target_layers = [model.vitb.encoder.layers[-1].ln_1]
    path_img = "/home/disi/pytorch-grad-cam/img.jpg"  # Ensure this path is correct

    rgb_img = cv2.imread(path_img)
    if rgb_img is None:
        raise FileNotFoundError(f"Image not found at path: {path_img}")

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = transformGradCam(Image.fromarray((rgb_img * 255).astype(np.uint8))).unsqueeze(0).to(device)
    
    cam_algorithm = GradCam(model=model, target_layers=target_layers, 
                            reshape_transform=reshape_transform)
    grayscale_cam = cam_algorithm(input_tensor=input_tensor)[0]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite('/home/disi/DeepLearning_project/GRADCAM_output.jpg', cam_image)
